import os
from PIL import Image
from scipy.io import loadmat
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from functools import partial

import torch
import torch.nn as nn
import torchvision
from torch.nn.functional import mse_loss
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class VGG19_Loss():
    """
    A truncated VGG19 network, such that its output is the 'feature map obtained by the j-th convolution (after activation)
    before the i-th maxpooling layer within the VGG19 network', as defined in the paper.
    Used to calculate the MSE loss in this VGG feature-space, i.e. the VGG loss.
    """
    def __init__(self, i, j):
        vgg19 = torchvision.models.vgg19(pretrained=True)
        
        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        # Iterate through the convolutional section ("features") of the VGG19
        for layer in vgg19.features.children():
            truncate_at += 1

            # Count the number of maxpool layers and the convolutional layers after each maxpool
            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            # Break if we reach the jth convolution after the (i - 1)th maxpool
            if maxpool_counter == i - 1 and conv_counter == j:
                break

        # Check if conditions were satisfied
        assert maxpool_counter == i - 1 and conv_counter == j, "One or both of i=%d and j=%d are not valid choices for the VGG19!" % (
            i, j)

        # Truncate to the jth convolution (+ activation) before the ith maxpool layer
        self.truncated_vgg19 = nn.Sequential(*list(vgg19.features.children())[:truncate_at + 1])
        self.truncated_vgg19.eval()
    
    def __call__(self, input, target):
        input_features = self.truncated_vgg19(input)
        target_features = self.truncated_vgg19(target)
        return mse_loss(input_features, target_features) #+ mse_loss(input, target)
    

class kamitani_data_handler():
    """Generate batches for FMRI prediction
    frames_back - how many video frames to take before FMRI frame
    frames_forward - how many video frames to take after FMRI frame
    """

    def __init__(self, matlab_file ,test_img_csv = "data/images/image_test_id.csv",train_img_csv = "data/images/image_training_id.csv",voxel_spacing =3,log = 0 ):
        mat = loadmat(matlab_file)
        self.data = mat['dataSet'][:,3:]
        self.sample_meta = mat['dataSet'][:,:3]
        meta = mat['metaData']


        self.meta_keys = list(l[0] for l in meta[0][0][0][0])
        self.meta_desc = list(l[0] for l in meta[0][0][1][0])
        self.voxel_meta = np.nan_to_num(meta[0][0][2][:,3:])
        test_img_df = pd.read_csv(test_img_csv, header=None)
        train_img_df =pd.read_csv(train_img_csv, header=None)
        self.test_img_id = test_img_df[0].values
        self.train_img_id = train_img_df[0].values
        self.sample_type = {'train':1 , 'test':2 , 'test_imagine' : 3}
        self.voxel_spacing = voxel_spacing

        self.log = log
        
    def get_meta_field(self,field = 'DataType'):
        index = self.meta_keys.index(field)
        if(index <3): # 3 first keys are sample meta
            return self.sample_meta[:,index]
        return self.voxel_meta[index]
        
    def print_meta_desc(self):
        print(self.meta_desc)
        
    def get_labels(self, imag_data = 0,test_run_list = None):
        le = preprocessing.LabelEncoder()

        img_ids = self.get_meta_field('Label')
        type = self.get_meta_field('DataType')
        train = (type == self.sample_type['train'])
        test = (type == self.sample_type['test'])
        imag = (type == self.sample_type['test_imagine'])

        img_ids_train = img_ids[train]
        img_ids_test = img_ids[test]
        img_ids_imag = img_ids[imag]


        train_labels  = []
        test_labels  =  []
        imag_labels = []
        for id in img_ids_test:
            idx = (np.abs(id - self.test_img_id)).argmin()
            test_labels.append(idx)

        for id in img_ids_train:
            idx = (np.abs(id - self.train_img_id)).argmin()
            train_labels.append(idx)

        for id in img_ids_imag:
            idx = (np.abs(id - self.test_img_id)).argmin()
            imag_labels.append(idx)

        if (test_run_list is not None):
            run = self.get_meta_field('Run')
            test = (self.get_meta_field('DataType') == 2).astype(bool)
            run = run[test]

            select = np.in1d(run, test_run_list)
            test_labels = test_labels[select]

        #imag_labels = le.fit_transform(img_ids_imag)
        if(imag_data):
            return np.array(train_labels), np.array(test_labels), np.array(imag_labels)
        else:
            return np.array(train_labels),np.array(test_labels)
    
    def get_data(self,normalize =1 ,roi = 'ROI_VC',imag_data = 0,test_run_list = None):   # normalize 0-no, 1- per run , 2- train/test seperatly
        type = self.get_meta_field('DataType')
        train = (type == self.sample_type['train'])
        test = (type == self.sample_type['test'])
        test_imag = (type == self.sample_type['test_imagine'])
        test_all  = np.logical_or(test,test_imag)

        roi_select = self.get_meta_field(roi).astype(bool)
        data = self.data[:,roi_select]

        if(self.log ==1):
            data = np.log(1+np.abs(data))*np.sign(data)


        if(normalize==1):

            run = self.get_meta_field('Run').astype('int')-1
            num_runs = np.max(run)+1
            data_norm = np.zeros(data.shape)

            for r in range(num_runs):
                data_norm[r==run] = preprocessing.scale(data[r==run])
            train_data = data_norm[train]
            test_data  = data_norm[test]
            test_all = data_norm[test_all]
            test_imag = data_norm[test_imag]

        else:
            train_data = data[train]
            test_data  =  data[test]
            if(normalize==2):
                train_data = preprocessing.scale(train_data)
                test_data = preprocessing.scale(test_data)


        if(self.log ==2):
            train_data = np.log(1+np.abs(train_data))*np.sign(train_data)
            test_data = np.log(1+np.abs(test_data))*np.sign(test_data)
            train_data = preprocessing.scale(train_data)
            test_data = preprocessing.scale(test_data)



        test_labels =  self.get_labels()[1]
        imag_labels = self.get_labels(1)[2]
        num_labels = max(test_labels)+1
        test_data_avg = np.zeros([num_labels,test_data.shape[1]])
        test_imag_avg = np.zeros([num_labels,test_data.shape[1]])

        if(test_run_list is not None):
            run = self.get_meta_field('Run')
            test = (self.get_meta_field('DataType') == 2).astype(bool)
            run = run[test]

            select = np.in1d(run, test_run_list)
            test_data = test_data[select,:]
            test_labels = test_labels[select]

        for i in range(num_labels):
            test_data_avg[i] = np.mean(test_data[test_labels==i],axis=0)
            test_imag_avg[i] = np.mean(test_imag[imag_labels == i], axis=0)
        if(imag_data):
            return train_data, test_data, test_data_avg,test_imag,test_imag_avg

        else:
            return train_data, test_data, test_data_avg

    def get_voxel_loc(self):
        x = self.get_meta_field('voxel_x')
        y = self.get_meta_field('voxel_y')
        z = self.get_meta_field('voxel_z')
        dim = [int(x.max() -x.min()+1),int(y.max() -y.min()+1), int(z.max() -z.min()+1)]
        return [x,y,z] , dim

def calc_snr(y, y_avg, labels):
    sig = np.var(y_avg, axis=0)
    noise = 0
    for l in labels:
        noise += np.var(y[labels == l], axis=0)
    noise /= len(labels)
    return sig/noise    

def mse_image(input, target):
    return ((input-target)**2).mean() * (255*255)

class fMRI_DataSet(Dataset):
    def __init__(self, act_BOLD, path, img_names):
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.eval()
        self.act_BOLD = torch.from_numpy(act_BOLD)
        self.img_names = img_names
        self.path = path

        self.transforms = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img = Image.open(self.path + self.img_names[idx])
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = self.transforms(img)
        pred = torch.argmax(self.resnet(img[None]).squeeze())
        return (self.act_BOLD[idx].float(), pred), img.float()
    
def Create_DLs(path, subject, bs, apply_pca=True):
    kamitani_data = kamitani_data_handler(f"{path}{subject}.mat")
    
    train_act, test_act, test_act_avg = kamitani_data.get_data()
    train_labels, test_labels = kamitani_data.get_labels()
    snr = calc_snr(test_act, test_act_avg, test_labels)
    mask = (snr/snr.mean())>0.8
    
    train_act = train_act[:, mask]
    test_act = test_act[:, mask]
    
    if apply_pca:
        pca = PCA(0.95)
        pca.fit(np.concatenate((train_act, test_act)))
        train_act = pca.transform(train_act)
        test_act = pca.transform(test_act)
    
    num_voxels = train_act.shape[1]
    
    test_df = pd.read_csv(path + "images/image_test_id.csv", header = None)
    train_df = pd.read_csv(path + "images/image_training_id.csv", header = None)
    
    test_image_names = list(test_df.iloc[test_labels, 1])
    train_image_names = list(train_df.iloc[train_labels, 1])
    
    test_DS = fMRI_DataSet(test_act, path + "images/test/", test_image_names)
    train_DS = fMRI_DataSet(train_act, path + "images/training/", train_image_names)
    
    return DataLoader(train_DS, batch_size=bs), DataLoader(test_DS, batch_size=bs), num_voxels
    
def load_freeze_gen(model, path):
    model.BigGAN.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    for parameter in model.BigGAN.parameters():
        parameter.requires_grad = False