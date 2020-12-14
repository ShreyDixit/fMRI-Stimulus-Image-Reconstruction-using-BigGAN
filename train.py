from fastai.basic_data import *
from fastai.basic_train import *
from fastai.train import *

from models import fMRI_to_BigGAN
from utils import Create_DLs, load_freeze_gen, mse_image

import torch.nn as nn



dl_train, dl_test, num_voxels = Create_DLs("data/", "Subject1", 64)
data = DataBunch(dl_train, dl_test)

model = fMRI_to_BigGAN(num_voxels, z_dim=120)
load_freeze_gen(model, path="models/100k/G.pth")

loss_fn = VGG19_Loss(5, 4)
learn = Learner(data, model, loss_func = loss_fn, metrics = mse_image)
learn.fit_one_cycle(5, 3e-2)