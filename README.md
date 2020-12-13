# fMRI-Stimulus-Image-Reconstruction-using-BigGAN

## Basic usage:
1. Download "Generic Object Decoding" dataset (by Kamitani Lab) and place the subject mat files in data folder
```
http://brainliner.jp/data/brainliner/Generic_Object_Decoding
```

2. Request for images used in this experiment from Kamitani Lab and place them in data folder
```
https://docs.google.com/forms/d/e/1FAIpQLSfuAF-tr4ZUBx2LvxavAjjEkqqUOj0VpqpeJNCe-IcdlqJekg/viewform
```

3. Download the pretrained BigGAN from place it in models folder
```
https://drive.google.com/file/d/1eXGjqs3Vh8W30NXmppWyl_eomXzrzF_s/view?usp=sharing
```

Train the model using
```
python train.py
```