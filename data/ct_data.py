import glob
import numpy as np
import torch, os, pdb
import random as rn
from torch.utils import data
from torchvision import transforms
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
import torch.multiprocessing
import os, glob, numpy as np
from os.path import join as osj
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import pandas as pd
    
def qz(x, step=10):
    x = x//step
    return int(x)
    
    
class DFD(torch.utils.data.Dataset):
    
    def __init__(self, istrain, seed=1027):
        super(DFD, self).__init__()

        self.mode        = istrain
        self.fns = []
        self.y   = []
        self.y2   = []
        
        df = pd.read_csv("../data/Train.csv", header=None)
        tlen=len(df)
        rn.seed(seed)
        
        idx = np.arange(tlen)
        rn.shuffle(idx)
        print(idx)
        len2 = int(tlen*0.9)
#         train_ind = 2728
        if istrain:
            self.fns = [os.path.join('../data/Train/', df[0][f]) for f in idx[:len2]]
            self.y = [qz(df[1][f]) for f in idx[:len2]]
            self.y2 = [qz(df[1][f], step=5) for f in idx[:len2]]
            self.score = [float(df[1][f]/100.0) for f in idx[:len2]]
        else:
            self.fns = [os.path.join('../data/Train/', df[0][f]) for f in idx[len2:]]
            self.y = [qz(df[1][f]) for f in idx[len2:]]
            self.y2 = [qz(df[1][f], step=5) for f in idx[len2:]]
            self.score = [float(df[1][f]/100.0) for f in idx[len2:]]
            
        self.train_transform = A.Compose([
                                A.Resize(440, 440),
#                                 A.CenterCrop(height=480, width=480),
                                A.RandomCrop(height=416, width=416),
                                A.HorizontalFlip(p=0.5),
#                                 A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
                                A.RandomBrightnessContrast(p=0.5),
#                                 A.RandomRotate90(p=0.5),
#                                 A.OneOf([
#                                     A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
#                                     A.GridDistortion(p=0.5),
#                                     A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)                  
#                                     ], p=0.5),
                                A.CLAHE(p=0.5),
                                A.Cutout (num_holes=2, max_h_size=32, max_w_size=32, fill_value=0, always_apply=False, p=0.5),
                                A.RandomGamma(p=0.5),
                                A.Normalize(
                                            mean=[0.49313889,0.49313889, 0.49313889],
                                            std=[0.36952964,0.36952964,0.36952964],
                                            ),
                                ToTensorV2()
                            ])
        
        self.val_transform = A.Compose([
                                                A.Resize(512, 512),
                                                A.CenterCrop(height=416, width=416),
                                                
                                                A.Normalize(
                                                            mean=[0.49313889,0.49313889, 0.49313889],
                                                            std=[0.36952964,0.36952964,0.36952964],
                                                            ),
                                                ToTensorV2()
                                            ])
        

        self.n_images = len(self.fns)
        print('The # of images is:', self.n_images, 'on', self.mode, 'mode!')
        
        
    def __getitem__(self, index):
        fn = self.fns[index]
        lab = self.y[index]
        lab2 = self.y2[index]
        score=self.score[index]
        
        im = cv2.imread(fn)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        if self.mode:
            
            im = self.train_transform(image=im)['image']
            
            
            return im,  score
        else:

            im = self.val_transform(image=im)['image']
            
            return im, score

    def __len__(self):
        return self.n_images