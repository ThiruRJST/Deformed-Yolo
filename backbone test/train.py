from pandas.core.algorithms import mode
import torch
import torch.nn as nn
from albumentations import Compose,Resize
from albumentations.pytorch import ToTensorV2
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
import random

from sklearn.model_selection import StratifiedKFold

from build_model import Deformed_Darknet53

torch.manual_seed(2021)
np.random.seed(2021)
random.seed(2021)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print("***** Loading the Model in {} *****".format(DEVICE))

Model = Deformed_Darknet53().to(DEVICE)

print("Model Shipped to {}".format(DEVICE))

class dog_cat(Dataset):
    def __init__(self,df,mode="train",folds=0,transforms=None):
        super(dog_cat,self).__init__()
        self.df = df
        self.mode = mode
        self.folds = folds
        self.transforms = transforms

        if self.mode == "train":
            self.data = self.df[self.df.folds != self.folds].reset_index(drop=True)
            
        else:
            self.data = self.df[self.df.folds == self.folds].reset_index(drop=True)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,idx):

        img = cv2.imread(self.data.loc[idx,"Paths"])
        label = self.data.loc[idx,'Labels']

        if self.transforms is not None:
            image = self.transforms(image=img)['image']

        
        return image,label







