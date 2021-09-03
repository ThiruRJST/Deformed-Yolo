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
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
import torch.nn.functional as F
import random

from sklearn.model_selection import StratifiedKFold

from build_model import Deformed_Darknet53

torch.manual_seed(2021)
np.random.seed(2021)
random.seed(2021)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TOTAL_EPOCHS = 100
writer = SummaryWriter()


print("***** Loading the Model in {} *****".format(DEVICE))

Model = Deformed_Darknet53().to(DEVICE)

print("Model Shipped to {}".format(DEVICE))

data = pd.read_csv("data.csv")

loss_fn = F.binary_cross_entropy()

optim = torch.optim.Adam(Model.parameters())

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


def train_loop(epoch,dataloader,model,loss_fn,optim,device=DEVICE):
    epoch_loss = 0
    pbar = tqdm(enumerate(dataloader))
    for i,(img,label) in pbar:
        img = img.to(DEVICE)
        label = label.to(DEVICE)
        yhat = model(img)
        #Loss Calculation
        train_loss = loss_fn(input = yhat, target = label)

        #Reset Optimizer
        optim.zero_grad()
        train_loss.backward()
        optim.step()

        
        epoch_loss += train_loss.item()
    print(f"Epoch:{epoch}/{TOTAL_EPOCHS} Epoch Loss:{epoch_loss / len(dataloader):.5f}")
        
    


if __name__ == "__main__":

    train = dog_cat(data,transforms=Compose([Resize(512,512),ToTensorV2()]))
    val = dog_cat(data,mode='val',transforms=Compose([Resize(512,512),ToTensorV2()]))

    train_load = DataLoader(train,batch_size=16,num_workers=4)
    val_load = DataLoader(val,batch_size=16,num_workers=4)

    images,labels = next(iter(train_load))

    print(images.shape,labels.shape)



