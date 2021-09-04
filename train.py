from pandas.core.algorithms import mode

import torch
import torch.nn as nn
from albumentations import Compose,Resize
from albumentations.pytorch import ToTensorV2
import time
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast,GradScaler
import os
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from callbacks import EarlyStopping
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
import torch.nn.functional as F
import random


from build_model import Deformed_Darknet53

torch.manual_seed(2021)
np.random.seed(2021)
random.seed(2021)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TOTAL_EPOCHS = 100
scaler = GradScaler()
writer = SummaryWriter()
early_stop = EarlyStopping()



print("***** Loading the Model in {} *****".format(DEVICE))

Model = Deformed_Darknet53().to(DEVICE)

print("Model Shipped to {}".format(DEVICE))

data = pd.read_csv("data.csv")

train_loss_fn = nn.BCEWithLogitsLoss()
val_loss_fn = nn.BCEWithLogitsLoss()

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
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    #start_time = time.time()
    pbar = tqdm(enumerate(dataloader))
    for i,(img,label) in pbar:
        optim.zero_grad()

        img = img.to(DEVICE).float()
        label = label.to(DEVICE).float()
        
        #LOAD_TIME = time.time() - start_time

        with autocast():
            yhat = model(img)
            #Loss Calculation
            train_loss = loss_fn(input = yhat.flatten(), target = label)
        
        out = (yhat.flatten().sigmoid() > 0.5).float()
        correct = (label == out).float().sum()

        scaler.scale(train_loss).backward()
        scaler.step(optim)
        scaler.update()

        
        epoch_loss += train_loss.item()
        epoch_acc += correct.item() / out.shape[0]

    train_epoch_loss = epoch_loss / len(dataloader)
    train_epoch_acc = epoch_acc / len(dataloader)

    writer.add_scalar("Training_Loss",train_epoch_loss,epoch)
    writer.add_scalar("Training_Acc",train_epoch_acc,epoch)
        
    #print(f"Epoch:{epoch}/{TOTAL_EPOCHS} Epoch Loss:{epoch_loss / len(dataloader):.4f} Epoch Acc:{epoch_acc / len(dataloader):.4f}")
    
    return train_epoch_loss,train_epoch_acc

def val_loop(epoch,dataloader,model,loss_fn,device = DEVICE):
    model.eval()
    val_epoch_loss = 0
    val_epoch_acc = 0
    pbar = tqdm(enumerate(dataloader))

    with torch.no_grad():
        for i,(img,label) in pbar:
            img = img.to(device)
            label = label.to(device)

            yhat = model(img).flatten()
            val_loss = loss_fn(input=yhat,target=label)

            out = (yhat>0.5).float()
            correct = (yhat == label).float().sum()

            val_epoch_loss += val_loss.item()
            val_epoch_acc += correct.item() / out.shape[0]

        val_lossd = val_epoch_loss / len(dataloader)
        val_accd = val_epoch_acc / len(dataloader)
        
        writer.add_scalar("Val_Loss",val_lossd,epoch)
        writer.add_scalar("Val_Acc",val_accd/len(dataloader),epoch)

        return val_lossd,val_accd






        
    


if __name__ == "__main__":

    train_per_epoch_loss,train_per_epoch_acc = [],[]
    val_per_epoch_loss,val_per_epoch_acc = [],[]
    train = dog_cat(data,transforms=Compose([Resize(256,256),ToTensorV2()]))
    val = dog_cat(data,mode='val',transforms=Compose([Resize(256,256),ToTensorV2()]))

    train_load = DataLoader(train,batch_size=16,num_workers=4)
    val_load = DataLoader(val,batch_size=16,num_workers=4)

    for e in range(TOTAL_EPOCHS):
        train_loss,train_acc = train_loop(e,train_load,Model,train_loss_fn,optim)
        val_loss,val_acc = val_loop(e,val_load,Model,val_loss_fn)
        train_per_epoch_loss.append(train_loss)
        train_per_epoch_acc.append(train_acc)
        val_per_epoch_loss.append(val_loss)
        val_per_epoch_acc.append(val_acc)
        early_stop(val_loss)
        if early_stop.early_stop:
            break


        






