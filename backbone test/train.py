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

train_data = torchvision.datasets.ImageNet(split="train",transforms=Compose([Resize(512,512),ToTensorV2()]),download=True)
val_data = torchvision.datasets.ImageNet(split="val",transforms=Compose([Resize(512,512),ToTensorV2()]),download=True)




if __name__ == "__main__":
    print("Sanity Check")
   

    train_load = DataLoader(train_data, batch_size=16, num_workers=4, shuffle=True)
    val_load = DataLoader(val_data, batch_size=16, num_workers=4, shuffle=False)

    image,labels = next(iter(train_load))
    val_image,val_labels = next(iter(val_load))
    print("Train:{}\n".format(image.shape),
          "Val:{}".format(val_image.shape))
