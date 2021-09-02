import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from build_model import Deformed_Darknet53

torch.manual_seed(2021)
np.random.seed(2021)
random.seed(2021)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print("***** Loading the Model in {} *****".format(DEVICE))

Model = Deformed_Darknet53().to(DEVICE)




