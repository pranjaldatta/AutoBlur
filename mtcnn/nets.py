import torch
import torch.nn as nn 
import torch.nn.functional as F  
import numpy as np  
from collections import OrderedDict
from colorama import Fore
import os

WEIGHTS_PATH = os.path.dirname(os.path.abspath(__file__))+"/weights/"

class PNet(nn.Module):

    

    def __init__(self):

        super(PNet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            
            ("conv1", nn.Conv2d(3, 10, 3, 1)),
            ("prelu1", nn.PReLU(10)),
            ("pool1", nn.MaxPool2d(3,2,ceil_mode=True)),

            ("conv2", nn.Conv2d(10, 16, 3, 1)),
            ("prelu2", nn.PReLU(16)),

            ("conv3", nn.Conv2d(16, 32, 3, 1)),
            ("prelu3", nn.PReLU(32)),

        ]))

        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)
        
        try:
            self.weights = np.load(WEIGHTS_PATH+"pnet.npy", allow_pickle=True)[()]
            for idx, wts in self.named_parameters():
                wts.data = torch.FloatTensor(self.weights[idx])
        except Exception as err:
            print(Fore.RED+"ERROR: At Pnet Weight Init: {}".format(err)+Fore.RESET)
            exit()


    def summary(self):
        print(self.features)
        print("-"*50)
        print(self.conv4_1)
        print(self.conv4_2)

    def forward(self, x):
        x = self.features(x)
        probs = nn.Softmax(self.conv4_1(x))  #holds probilities and box preds respec.
        boxes = self.conv4_2(x)

        return probs, boxes   