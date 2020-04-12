import torch
import torch.nn as nn 
import torch.nn.functional as F  
import numpy as np  
from collections import OrderedDict
from colorama import Fore
import os

WEIGHTS_PATH = os.path.dirname(os.path.abspath(__file__))+"/weights/"

class FlattenTensorCustom(nn.Module):

    def __init__(self):
        
        super(FlattenTensorCustom, self).__init__()

    def forward(self, x):
        """
        Input:
        
        A Tensor x of shape [batch_no, c, h, w]

        Output:

        A Tensor x of shape [batch_no, c*h*w]        
        """

        x = x.transpose(3,2).contiguous() #wierd fix
        
        return x.view(x.size(0), -1)



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
        print("PNet Summary:")
        print(self.features)    
        print(self.conv4_1)
        print(self.conv4_2)

    def forward(self, x):
        x = self.features(x)
        probs = nn.Softmax(self.conv4_1(x), dim=1) #ERROR PRONE  #holds probilities and box preds respec.
        boxes = self.conv4_2(x)

        return probs, boxes   

class RNet(nn.Module):

     
    def __init__(self):
         
        super(RNet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 28, 3, 1)),
            ("prelu1", nn.PReLU(28)),
            ("pool1", nn.MaxPool2d(3, 2, ceil_mode=True)),

            ("conv2", nn.Conv2d(28, 48, 3, 1)),
            ("prelu2", nn.PReLU(48)),
            ("pool2", nn.MaxPool2d(3, 2, ceil_mode=True)),

            ("conv3", nn.Conv2d(48, 64, 2, 1)),

            ("flatten", FlattenTensorCustom()),
            ("conv4", nn.Linear(576, 128)),
            ("prelu4", nn.PReLU(128)),      
        ]))       

        self.conv5_1 = nn.Linear(128, 2) #boxes
        self.conv5_2 = nn.Linear(128, 4)

        try:
            self.weights = np.load(WEIGHTS_PATH+"rnet.npy", allow_pickle=True)[()]
            for idx, wts in self.named_parameters():
                wts.data = torch.FloatTensor(self.weights[idx])
        except Exception as err:

            print(Fore.RED+"ERROR: at loading onet weights: {}".format(err)+Fore.RESET)
            exit()
    
    def summary(self):
        print("RNet Summary:")
        print(self.features)
        print("\n")
        print(self.conv5_1)
        print(self.conv5_2)

    def forward(x):
        x = self.features(x)
        prods = nn.Softmax(self.conv5_1(x), dim=1)
        boxes = self.conv5_2(x)
        return probs, boxes    


