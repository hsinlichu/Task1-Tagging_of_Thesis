import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable

def bce_loss(output, target):
    return nn.BCELoss()(output.float(), target.float())


### Below are wilber loss
def nll_loss(output, target):
    return F.nll_loss(output, target)

def f1_loss(output, target):
    f1_loss = F1_Loss()
    return f1_loss(output,target)

def fcl_loss(output, target):#focal loss
    fcl_loss = Focal_Loss()
    return fcl_loss(output,target)

def my_loss(output,target):
    fcl_loss = Focal_Loss()
    f1_loss = F1_Loss()
    hm_loss = Hamming_Loss()
    return  0.5*f1_loss(output,target)+0.5*fcl_loss(output,target)

def Hingeloss(output,target):
    return nn.MultiLabelMarginLoss()(output,target)

def HammingLoss(output,target):
    hmloss = Hamming_Loss()
    return hmloss(output,target)


class F1_Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(F1_Loss, self).__init__()
        self.epsilon = epsilon
 
    def forward(self, output, target):
        probas = output
        #probas = nn.Sigmoid()(output)
        target = target.type(torch.cuda.FloatTensor)
        TP = (probas * target).sum(dim=1)
        precision = TP / (probas.sum(dim=1) + self.epsilon)
        recall = TP / (target.sum(dim=1) + self.epsilon)
        f1 = 2 * precision * recall / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()

class Focal_Loss(nn.Module):
    def __init__(self, gamma=2):
        super(Focal_Loss, self).__init__()
        self.gamma = 2
 
    def forward(self, output, target):
        ce = bce_loss(output, target)
        p = 1-10**(ce*(-1))
        par = p**self.gamma
        FL = par * ce
        return FL


class Hamming_Loss(nn.Module):
    def __init__(self,):
        super(Hamming_Loss,self).__init__()
    
    def forward(self, output, target):
        target = target.type(torch.cuda.FloatTensor)
        return (target * (1-output)+(1-target) * output).mean()
