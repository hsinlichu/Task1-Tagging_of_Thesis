import torch.nn as nn
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def bce_loss(output, target):
    print(output,target)
    return nn.BCEWithLogitsLoss(output, target)
