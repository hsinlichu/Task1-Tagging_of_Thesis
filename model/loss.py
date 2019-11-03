import torch.nn as nn
import torch.nn.functional as F

def bce_loss(output, target):
    return nn.BCELoss()(output.double(), target.double())
