import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score

def compute_metrics(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    

    acc = (y_true == y_pred).mean()
    

    f1 = f1_score(y_true, y_pred, average='macro')
    
    return acc, f1

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']


class CustomLoss(nn.Module):
    def __init__(self, alpha=0.2):
        super(CustomLoss, self).__init__()
        self.alpha = alpha

    def forward(self, outputs, targets):
        
        ce_loss = F.cross_entropy(outputs, targets)
        smooth_l1_loss = F.smooth_l1_loss(outputs, F.one_hot(targets, num_classes=outputs.size(1)).float())
        
       
        loss = (1 - self.alpha) * ce_loss + self.alpha * smooth_l1_loss
        
        return loss