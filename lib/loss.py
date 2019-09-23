import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    'Focal Loss - https://arxiv.org/abs/1708.02002'

    def __init__(self, alpha=0.25, gamma=2, type_='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.type_ = type_

    def forward(self, pred_logits, target):
        pred = pred_logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        #print(ce.sum().item())
        alpha = target * self.alpha + (1. - target) * (1. - self.alpha)
        pt = torch.where(target == 1,  pred, 1 - pred)
        loss = alpha * (1. - pt) ** self.gamma * ce
        #print(loss.sum().item())
        #print((loss.sum() / (target.float().sum() + 1)).item())
        #print(target.float().sum())
        #print(target.device, target.dtype)
        #print('----')
        if self.type_ == 'mean':
            return loss.mean()
        else:
            return loss.sum() / (target.float().sum() + 1)


class BalanceBCE(nn.Module):

    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def forward(self, pred_logits, target):
        pred = pred_logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        #print(ce.sum().item())
        loss_pos = (target * ce).mean() 
        loss_neg = ((1 - target) * ce).mean()
        loss = (self.ratio * loss_pos + loss_neg).mean()
        print('loss %.5f, loss_pos %.5f, loss_neg %.5f' % (loss.item(), loss_pos.item() , loss_neg.item()))
        #print(loss.sum().item())
        #print((loss.sum() / (target.float().sum() + 1)).item())
        #print(target.float().sum())
        #print(target.device, target.dtype)
        #print('----')
        return loss


class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_logits, target):
        iflat = pred_logits.sigmoid().view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection + self.smooth) /
            (iflat.sum() + tflat.sum() + self.smooth))
