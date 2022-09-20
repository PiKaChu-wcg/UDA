r'''
Author       : PiKaChu_wcg
Date         : 2022-09-20 12:59:11
LastEditors  : PiKachu_wcg
LastEditTime : 2022-09-20 13:41:56
FilePath     : /model_new/loss.py
'''
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, ignore=255):
        super().__init__()
        self.smooth = 1
        self.ignore = ignore

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        #print(preds.size())
        preds = preds[targets != self.ignore]
        targets = targets[targets != self.ignore]
        iflat = preds.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()

        #p = nn.functional.softmax(preds, dim=-1)

        return 1 - ((2. * intersection + self.smooth) /
                    (iflat.sum() + tflat.sum() + self.smooth))  #- 1 * torch.sum(p * nn.functional.log_softmax(preds, dim=-1))

class entropy_loss(nn.Module):
    def forward(self, p_logit):
        p = p_logit.sigmoid()
        out = -1 * p * p.log()
        out = out.sum() / (p_logit.size()[2] * p_logit.size()[3])
        return out