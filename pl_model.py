r'''
Author       : PiKaChu_wcg
Date         : 2022-09-19 19:58:19
LastEditors  : PiKachu_wcg
LastEditTime : 2022-09-20 13:42:35
FilePath     : /model_new/pl_model.py
'''
import pytorch_lightning as pl
from gate_crf_loss import ModelLossSemsegGatedCRF
from unet.Unet_model_tent import UNet_tent
from loss import  DiceLoss, entropy_loss
import os
import torch
import torch.optim as optim
from medpy.metric import binary
import numpy as np
from torch import nn

class Tent(pl.LightningModule):
    def __init__(self, args):
        self.args=args
        self.model = UNet_tent(n_channels=3, n_classes=args.S_num_classes-1)#unet's n_classes should be 1 when actually 2 classes
        self.entropymin_loss=entropy_loss()
        self.load_model()
    def load_model(self, checkpoint) -> None:
        if os.path.isfile(self.args.resume):
            print('loading checkpoint:{}'.format(self.args.resume))
            checkpoint = torch.load(self.args.resume)
            self.model.load_state_dict(checkpoint['state_dict_S'])
            print('load successfully!')
        else:
            print(f'{self.args.resume} is not a file')
            print('skip load checkpoint')
    def configure_optimizers(self):
        for idx in self.model.named_modules():
            if 'bn' not in idx[0]:
                for param in idx[1].parameters():
                    param.requires_grad = False
            if 'bn' in idx[0]:
                for param in idx[1].parameters():
                    param.requires_grad = True
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.learning_rate_S)
        return optimizer
    def forward(self,image):
        return self.model(image)
    def training_step(self, batch, batch_idx):
        image,_=batch
        pred_mask=self(image)
        loss=self.entropymin_loss(pred_mask)
        return loss

    def validation_step(self, batch,batch_idx) :
        image,mask,name=batch
        output=self(image)
        probs=torch.sigmoid(output)
        probs=(probs>0.5).long()
        mask=mask
        return {'pred':probs.cpu().numpy(),'mask':mask.cpu().numpy()}
    def validation_epoch_end(self, outputs):
        pred= np.stack([i['pred'] for i in outputs],axis=0)
        mask= np.stack([i['mask'] for i in outputs],axis=0)
        inter=pred*mask
        dice=2*inter.sum()/(pred.sum()+mask.sum())
        miou=inter.sum()/(pred.sum()+mask.sum()-inter.sum()+1e-6)
        assd = binary.assd(pred, mask, connectivity=1)
        hd95 = binary.hd95(pred, mask)
        self.log_dict(dict(dice=dice,miou=miou,assd=assd,hd95=hd95))
        



        

    
class ModifiedTent(Tent):
    def __init__(self,args):
        super(Tent,self).__init__(args)
        self.seg_loss = DiceLoss()
        self.mse_liss=nn.MSELoss()
        self.gatecrf_loss=ModelLossSemsegGatedCRF()
        self.loss_gatedcrf_kernels_desc = [{"weight": 1, "xy": 6, "rgb": 0.1}]
        self.loss_gatedcrf_radius = 5
        self.weight_crf = 0.1
    
    def configure_optimizers(self):
        for idx in self.model.named_modules():
            if 'bn' not in idx[0]:
                for param in idx[1].parameters():
                    param.requires_grad = False
            if 'bn' in idx[0]:
                for param in idx[1].parameters():
                    param.requires_grad = True
            if 'inc' in idx[0]:
                for param in idx[1].parameters():
                    param.requires_grad=True
        optimizer = optim.Adam(filter(
            lambda p: p.requires_grad, self.model.parameters()), lr=self.args.learning_rate_S)

    def training_step(self, batch, batch_idx):
        image, image_R = batch
        pred_mask = self(image)
        loss = self.entropymin_loss(pred_mask)
        pred_mask_soft=pred_mask_soft.softmax(1)
        _,s_tgt,_=torch.linalg.svd(pred_mask_soft)
        if weight_crf < 1:
            weight_crf = 0.05 * self.current_epoch
        loss_Entorpy = self.entropymin_loss(pred_mask)
        BNM_method_loss = -s_tgt.mean()
        pred_mask_R=self(image_R)
        pred_mask_RR=torch.rot90(pred_mask_R,-1,[2,3])
        loss_r=self.mse_loss(pred_mask_R,pred_mask_RR)
        out_gatedcrf = self.gatecrf_loss(
            pred_mask_soft,
            self.loss_gatedcrf_kernels_desc,
            self.loss_gatedcrf_radius,
            image,
            512,
            512,
        )["loss"]
        self.log_dict(
            dict(
                entropy=loss_Entorpy,
                BNM=BNM_method_loss,
                CRF=out_gatedcrf,
                CONSISTENT=loss_r
            )
        )
        loss = loss_Entorpy + weight_crf * out_gatedcrf + 0.5* BNM_method_loss + loss_r
        self.log("loss",loss)
        return loss
