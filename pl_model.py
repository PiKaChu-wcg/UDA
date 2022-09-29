r'''
Author       : PiKaChu_wcg
Date         : 2022-09-19 19:58:19
LastEditors  : PiKachu_wcg
LastEditTime : 2022-09-28 22:49:10
FilePath     : /model_new/pl_model.py
'''
from lib2to3.pytree import Base
import pytorch_lightning as pl
from gate_crf_loss import ModelLossSemsegGatedCRF
from unet.Unet_model_tent import UNet_tent
from Loss import DiceLoss, entropy_loss
import os
import torch
import torch.optim as optim
from medpy.metric import binary
import numpy as np
from torch import nn
from PIL import Image
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
from monai.metrics.surface_distance import SurfaceDistanceMetric


class BaseModel(pl.LightningModule):
    def __init__(self,args):
        super(BaseModel,self).__init__()
        self.assd=SurfaceDistanceMetric(symmetric=True)
        self.hd95=HausdorffDistanceMetric(percentile=95,get_not_nans=True)
        self.args=args
    def forward(self,image):
        return self.model(image)
    def validation_step(self, batch,batch_idx) :
        image,mask,_=batch
        output=self(image)
        probs=torch.sigmoid(output)
        probs=(probs>0.5).long()
        assd=self.assd(probs,mask)
        hd95=self.hd95(probs,mask)
        assd[assd>1000]=1000
        hd95[hd95>1000]=1000
        hd95[hd95<0]=0
        return {'pred':probs.cpu().numpy(),'mask':mask.cpu().numpy(),'assd':assd,'hd95':hd95}
    def validation_epoch_end(self, outputs):
        pred= np.concatenate([i['pred'] for i in outputs],axis=0)
        mask= np.concatenate([i['mask'] for i in outputs],axis=0)
        inter=pred*mask
        dice=2*inter.sum()/(pred.sum()+mask.sum())
        miou=inter.sum()/(pred.sum()+mask.sum()-inter.sum()+1e-6)
        assd=torch.concat([i['assd'] for i in outputs],dim=0).mean()
        hd95=torch.concat([i['hd95'] for i in outputs],dim=0).mean()
        self.log_dict(dict(miou=miou,assd=assd,hd95=hd95),sync_dist=True)
        # self.log('miou',miou,prog_bar=True)
        self.log('dice',dice,prog_bar=True,sync_dist=True)   

class Pretrain(BaseModel):
    def __init__(self,args):
        super(Pretrain,self).__init__(args)
        self.model=UNet_tent(n_channels=3, n_classes=1)
        self.loss=DiceLoss()

    def training_step(self, batch,batch_idx) :
        img,mask,_=batch
        pred=self(img)
        loss=self.loss(pred,mask)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(),lr=self.args.learning_rate_S)


class Tent(BaseModel):
    def __init__(self, args):
        """args use the params including resume(the model checkpoint), learning_rate_S(the learning rate)

        Args:
            args (_type_): _description_
        """
        super(Tent,self).__init__(args)
        self.model = UNet_tent(n_channels=3, n_classes=1)#unet's n_classes should be 1 when actually 2 classes
        self.entropymin_loss=entropy_loss()
        self.load_model()
    def load_model(self):
        if os.path.isfile(self.args.resume):
            print('loading checkpoint:{}'.format(self.args.resume))
            checkpoint = torch.load(self.args.resume)
            self.load_state_dict(checkpoint['state_dict'])
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

    def training_step(self, batch, batch_idx):
        image,_,_=batch
        pred_mask=self(image)
        loss=self.entropymin_loss(pred_mask)
        return loss

        
    def test_step(self, batch,batch_idx) :
        image,mask,name=batch
        image,mask,_=batch
        output=self(image)
        probs=torch.sigmoid(output)
        probs=(probs>0.5).long().cpu().numpy()
        self.save_img(probs,mask,name)

        return {'pred':probs,'mask':mask.cpu().numpy()}
    
    def save_img(self,pred,mask,name):
        def mask2img(mask):
            return Image.fromarray((mask * 255).astype(np.uint8))
        for i,(p,_,n) in enumerate(zip(pred,mask,name)):
            out_file=self.args.save_temp_mask_dir + '/' + str(n)
            p=mask2img(p[0,:,:,None].repeat(3,axis=-1))
            p.save(out_file)
    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs=outputs)

    

    
class ModifiedTent(Tent):
    def __init__(self,args):
        super(ModifiedTent,self).__init__(args)
        self.mse_loss=nn.MSELoss()
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
        return optimizer

    def training_step(self, batch, batch_idx):
        image, _ ,_= batch
        image_R=torch.rot90(image,1,[2,3])
        pred_mask = self(image)
        loss = self.entropymin_loss(pred_mask)
        pred_mask_soft=pred_mask.softmax(1)
        _,s_tgt,_=torch.linalg.svd(pred_mask_soft)
        if self.weight_crf < 1:
            self.weight_crf = 0.05 * self.current_epoch
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
        loss = loss_Entorpy + self.weight_crf * out_gatedcrf + 0.5* BNM_method_loss + loss_r
        self.log("loss",loss)
        return loss
