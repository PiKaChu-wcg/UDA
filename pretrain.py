r'''
Author       : PiKaChu_wcg
Date         : 2022-09-24 15:36:30
LastEditors  : PiKachu_wcg
LastEditTime : 2022-09-28 17:25:07
FilePath     : /model_new/pretrain.py
'''
import argparse

from torch.utils import data
from data import UniDataset
import pytorch_lightning as pl
from pl_model import Pretrain


def get_args():
    parser = argparse.ArgumentParser(description='Train the CellSegUDA/CellSegSSDA on source images and target images')
    parser.add_argument('--epochs', type=int, default= 100,
                        help='Number of epoches')
    parser.add_argument("--train-data-file",default='./TCIA.txt')
    parser.add_argument("--valid-data-file",default='./TNBC.txt')
    parser.add_argument('--batch-size', type=int, default=24,
                        help='Number of images sent to the network in one step')
    parser.add_argument('--learning-rate-S', type=float, default=0.0001,
                        help='learning rate of Segmentation network')
    parser.add_argument('--gpus',type=int,default=1)
    return parser.parse_args()


def main():
    args = get_args()
    train_loader = data.DataLoader(UniDataset(args.train_data_file), batch_size=args.batch_size, shuffle=True, pin_memory=True,num_workers=8)
    valid_loader = data.DataLoader(UniDataset(args.valid_data_file), batch_size=2*args.batch_size, shuffle=False, pin_memory=True,num_workers=8)
    
    trainer=pl.Trainer(
        gpus=list(range(args.gpus)),
        strategy='ddp',
        max_epochs=args.epochs,
        precision=16
    )
    model=Pretrain(args)
    trainer.fit(model,train_dataloaders=train_loader,val_dataloaders=valid_loader)

    

if __name__=='__main__':
    main()