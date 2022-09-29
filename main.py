r'''
Author       : PiKaChu_wcg
Date         : 2022-09-20 13:14:44
LastEditors  : PiKachu_wcg
LastEditTime : 2022-09-29 14:05:27
FilePath     : /model_new/main.py
'''

import argparse
import os
from torch.utils import data
from data import UniDataset
import pytorch_lightning as pl
from pl_model import ModifiedTent, Tent

PRETRAIN=dict(
    tcia="lightning_logs/TCIA_pretrain/checkpoints/epoch=99-step=300.ckpt",
    tnbc='lightning_logs/TNBC_pretrain/checkpoints/epoch=99-step=200.ckpt',
    kirc='lightning_logs/KIRC_pretrain/checkpoints/epoch=99-step=1000.ckpt',
)
DATA_FILE=dict(
    tcia='./TCIA.txt',
    tnbc='./TNBC.txt',
    kirc='./KIRC.txt',
)
def get_args():
    parser = argparse.ArgumentParser(description='Train the CellSegUDA/CellSegSSDA on source images and target images')
    parser.add_argument('--epochs', type=int, default= 100,
                        help='Number of epoches')
    parser.add_argument("--data-file",choices=['tcia','tnbc','kirc'],default='tica')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Number of images sent to the network in one step')
    parser.add_argument('--learning-rate-S', type=float, default=0.0001,
                        help='learning rate of Segmentation network')
    parser.add_argument('--resume', choices=['tcia','tnbc','kirc'],default='tica')
    parser.add_argument('--gpus',type=int,default=1)
    parser.add_argument('--save_temp_mask_dir',type=str,default='./temp_mask/')
    parser.add_argument('--modified',action='store_true')
    parser.add_argument('--test',action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    args.save_temp_mask_dir=os.path.join(args.save_temp_mask_dir,f"{args.resume}2{args.data_file}{'_modified' if args.modified else ''}")
    args.data_file=DATA_FILE[args.data_file]
    args.resume=PRETRAIN[args.resume]
    loader = data.DataLoader(UniDataset(args.data_file), batch_size=args.batch_size, shuffle=True, pin_memory=True,num_workers=8)
    trainer=pl.Trainer(
        gpus=list(range(args.gpus)),
        strategy='ddp',
        max_epochs=args.epochs,
        precision=16
    )
    model= ModifiedTent(args) if args.modified else Tent(args) 
    trainer.fit(model,train_dataloaders=loader,val_dataloaders=loader)
    if args.test:
        if not os.path.exists(args.save_temp_mask_dir):
            os.mkdir(args.save_temp_mask_dir)
        trainer.test(model,dataloaders=loader)
    

if __name__=='__main__':
    main()