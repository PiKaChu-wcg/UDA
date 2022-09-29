r'''
Author       : PiKaChu_wcg
Date         : 2022-09-24 12:52:47
LastEditors  : PiKachu_wcg
LastEditTime : 2022-09-28 23:16:41
FilePath     : /model_new/data.py
'''
import os
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as T
import albumentations as A


mean=[0.6519246207720975, 0.5085459403364605, 0.7288387940033376]
std=[0.1696320206422359, 0.21996593203485643, 0.1902185935660757]
class UniDataset(Dataset):
    def __init__(self,file,aug=None):
        with open(file,'r') as f:
            self.data=[l.split() for l in f.readlines()]
        if aug is None:
            self.aug=A.Compose([
                A.Resize(512,512),
            ])
        else:
            self.aug=aug
        self.toTensor=T.ToTensor()
        self.norm=T.Normalize(mean,std)
                
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx) :
        img=cv2.imread(self.data[idx][0])
        mask=cv2.imread(self.data[idx][1])
        transformed=self.aug(image=img,mask=mask)
        img,mask=transformed['image'],transformed['mask']
        img,mask=self.toTensor(img),self.toTensor(mask)
        img=self.norm(img)
        mask=(mask.mean(0,keepdim=True)>0.00001).long()
        name=os.path.basename(self.data[idx][0])
        return img,mask,name




if __name__=='__main__':
    data=UniDataset('/opt/data/private/UDA/model_new/KIRC.txt')
    for i in data:
        print(i)
    