import argparse
from numpy.core.fromnumeric import ptp
from scipy.ndimage.morphology import distance_transform_edt
import torch
import random
import torch.nn as nn
from torch.nn.modules.loss import MSELoss
from torch.utils import data, model_zoo
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable#maybe not used

import logging
import os
import sys
import math
import denseCRF

from unet.unet_model import UNet
from unet.attUnet import AttU_Net
from Discriminator import GANDiscriminator
from model.resnet_encoder import resnet101
from Loss import Adversarial_loss, DiceLoss, Reconstruction_loss, DisCrossEntropyLoss, CRloss, em_loss, entropy_loss
from dataset import BasicDataset, TargetDataset, Basic_TS_Data
from SSDA_train import SSDA_main
from evaluate import mask_to_image

from tqdm import tqdm
from collections import OrderedDict
from tensorboardX import SummaryWriter
from medpy.metric import binary
import shutil
import cv2
from PIL import Image



DIR_IMG_TARGET = 'data/img_target/'
DIR_IMG_SOURCE = 'data/img_source/'
DIR_MASK_SOURCE = 'data/mask_source/'
DIR_CHECKPOINT = 'checkpoints/'
DIR_IMG_TARGET_LABELED = 'data/img_target_labeled/'
DIR_MASK_TARGET_LABELED = 'data/mask_target_labeled/'
#dir_mask_target = 'data/mask_target'#这个要与无标签的target做个区分进入不同的S2


#后面可不可以整成输入size可变的



def get_args():
    parser = argparse.ArgumentParser(description='Train the CellSegUDA/CellSegSSDA on source images and target images')
    parser.add_argument('--model', type=str, default='Unet',
                        help="available options : Unet,AttUnet")
    parser.add_argument('--type', type=str, default='UDA',
                        help='UDA or SSDA')
    parser.add_argument('--mode', type=int, default=1,
                        help='0 default,1 plus rotate,2 plus cut in unet1,3 plus cut in unet2,4 1+2, 5 1+3')
    parser.add_argument('--epochs', type=int, default= 100,
                        help='Number of epoches')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Number of images sent to the network in one step')
    parser.add_argument('--input-size-source', type=str, default='400,400', 
                        help='Comma-separated string with height and width of source images')
    parser.add_argument('--images-dir-source', type=str, default=DIR_IMG_SOURCE,
                        help='Path to the dictionary containing the source images')
    parser.add_argument('--masks-dir-source',type=str,default=DIR_MASK_SOURCE,
                        help='Path to the dictionary containing the source masks')
    parser.add_argument('--input-size-target', type=str, default='512,512',
                        help='Comma-separated string with height and width of target images')
    parser.add_argument('--images-dir-target',type=str, default=DIR_IMG_TARGET,
                        help='Path to the dictionary containing the target images')
    parser.add_argument('--images-dir-target_labeled',type=str, default=DIR_IMG_TARGET_LABELED)
    parser.add_argument('--masks-dir-target_labeled',type=str, default=DIR_MASK_TARGET_LABELED)
    parser.add_argument('--images-dir-eval', default='./data/img_evaluate/')#eval
    parser.add_argument('--masks-dir-eval', metavar='INPUT_masks', default='./data/masks_evaluate/')#eval
    parser.add_argument('--img-source-vali', metavar='vali_img', default='./data/img_source_vali/')#vali
    parser.add_argument('--mask-source-vali', metavar='vali_mask', default='./data/mask_source_vali/')#vali
    parser.add_argument('--img-target-vali', metavar='vali_target_img', default='./data/img_target_vali/')#vali
    parser.add_argument('--mask-target-vali', metavar='vali_target_mask', default='./data/mask_target_vali/')#vali
    parser.add_argument('--save-mask-dir', metavar='save_mask_dir', default='./data/result_target_mask/')
    parser.add_argument('--save-temp-mask-dir', metavar='save_temp_mask_dir', default='./data/result_temp_target_mask/')
    parser.add_argument('--logs-dir', type=str, default='./logs/',
                        help='Path to the dictionary containing the logs')
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--txt-dir', type=str, default= './txts/',
                        help='Path to the dictionary containing the txts')
    parser.add_argument('--learning-rate-S', type=float, default=0.0001,
                        help='learning rate of Segmentation network')
    parser.add_argument('--learning-rate-D', type=float, default=0.001,
                        help='learning rate of Discriminator')
    parser.add_argument('--learning-rate-R', type=float, default=0.001,
                        help='learning rate of Decoder')
    parser.add_argument("--S-num-classes", type=int, default=2,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--R-num-classes", type=int, default=3,
                        help="Number of classes Reconstruction need to predict (including background).")
    parser.add_argument('--power',type=float, default=0.9,
                        help='Decay parameter to compute the learning rate')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Regularisation parameter for L2-loss')
    parser.add_argument('--lambda-adv', type=float, default=0.001,
                        help='The weight of adv loss')
    parser.add_argument('--lambda-recons', type=float, default=0.01,
                        help='The weight of recons loss')
    #parser.add_argument('--start-epoch', type=int, default=0)
    # parser.add_argument("--gpu", type=int, default=0,
    #                     help="choose gpu device.")
    parser.add_argument('--seed', type=int, default=256)#随机种子数
    parser.add_argument('--device', type=str, default='cuda')#运行设备
    parser.add_argument('--set', type=str, default='train',
                        help='choose adaptation set')
    parser.add_argument('--crf', type=int, default=0)
    parser.add_argument('--save-crf-dir', metavar='save_crf_dir', default='./data/result_target_crf_mask/')
    return parser.parse_args()

args = get_args()

def load_model(model, model_path):
    state_dict = torch.load(model_path)
    # create new OrderedDict that does not contain `module.`
        
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model

def main():
    #create the model and start training
    # h, w = map(int, args.input_size_source.split(','))
    # input_size_source = (h, w)
    # h, w = map(int, args.input_size_target.split(','))
    # input_size_target = (h, w)
    TensorboardWriter = SummaryWriter(comment='UDA_method')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.enabled = True
    #create network
    if args.model == 'Unet':
        model_S = UNet(n_channels=3, n_classes=args.S_num_classes-1)#unet's n_classes should be 1 when actually 2 classes
        model_R = UNet(n_channels=1, n_classes=args.R_num_classes)#这里是不是3呢？
        model_T = UNet(n_channels=3, n_classes=args.S_num_classes-1)
    elif args.model == 'AttUnet':
        model_S = AttU_Net(img_ch=3, output_ch=args.S_num_classes-1)
        model_R = AttU_Net(img_ch=1, output_ch=args.R_num_classes)
    
    model_S = torch.nn.DataParallel(model_S).to(args.device)
    model_T = torch.nn.DataParallel(model_T).to(args.device)
    #Init D and R
    model_D = GANDiscriminator(num_classes=args.S_num_classes-1)
    model_D = torch.nn.DataParallel(model_D).to(args.device)
    model_R = UNet(n_channels=1, n_classes=args.R_num_classes)
    model_R = torch.nn.DataParallel(model_R).to(args.device)

    start_epoch = 0
    lr_S = args.learning_rate_S
    lr_D = args.learning_rate_D
    lr_R = args.learning_rate_R

    if os.path.isfile(args.resume):
        print('loading checkpoint:{}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        lr_S = checkpoint['lr_S']
        lr_D = checkpoint['lr_D']
        lr_R = checkpoint['lr_R']
        model_S.load_state_dict(checkpoint['state_dict_S'])
        model_D.load_state_dict(checkpoint['state_dict_D'])
        model_R.load_state_dict(checkpoint['state_dict_R'])
        print('load successfully!')

    model_S.train()

    model_D.train()

    model_R.train()

    cudnn.benchmark = True

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)


    train_S_dataset = Basic_TS_Data(args.images_dir_source, args.masks_dir_source,args.images_dir_target, 1)#image_scale = 1 
    train_S_loader = data.DataLoader(train_S_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)

    #train_S_loader_iter = enumerate(train_S_loader)

    testloader = data.DataLoader(BasicDataset(args.img_target_vali, args.mask_target_vali), batch_size=1, shuffle=False, pin_memory=False)

    #targetloader_iter = enumerate(target_loader)

    logging.info(
        '''
        Start Training:
        '''
    )#log need

    optimizer_S = optim.Adam(model_S.parameters(), lr=lr_S)#adam default
    optimizer_S.zero_grad()

    optimizer_D = optim.Adam(model_D.parameters(), lr=lr_D)
    optimizer_D.zero_grad()

    optimizer_R = optim.Adam(model_R.parameters(), lr=lr_R)
    optimizer_R.zero_grad()

    adv_loss = Adversarial_loss()
    seg_loss = DiceLoss()
    dis_loss = DisCrossEntropyLoss()
    recons_loss = Reconstruction_loss()
    CR_loss = CRloss()
    entropymin_loss = entropy_loss()
    mse_loss = MSELoss()

    # Labels for Discriminater loss
    source_label = 1
    target_label = 0
    best_dice = 0
    alpha = 0

    for epoch in range(start_epoch, args.epochs):

        model_S.train()#for eval

        optimizer_D.zero_grad()
        optimizer_R.zero_grad()
        optimizer_S.zero_grad()

        #train S
        #remove grads
        for param in model_D.parameters():
            param.requires_grad = False
        for param in model_R.parameters():
            param.requires_grad = False
        
        #train source
        for i,(images,mask,imagesT,idx,idt) in enumerate(train_S_loader):
            images_S = images
            True_masks_S = mask
            True_masks_S = True_masks_S.to(args.device)
            images_S = images_S.to(args.device)

            pred_masks_S = model_S(images_S)
            pred_masks_S_faltten = pred_masks_S.flatten()#this flatten might be not right
            True_masks_S_faltten = True_masks_S.flatten()

            #Loss_seg
            loss_seg = seg_loss(pred_masks_S_faltten, True_masks_S_faltten)
            loss_em = entropymin_loss(pred_masks_S)
            loss_seg_o = loss_seg # + loss_em
            optimizer_S.zero_grad()

            loss_seg_o.backward()
            
            # if args.mode == 1:
            #     # images_S = images_S.detach()
            #     images_S_rotate = torch.rot90(images_S, 1, [2,3])
            #     #True_masks_S_rotate = batch['mask_rotate']
            #     images_S_rotate = images_S_rotate.to(args.device)
            #     #True_masks_S_rotate = True_masks_S_rotate.to(args.device)
            #     pred_masks_S_rotate = model_S(images_S_rotate)
            #     pred_masks_S = pred_masks_S.detach()
            #     pred_masks_S_2_rotate = torch.rot90(pred_masks_S, 1, [2,3])

            #     #Loss_Rotate
            #     loss_rotate = recons_loss(pred_masks_S_rotate, pred_masks_S_2_rotate)
            #     loss_rotate = math.exp(-5 * pow(1 - epoch, 2)) * loss_rotate
            #     loss_rotate.backward()

            if args.mode == 2:
                images_S_anchor = batch['image_cut_anchor']
                images_S_P = batch['image_cut_P']
                images_S_N = batch['image_cut_N']
                images_S_anchor = images_S_anchor.to(args.device)
                images_S_P = images_S_P.to(args.device)
                images_S_N = images_S_N.to(args.device)
                pred_masks_S_anchor = model_S(images_S_anchor)
                pred_masks_S_P = model_S(images_S_P)
                pred_masks_S_N = model_S(images_S_N)

                loss_CR = CR_loss(pred_masks_S_P, pred_masks_S_N, pred_masks_S_anchor, args.device)
                loss_CR.backward()

            optimizer_S.step()

            #compute the pred masks of target
            images_T = imagesT
            images_T = images_T.to(args.device)

            if args.mode == 1:
                # images_S = images_S.detach()
                images_T_rotate = torch.rot90(images_T, 1, [2,3])
                #True_masks_S_rotate = batch['mask_rotate']
                images_T_rotate = images_T_rotate.to(args.device)
                #True_masks_S_rotate = True_masks_S_rotate.to(args.device)
                pred_masks_T_rotate = model_T(images_T_rotate)
                pred_masks_T = model_S(images_T)
                # pred_masks_T = pred_masks_T.detach()
                pred_masks_T_2_rotate = torch.rot90(pred_masks_T, 1, [2,3])

                #Loss_Rotate
                optimizer_S.zero_grad()

                loss_rotate = recons_loss(pred_masks_T_rotate, pred_masks_T_2_rotate)
                if epoch < 50:
                    loss_rotate = (1 - math.exp(- epoch)) * loss_rotate
                if epoch > 55:
                    loss_rotate = math.exp(50 - epoch) * loss_rotate
                loss_rotate.backward()
                optimizer_S.step()

            pred_masks_T = model_S(images_T)
            # pred_masks_T = pred_masks_T.detach()

            D_out = model_D(pred_masks_T)
            loss_adv = adv_loss(D_out, args.device)#gpu need to be writed dis and adv

            loss_adv = loss_adv * args.lambda_adv

            optimizer_D.zero_grad()
            loss_adv.backward()
            
            # if args.mode == 1:
            #     images_T_rotate = batch['image_rotate']
            #     images_T_rotate = images_T_rotate.to(args.device)
            #     pred_masks_T_rotate = model_S(images_T)
            #     pred_masks_T_rotate = torch.rot90(pred_masks_T_rotate, 1, [2,3])
            #     pred_masks_T = pred_masks_T.detach()
            #     #Loss_Rotate
            #     optimizer_S.zero_grad()
            #     loss_rotate_T = recons_loss(pred_masks_T_rotate, pred_masks_T)
            #     loss_rotate_T = math.exp(-5 * pow(1 - epoch, 2)) * loss_rotate_T
            #     loss_rotate_T.backward()
            #     optimizer_S.step()

            # if args.mode == 2:
            #     images_T_A = batch['image_cut_A']
            #     images_T_B = batch['image_cut_B']
            #     images_T_A = images_T_A.to(args.device)
            #     images_T_B = images_T_B.to(args.device)
            #     pred_masks_T_A = model_S(images_T_A)
            #     pred_masks_T_B = model_S(images_T_B)
            #     loss_CR = CR_loss(pred_masks_T_A, pred_masks_T_B)
            #     optimizer_S.zero_grad()
            #     loss_CR.backward()
            #     optimizer_S.step()
            

            #train R
            for param in model_R.parameters():
                param.requires_grad = True
            
            pred_masks_TR = model_S(images_T)
            #pred_masks_T = pred_masks_T.detach()
            pred_images_TR = model_R(pred_masks_TR)

            loss_recons = recons_loss(pred_images_TR, images_T)
            loss_recons = loss_recons * args.lambda_recons

            optimizer_R.zero_grad()
            loss_recons.backward()
            # loss = loss_adv + loss_recons + loss_seg
            # loss.backward()

            if args.mode == 3:
                images_T_P = batch['image_cut_P']
                images_T_N = batch['image_cut_N']
                images_T_anchor = batch['image_cut_anchor']
                images_T_P = images_T_P.to(args.device)
                images_T_N = images_T_N.to(args.device)
                images_T_anchor = images_T_anchor.to(args.device)
                pred_masks_T_P = model_S(images_T_P)
                pred_masks_T_N = model_S(images_T_N)
                pred_masks_T_anchor = model_S(images_T_anchor)
                pred_images_T_P = model_R(pred_masks_T_P)
                pred_images_T_N = model_R(pred_masks_T_N)
                pred_images_T_anchor = model_R(pred_masks_T_anchor)

                loss_CR_R = CR_loss(pred_images_T_P, pred_images_T_N, pred_images_T_anchor)
                loss_CR_R.backward()

            #train D
            for param in model_D.parameters():
                param.requires_grad = True
            pred_masks_S = pred_masks_S.detach()
            D_out_S = model_D(pred_masks_S)
            loss_dis_s = dis_loss(D_out_S, source_label, args.device)
            
            loss_dis_s.backward()

            pred_masks_T = pred_masks_T.detach()#detach切断与之前的联系
            D_out_T = model_D(pred_masks_T)
            loss_dis_t = dis_loss(D_out_T, target_label, args.device)# Z caution
            loss_dis_t.backward()

            optimizer_D.step()
            optimizer_R.step()

        print('path = {}'.format(args.logs_dir))
        print(
            'epoch = {0:6d}, loss_seg = {1:.4f}, loss_adv = {2:.4f}, loss_recons = {3:.4f}, loss_dis_S = {4:.4f}, loss_dis_T = {5:.4f}, loss_em = {6:.4f}'.format(
                epoch, loss_seg, loss_adv, loss_recons, loss_dis_s, loss_dis_t, loss_em
            )
        )
        TensorboardWriter.add_scalar('loss_seg', loss_seg, global_step=epoch)
        TensorboardWriter.add_scalar('loss_adv', loss_adv, global_step=epoch)
        TensorboardWriter.add_scalar('loss_recons', loss_recons, global_step=epoch)
        TensorboardWriter.add_scalar('loss_dis_s', loss_dis_s, global_step=epoch)
        TensorboardWriter.add_scalar('loss_dis_t', loss_dis_t, global_step=epoch)
        TensorboardWriter.add_scalar('loss_em', loss_em, global_step=epoch)

        # print('loss em:{}'.format(loss_em))

        if args.mode == 1:
            print('loss rotate = {}'.format(loss_rotate))
            TensorboardWriter.add_scalar('loss_rotate', loss_rotate, global_step=epoch)
        if args.mode == 2:
            print('loss CR = {}'.format(loss_CR))
        if args.mode == 3:
            print('loss CRR = {}'.format(loss_CR_R))

        f_loss = open(os.path.join(args.logs_dir, 'loss.txt'), 'a')
        f_loss.write('{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:.4f} \n'.format(
            loss_seg, loss_adv, loss_recons, loss_dis_s, loss_dis_t, loss_em
        ))
        f_loss.close()
        if epoch >= args.epochs - 1:
            print('save model ...')
            torch.save({
                'epoch' : epoch,
                'state_dict_S': model_S.state_dict(),
                'state_dict_D': model_D.state_dict(),
                'state_dict_R': model_R.state_dict(),
                'lr_S': optimizer_S.param_groups[0]['lr'],
                'lr_D': optimizer_D.param_groups[0]['lr'],
                'lr_R': optimizer_R.param_groups[0]['lr']
            }, os.path.join(args.logs_dir, 'UDA' + str(args.epochs) + '.pth'))

        #eval
        print('begin eval!')
        model_S.eval()
        
        temp = 0
        dice_temp = 0
        num = 0
        assd_temp = 0
        hd95_temp = 0
        with torch.no_grad():
            for i,(images,mask,name) in enumerate(testloader):
                mask = mask
                image = images
                #img = image.unsqueeze(0)
                img = image.to(device=args.device, dtype=torch.float32)
                mask = mask.to(device=args.device, dtype=torch.float32)
                
                output = model_S(img)
                probs = torch.sigmoid(output)

                probs = probs.squeeze(0)

                tf = transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        transforms.ToTensor()
                    ]
                )

                probs = tf(probs.cpu())
                full_mask = probs.squeeze().cpu().numpy()

                out = full_mask > 0.5
                Iq = img.squeeze(0)
                Iq = Iq.squeeze().cpu().numpy()
                Iq = Iq.transpose((2,0,1))
                Iq = Iq.astype(np.uint8)
                # Iq = Image.fromarray(Iq).convert('RGB')
                result = mask_to_image(out)
                out_file = args.save_temp_mask_dir + '/' + str(name)
                result.save(out_file +'.png')
                # L = Image.open(out_file +'.png')

                if args.crf == 1:
                    crf_mask_path = args.save_crf_dir + '/'
                    if not os.path.exists(crf_mask_path):
                        os.makedirs(crf_mask_path)
                    crf_mask_name = crf_mask_path + str(name)

                    Lq = np.asarray(result, np.float32)
                    # prob = Lq[:, :, np.newaxis]
                    prob = cv2.cvtColor(Lq, cv2.COLOR_GRAY2BGR) / 255
                    # print(prob.size, Iq.size, image.size)
                    # print(Iq.shape, image.shape, prob.shape)
                    # print(Iq.dtype, prob.dtype)
                    prob = prob[:, :, :2]
                    prob[:, :, 0] = 1.0 - prob[:, :, 0]
                    w1 = 10.0  # weight of bilateral term
                    alpha = 20  # spatial std
                    beta = 13  # rgb  std
                    w2 = 3.0  # weight of spatial term
                    gamma = 3  # spatial std
                    it = 5.0  # iteration
                    param = (w1, alpha, beta, w2, gamma, it)
                    lab = denseCRF.densecrf(Iq, prob, param)
                    cv2.imwrite(crf_mask_name + '.png', lab * 255) 
                    result = lab * 255

                tf2 = transforms.Compose([transforms.ToTensor()])
                result_t = tf2(result)
                result_t = result_t.to(device=args.device, dtype=torch.float32)
                a = torch.flatten(mask)
                b = torch.flatten(result_t)

                a[a > 0.5] = 1
                a[a <= 0.5] = 0
                b[b > 0.5] = 1
                b[b <= 0.5] = 0
                nul = a * b

                dice = 2 * torch.sum(nul) / (torch.sum(a) + torch.sum(b))
                dice_temp = dice_temp + dice

                intersection = torch.sum(nul)
                union = torch.sum(a) + torch.sum(b) - intersection + 1e-6
                miou = intersection / union

                mask = mask.cpu().numpy()
                result_t = result_t.cpu().numpy()
                mask = np.reshape(mask, result_t.shape)
                if(epoch > 500):
                    assd = binary.assd(result_t, mask, connectivity=1)
                    hd95 = binary.hd95(result_t, mask)
                    hd95_temp = hd95_temp + hd95
                    assd_temp = assd_temp + assd
                temp = temp + miou
                num = num + 1

        t_assd = assd_temp / num
        t_miou = temp / num
        t_dice = dice_temp / num
        t_hd95 = hd95_temp / num
        if t_dice > best_dice:
            print('save best model ...')
            torch.save({
                'epoch' : epoch,
                'state_dict_S': model_S.state_dict(),
                'lr_S': optimizer_S.param_groups[0]['lr']
            }, os.path.join(args.logs_dir, 'UDA' + str(args.epochs) +'best.pth'))
            for root, _, fnames in os.walk(args.save_temp_mask_dir):
                for fname in sorted(fnames):  # sorted函数把遍历的文件按文件名排序
                    fpath = os.path.join(root, fname)
                    shutil.copy(fpath, args.save_mask_dir)  # 完成文件拷贝
            best_dice = t_dice

        TensorboardWriter.add_scalar('target vali eval miou', t_miou, global_step=epoch)
        TensorboardWriter.add_scalar('target vali eval dice', t_dice, global_step=epoch)
        TensorboardWriter.add_scalar('target vali eval assd', t_assd, global_step=epoch)
        TensorboardWriter.add_scalar('target vali eval hd95', t_hd95, global_step=epoch)
        TensorboardWriter.add_scalar('target vali eval best dice', best_dice, global_step=epoch)
        print("target vali eval miou is {0}, target vali eval dice is {1}".format(t_miou,t_dice))
if __name__ == '__main__':
    if args.type == 'UDA':
        main()
    elif args.type == 'SSDA':
        SSDA_main(args)
