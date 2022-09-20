import cv2
import os
import cv2
import numpy as np
from medpy.metric import binary
from PIL import Image
import sys

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass
path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()
sys.stdout = Logger('a.txt')

RES_LOG = 'E:/old_model_data/715paper/New_norepeat200'
MASK_LOG = 'E:/old_model_data/MASKS'
IMAGE_LOG = 'E:/old_model_data/Images'

listres1 = os.listdir(RES_LOG)
listmask = os.listdir(MASK_LOG)

for i in listres1:
    res_log2 = RES_LOG + '/' + i
    listres2 = os.listdir(res_log2)
    for j in listres2:
        res_log3 = res_log2 + '/' + j
        listres3 = os.listdir(res_log3)
        for x in listmask:
            if x == j[5:]:
                mask_log2 = MASK_LOG +'/' + x
                listmask2 = os.listdir(mask_log2)
                for k in listres3:
                    if k == 'result':
                        res_log4 = res_log3 + '/' + k
                        listres4 = os.listdir(res_log4) 
                        out_name = i+j+k
                        canny_name = res_log3 + '/' + 'canny'
                        if not os.path.exists(canny_name):
                            os.makedirs(canny_name)

                        num = 0
                        temp = 0
                        dice_temp = 0
                        assd_temp = 0
                        hd95_temp = 0
                        assd_all = []
                        dice_all = []
                        miou_all = []
                        hd95_all = []

                        for l in listres4:
                            mask_name = res_log4 +'/' + l
                            #print(l)
                    

                            for h in listmask2:
                                # print(mask_name)
                                if h[:-4] == l[2:-7]:
                                    mask = Image.open(mask_name)
                                    mask_canny = cv2.imread(mask_name)
                                    out_canny_name = canny_name + '/' + l
                                    ori_image = cv2.imread(IMAGE_LOG + '/' + x + '/' + h)
                                    ori_image = cv2.resize(ori_image,(512,512))
                                    mask_canny = cv2.Canny(mask_canny, 90, 200)
                                    mask_canny[mask_canny > 60] = 127
                                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
                                    dilate = cv2.dilate(mask_canny, kernel, iterations=2)
                                    ori_image[dilate > 60] = (117, 250, 76)
                                    cv2.imwrite(out_canny_name, ori_image)



                                    mask = np.array(mask)
                                    mask_black = np.zeros((512,512))
                                    gt = Image.open(mask_log2 + '/' + h)
                                    gt = gt.resize((512,512))
                                    gt = np.array(gt)
                                    # print(np.max(mask))
                                    mask[mask <= 128] = 0
                                    mask[mask > 128] = 1
                                    # print(np.max(mask))
                                    gt[gt <= 128] = 0
                                    gt[gt > 128] = 1
                                    nul = mask * gt


                                    dice = 2 * np.sum(nul) / (np.sum(mask) + np.sum(gt))
                                    dice_temp = dice_temp + dice

                                    intersection = np.sum(nul)
                                    union = np.sum(mask) + np.sum(gt) - intersection + 1e-6
                                    miou = intersection / union
                                    if(mask.all() != mask_black.all()):
                                        assd = 0
                                        hd95 = 0
                                    else:
                                        assd = binary.assd(mask, gt, connectivity=1)
                                        hd95 = binary.hd95(mask, gt)
                                        hd95_temp = hd95_temp + hd95
                                        assd_temp = assd_temp + assd
                                    temp = temp + miou
                                    num = num + 1

                                    assd_all.append(assd)
                                    miou_all.append(miou)
                                    hd95_all.append(hd95)
                                    dice_all.append(dice)

                            # t_assd = assd_temp / num
                            # t_miou = temp / num
                            # t_dice = dice_temp / num
                            # t_hd95 = hd95_temp / num

                        assd_mean = np.mean(assd_all)
                        assd_std = np.std(assd_all, ddof=1)
                        hd95_mean = np.mean(hd95_all)
                        hd95_std = np.std(hd95_all, ddof=1)
                        miou_mean = np.mean(miou_all)
                        miou_std = np.std(miou_all, ddof=1)
                        dice_mean = np.mean(dice_all)
                        dice_std = np.std(dice_all, ddof=1)
            
                        # print(out_name + 'assd_average' + t_assd)
                        # print(out_name + 'miou_average' + t_miou)
                        # print(out_name + 'dice_average' + t_dice)
                        # print(out_name + 'hd95_average' + t_hd95)
                        print(out_name + 'assd_average: ' + str(assd_mean))
                        print(out_name + 'miou_average: ' + str(miou_mean))
                        print(out_name + 'hd95_average: ' + str(hd95_mean))
                        print(out_name + 'dice_average: ' + str(dice_mean))
                        print(out_name + 'assd_std: ' + str(assd_std))
                        print(out_name + 'miou_std: ' + str(miou_std))
                        print(out_name + 'hd95_std: ' + str(hd95_std))
                        print(out_name + 'dice_std: ' + str(dice_std))

                        



