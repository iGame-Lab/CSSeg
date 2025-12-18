from MedSAM.segment_anything import sam_model_registry
from MedSAM.demo import BboxPromptDemo
import os
import numpy as np
import torch
import cv2
import joblib
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from CAM.utils import GradCAM, show_cam_on_image, center_crop_img
from CAM.vit_model import vit_base_patch16_224
from CAM.vit_model import vit_base_patch16_224_in21k as create_model
import argparse
import math
import train_utils.distributed_utils as utils
import CAM.main_vit
from CAM.main_vit import show_mask,dice_coeff
from Utils import *
import logging
import time
from CAM.utils import scoremap2bbox
from skimage import measure
import multiprocessing
import CRF
def scale_cam_image(cam, target_size=None):
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        if target_size is not None:
            img = cv2.resize(img, target_size)
        result.append(img)
    result = np.float32(result)

    return result
def preprossess(img,mask):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i][j]<=1e-9):
                mask[i][j]=0
    return mask
def find_max_eras(mask):
    # 图像读取
    img = mask
    img = np.array(img)
    img[img != 0] = 1  # 图像二值化
    # 图像实例化
    img = measure.label(img, connectivity=2)
    props = measure.regionprops(img)
    # 最大区域获取
    max_area = 0
    max_index = 0
    # props只包含像素值不为零区域的属性，因此index要从1开始
    for index, prop in enumerate(props, start=1):
        if prop.area > max_area:
            max_area = prop.area
            # index 代表每个联通区域内的像素值；prop.area代表相应连通区域内的像素个数
            max_index = index

    img[img != max_index] = 0
    img[img == max_index] = 1


    return img

def show_mask_image(mask, ax, random_color=False, alpha=0.95):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
def main(opt,logger):
    n_jobs = multiprocessing.cpu_count()
    device = "cuda:0"
    #CAM模型导入
    MedSAM_CKPT_PATH = "C:/Users/Administrator/Desktop/MedSAM-main/work_dir/MedSAM/medsam_vit_b.pth"
    medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()

    dice_CAM_CRF=0
    dice_cam_res=0
    dice_sam_crf=0
    dice_sam_cam=0
    dice_pre_max=0
    dice_pre_sam_cam=0
    dice_pre_sam_crf=0
    dice_max_crf_cam=0
    pic_len=0
    bbox_prompt_demo = BboxPromptDemo(medsam_model)
    pic_sum=len(os.listdir(opt.img_path))
    for i,img_name in enumerate(os.listdir(opt.img_path),1):

        # start_time=time.perf_counter()
        img_data = np.load(os.path.join(opt.img_path, img_name))["arr_0"]
        label_data = np.load(os.path.join(opt.label_path, img_name))["arr_0"]
        label_data[label_data > 0] = 1
        grad_cam = np.load(os.path.join(opt.out_path, img_name))['original_cam']
        original_grad_cam = np.array(grad_cam).copy()
        original_grad_cam[original_grad_cam<0.7]=0
        original_grad_cam[original_grad_cam>=0.7]=1
        original_grad_cam = preprossess(img_data,original_grad_cam)
        dice_cam_res+=dice_coeff(original_grad_cam,label_data)



        temp = []
        for _ in range(3):
            temp.append(img_data)
        tmpimage = np.array(temp)
        tmpimage = tmpimage.transpose(1, 2, 0)
        grad_cam = preprossess(img_data, grad_cam)

        CRF_label = CRF.crf(n_jobs, tmpimage * 255, label_data, grad_cam ,
                            mean_bgr=(img_data.mean(), img_data.mean(), img_data.mean()))
        dice_CAM_CRF+=dice_coeff(CRF_label,label_data)


        boxcrf, cntcrf = scoremap2bbox(scoremap=CRF_label, threshold=0.4, multi_contour_eval=True)

        boxcam, cntcam = scoremap2bbox(scoremap=original_grad_cam, threshold=0.4, multi_contour_eval=True)


        tempboxcrf = []
        for crfi in range(cntcrf):
            if original_grad_cam[ boxcrf[crfi][1]:boxcrf[crfi][3],boxcrf[crfi][0]:boxcrf[crfi][2]].sum() > 2:
                tempboxcrf.append(boxcrf[crfi])


        cntcrf = len(tempboxcrf)
        if cntcrf ==0:
            boxcrf=[[0,0,1,1]]
        else :
            boxcrf = tempboxcrf



        SAM_mask_crf = bbox_prompt_demo.show(image_path=os.path.join(opt.img_path, img_name), Box=boxcrf[0])
        for j_ in range(1, cntcrf):
            tmp_SAM_mask = bbox_prompt_demo.show(image_path=os.path.join(opt.img_path, img_name), Box=boxcrf[j_])
            SAM_mask_crf = SAM_mask_crf + tmp_SAM_mask
        # show_cam_on_image(tmpimage, SAM_mask, use_rgb=True, dice=dice_coeff(SAM_mask, label_data),
        #                       name="sam")
        SAM_mask_crf[SAM_mask_crf > 0] = 1
        dice_sam_crf+=dice_coeff(SAM_mask_crf,label_data)

        SAM_mask_cam = bbox_prompt_demo.show(image_path=os.path.join(opt.img_path, img_name), Box=boxcam[0])
        for j_ in range(1, cntcam):
            tmp_SAM_mask = bbox_prompt_demo.show(image_path=os.path.join(opt.img_path, img_name), Box=boxcam[j_])
            SAM_mask_cam = SAM_mask_cam + tmp_SAM_mask
        # show_cam_on_image(tmpimage, SAM_mask, use_rgb=True, dice=dice_coeff(SAM_mask, label_data),
        #                       name="sam")
        SAM_mask_cam[SAM_mask_cam > 0] = 1
        dice_sam_cam+=dice_coeff(SAM_mask_cam,label_data)


        if dice_coeff(original_grad_cam,label_data)<0.1:
            pic_len+=1
        else :
            dice_pre_sam_cam+=dice_coeff(SAM_mask_cam,label_data)
            dice_pre_sam_crf+=dice_coeff(SAM_mask_crf,label_data)
            if dice_coeff(SAM_mask_cam, label_data) > dice_coeff(SAM_mask_crf, label_data):
                dice_pre_max += dice_coeff(SAM_mask_cam, label_data)
            else:
                dice_pre_max += dice_coeff(SAM_mask_crf, label_data)

        if dice_coeff(SAM_mask_cam,label_data)>dice_coeff(SAM_mask_crf,label_data):
            dice_max_crf_cam+=dice_coeff(SAM_mask_cam,label_data)
        else :
            dice_max_crf_cam += dice_coeff(SAM_mask_crf, label_data)

        # end_time=time.perf_counter()
        # show_cam_on_image(tmpimage, CRF_label, use_rgb=True, dice=dice_coeff(CRF_label, label_data), name="CRF")
        # show_cam_on_image(tmpimage, original_grad_cam, use_rgb=True, dice=dice_coeff(original_grad_cam, label_data),
        #                   name="original_grad_cam")
        # show_cam_on_image(tmpimage, label_data, use_rgb=True, dice=dice_coeff(label_data, label_data),
        #                   name="label_data")
        # show_cam_on_image(tmpimage, SAM_mask_crf, use_rgb=True, dice=dice_coeff(SAM_mask_crf, label_data),
        #                   name="SAM_mask_crf")
        # show_cam_on_image(tmpimage, SAM_mask_cam, use_rgb=True, dice=dice_coeff(SAM_mask_cam, label_data),
        #                   name="SAM_mask_caa")
        # print("第i张图片用时：{}\n".format(end_time-start_time))
        if dice_coeff(SAM_mask_cam, label_data)<0.3:
            # show_cam_on_image(tmpimage, CRF_label, use_rgb=True, dice=dice_coeff(CRF_label, label_data), name="CRF")
            # show_cam_on_image(tmpimage, original_grad_cam, use_rgb=True, dice=dice_coeff(original_grad_cam, label_data),
            #                   name="original_grad_cam")
            # # show_cam_on_image(tmpimage, original_grad_cam, use_rgb=True,
            # #                   name="original_grad_cam")
            # # show_cam_on_image(tmpimage, grad_cam, use_rgb=True,
            # #                   name="grad_cam")
            # show_cam_on_image(tmpimage, label_data, use_rgb=True, dice=dice_coeff(label_data, label_data), name="label_data")
            # show_cam_on_image(tmpimage, SAM_mask_crf, use_rgb=True,dice=dice_coeff(SAM_mask_crf,label_data),
            #                   name="SAM_mask_crf")
            # show_cam_on_image(tmpimage, SAM_mask_cam, use_rgb=True, dice=dice_coeff(SAM_mask_cam, label_data),
            #                   name="SAM_mask_caa")
            # show_cam_on_image(tmpimage, SAM_mask, use_rgb=True, dice=dice_coeff(SAM_mask, label_data),
            #                   name="SAM_mask")
            logger.info(
                "图片:{},dice_sam_crf{},dice_sam_cam:{},dice_cam:{},dice_crf:{}".format(img_name, dice_coeff(SAM_mask_crf, label_data),
                                                                    dice_coeff(SAM_mask_cam, label_data),
                                                                                        dice_coeff(original_grad_cam,label_data),
                                                                                        dice_coeff(CRF_label,label_data)))




        if i%30==0:
            logger.info("\n以第{}张为止，dice_crf_sam_mean:{},dice_caa_sam_mean:{},dice_cam_mean:{},dice_crf_mean:{}".format(
                i,
                dice_sam_crf / i,
                dice_sam_cam/i,
                dice_cam_res/i,
                dice_CAM_CRF/i
            ))
    logger.info("\ndice_crf_sam_mean:{},dice_caa_sam_mean:{},dice_cam_mean:{},dice_crf_mean:{}".format(
        dice_sam_crf / pic_sum,
        dice_sam_cam / pic_sum,
        dice_cam_res / pic_sum,
        dice_CAM_CRF / pic_sum
    ))
    logger.info("\ndice_pre_sam_cam:{},dice_pre_sam_crf:{}".format(dice_pre_sam_crf/(pic_sum-pic_len),dice_pre_sam_cam/(pic_sum-pic_len)))
    logger.info("\npre_max：".format(dice_pre_max/(pic_sum - pic_len)))
    logger.info("\ndice_max：".format(dice_max_crf_cam / (pic_sum)))
    logger.info("\n数据集：".format(pic_sum - pic_len))


def loadLogger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")

    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)

    logger.addHandler(sHandler)

    if not args.not_save:
        work_dir = os.path.join(args.work_dir,
                                time.strftime("%Y.%m.%dT%H %M %S", time.localtime()))
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

        fHandler = logging.FileHandler(work_dir + '/log.txt', mode='w')
        fHandler.setLevel(logging.DEBUG)
        fHandler.setFormatter(formatter)

        logger.addHandler(fHandler)

    return logger
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--img_path',default="F:/brats/val/yes")
    parser.add_argument('--label_path',default="F:/brats/val/label")
    parser.add_argument('--not-save', default=False, action='store_true',
                          help='If yes, only output log to terminal.')
    parser.add_argument('--work-dir', default='./work_dir',
                        help='the work folder for storing results')
    parser.add_argument('--out_path', default="F:/brats/val/rescam")
    opt = parser.parse_args()
    logger = loadLogger(opt)
    main(opt,logger)











