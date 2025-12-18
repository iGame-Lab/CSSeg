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
from CAM.main_vit import show_mask,dice_coeff,UpsampledBackbone
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

    # 结果显示
    # return img
    # plt.imshow(img)
    # plt.show()
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
    device = "cuda:0"
    #CAM模型导入
    CAMmodel=create_model(num_classes=2, has_logits=False)
    CAMmodel=CAMmodel.to(device)
    weights_path = "F:/Brats-class/vision_transformer/weights/model-5.pth"
    CAMmodel.load_state_dict(torch.load(weights_path, map_location=device))
    target_layers = [CAMmodel.blocks[-1].norm1]
    cam = GradCAM(model=CAMmodel,
                  target_layers=target_layers,
                  use_cuda=True,
                  reshape_transform=CAM.main_vit.ReshapeTransform(CAMmodel))
    target_category = 1
    #MedSAM模型导入
    MedSAM_CKPT_PATH = "C:/Users/Administrator/Desktop/MedSAM-main/work_dir/MedSAM/medsam_vit_b.pth"
    medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
    no_caa_dice_SAM=0
    dice_CAM=0
    dice_CAM_CRF=0
    dice_SAM_enhance=0
    dice=0
    checkpoint_url ='https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/vit_jbu_stack_cocostuff.ckpt'
    bbox_prompt_demo = BboxPromptDemo(medsam_model)
    for i,img_name in enumerate(os.listdir(opt.img_path),1):
        img_data = np.load(os.path.join(opt.img_path, img_name))["arr_0"]
        label_data = np.load(os.path.join(opt.label_path, img_name))["arr_0"]
        label_data[label_data > 0] = 1

        grayscale_cam,attn_weight_list = show_mask(os.path.join(opt.img_path, img_name), cam, target_category, 0.5)
        box, cnt = scoremap2bbox(scoremap=grayscale_cam, threshold=0.4, multi_contour_eval=True)
        attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list]  # (b, hxw, hxw)
        attn_weight = torch.stack(attn_weight, dim=0)[-8:]
        attn_weight = torch.mean(attn_weight, dim=0)
        attn_weight = attn_weight[0].cpu().detach()
        attn_weight = attn_weight.float()
        aff_mask = torch.zeros((grayscale_cam.shape[0], grayscale_cam.shape[1]))
        for i_ in range(cnt):
            x0_, y0_, x1_, y1_ = box[i_]
            aff_mask[y0_:y1_, x0_:x1_] = 1
        aff_mask =torch.tensor(cv2.resize(np.array(aff_mask),(14,14)))
        aff_mask = aff_mask.view(1, 14 * 14)
        aff_mat = attn_weight

        trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
        trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)

        for _ in range(2):
            trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
            trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
        trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2
        for _ in range(1):
            trans_mat = torch.matmul(trans_mat, trans_mat)

        trans_mat = trans_mat * aff_mask
        grad_cam=torch.tensor(cv2.resize(np.array(grayscale_cam),(240,240)))
        grayscale_cam = torch.tensor(cv2.resize(np.array(grayscale_cam),(14,14)))
        cam_to_refine = torch.FloatTensor(grayscale_cam)
        cam_to_refine = cam_to_refine.view(-1, 1)
        cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(224 // 16, 224 // 16)
        cam_refined = cam_refined.cpu().numpy().astype(np.float32)
        cam_refined_highres = scale_cam_image([cam_refined], (240, 240))[0]
        temp = []
        for _ in range(3):
            temp.append(img_data)
        tmpimage = np.array(temp)
        tmpimage = tmpimage.transpose(1, 2, 0)


        original_cam_refined_highres=np.array(cam_refined_highres).copy()
        original_grad_cam=np.array(grad_cam).copy()
        cam_refined_highres[cam_refined_highres>=0.8]=1
        cam_refined_highres[cam_refined_highres < 0.8] = 0
        grad_cam[grad_cam>=0.5]=1
        grad_cam[grad_cam<0.5]=0
        dice+=dice_coeff(cam_refined_highres,label_data)
        #mask做完预处理+最大联通处理后，dice_CAM上升0.04，dice_SAM上升0.15
        cam_refined_highres = preprossess(img_data,cam_refined_highres)
        cam_refined_highres=find_max_eras(cam_refined_highres)

        # mask做完预处理+最大联通处理后，dice_CAM上升0.04，dice_SAM上升0.15
        Box=bounding_box(cam_refined_highres)
        dice_CAM+=dice_coeff(cam_refined_highres,label_data)


        n_jobs = multiprocessing.cpu_count()
        # original_cam_refined_highres[original_cam_refined_highres<0.7]=0
        CRF_label=CRF.crf(n_jobs,tmpimage*255,label_data,original_cam_refined_highres,mean_bgr=(img_data.mean(),img_data.mean(),img_data.mean()))
        dice_CAM_CRF+=dice_coeff(CRF_label,label_data)




        SAM_mask=bbox_prompt_demo.show(image_path=os.path.join(opt.img_path, img_name), Box=Box)
        dice_SAM_enhance+=dice_coeff(SAM_mask,label_data)
        grad_cam = preprossess(img_data, grad_cam)
        grad_cam = find_max_eras(grad_cam)
        Box = bounding_box(grad_cam)
        no_caa_SAM_mask=bbox_prompt_demo.show(image_path=os.path.join(opt.img_path, img_name), Box=Box)
        CRF_label=preprossess(img_data,CRF_label)
        CRF_label=find_max_eras(CRF_label)
        Box=bounding_box(CRF_label)
        box_test = [Box[0] - 10, Box[1] - 10, Box[2] + 10, Box[3] + 10]
        no_caa_dice_SAM+=dice_coeff(no_caa_SAM_mask,label_data)
        SAM_CRF_mask=bbox_prompt_demo.show(image_path=os.path.join(opt.img_path, img_name), Box=box_test)




        if float(dice_coeff(SAM_mask, label_data))<float(dice_coeff(no_caa_SAM_mask, label_data)) and (float(dice_coeff(cam_refined_highres, label_data))>float(dice_coeff(grad_cam, label_data))):
            show_cam_on_image(tmpimage, original_cam_refined_highres, use_rgb=True,name="caa_cam")
            show_cam_on_image(tmpimage, original_grad_cam, use_rgb=True,  name="grad_cam")
            show_cam_on_image(tmpimage, cam_refined_highres, use_rgb=True,dice=dice_coeff(cam_refined_highres, label_data),name="caa_cam")
            show_cam_on_image(tmpimage, grad_cam, use_rgb=True,dice=dice_coeff(grad_cam, label_data),name="grad_cam")
            show_cam_on_image(tmpimage,CRF_label,use_rgb=True,dice=dice_coeff(CRF_label, label_data),name="CRF_cam")
            show_cam_on_image(tmpimage, SAM_mask, use_rgb=True,dice=dice_coeff(SAM_mask, label_data),name="caa_sam")
            show_cam_on_image(tmpimage, no_caa_SAM_mask, use_rgb=True,dice=dice_coeff(no_caa_SAM_mask, label_data),name="sam")

        # if(dice_coeff(SAM_mask,label_data)<0.8)  :
        #     fig,axes = plt.subplots(1, 1, figsize=(5, 5))
        #     fig.canvas.header_visible = False
        #     fig.canvas.footer_visible = False
        #     fig.canvas.toolbar_visible = False
        #     fig.canvas.resizable = False
        #     plt.tight_layout()
        #     img_data = np.array(img_data)
        #     temp = []
        #     for i in range(3):
        #         temp.append(img_data)
        #     img_data = np.array(temp)
        #     img_data = img_data.transpose(1, 2, 0)
        #     plt.tight_layout()
        #     axes.imshow(img_data)
        #     show_mask_image(mask,axes,True,0.65)
        #     plt.show()
        #     fig, axes = plt.subplots(1, 1, figsize=(5, 5))
        #     fig.canvas.header_visible = False
        #     fig.canvas.footer_visible = False
        #     fig.canvas.toolbar_visible = False
        #     fig.canvas.resizable = False
        #     plt.tight_layout()
        #     axes.imshow(img_data)
        #     show_mask_image(label_data,axes,True,0.65)
        #     plt.show()
        #     fig, axes = plt.subplots(1, 1, figsize=(5, 5))
        #     fig.canvas.header_visible = False
        #     fig.canvas.footer_visible = False
        #     fig.canvas.toolbar_visible = False
        #     fig.canvas.resizable = False
        #     plt.tight_layout()
        #     axes.imshow(img_data)
        #     show_mask_image(SAM_mask, axes, True, 0.65)
        #     plt.show()
        if i%30==0:
            logger.info("以第 {}张图片为止时dice:{} ,dice_CAM:{} ,dice_CAM_CRF:{} dice_SAM_enhance:{}".format(i,dice/(i),dice_CAM/(i),dice_CAM_CRF/(i),dice_SAM_enhance/(i)))
            # logger.info(
            #     "以第 {}张图片为止时 dice_CAM:{} ".format(i, dice_CAM / i))


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
    opt = parser.parse_args()
    logger = loadLogger(opt)
    main(opt,logger)

































