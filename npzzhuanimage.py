import numpy as np
from tqdm import tqdm
from MedSAM.segment_anything import sam_model_registry
from MedSAM.demo import BboxPromptDemo
import os
import argparse
from CAM.main_vit import show_mask, dice_coeff
from Utils import *
import logging
import time
from CAM.utils import scoremap2bbox
from skimage import measure
import multiprocessing


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


def preprossess(img, mask):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i][j] <= 1e-9):
                mask[i][j] = 0
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
    if max_index == 0:
        return img
    img[img != max_index] = 0
    img[img == max_index] = 1

    return img


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
    if max_index == 0:
        return img
    img[img != max_index] = 0
    img[img == max_index] = 1

    return img


def find_all_eras(mask):
    # 图像读取
    img = np.array(mask, dtype=np.uint8)  # 确保输入类型为 uint8
    img[img != 0] = 1  # 图像二值化

    # 图像实例化
    labeled_img = measure.label(img, connectivity=2)
    props = measure.regionprops(labeled_img)

    # 根据区域的面积对 props 进行排序，按面积从大到小排序
    props_sorted = sorted(props, key=lambda x: x.area, reverse=True)

    # 保存每个区域的二值图
    all_regions = []

    # 遍历每个区域，将其提取为二值图
    for prop in props_sorted:
        region_mask = np.zeros_like(labeled_img, dtype=np.uint8)  # 明确指定 dtype 为 uint8
        region_mask[labeled_img == prop.label] = 1
        all_regions.append(region_mask)

    return all_regions


def show_mask_image(mask, ax, random_color=False, alpha=0.95):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def main(opt):

    image_path = "F:/brats/test/yes"
    for image_name in os.listdir(image_path):
        video_name = image_name.split('_')[0]+'_'+image_name.split('_')[1]
        npztoimage(os.path.join(image_path,image_name),os.path.join(image_path,"valyes",video_name),image_name.split('.')[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--img_path', default="/root/data1/brats/val/yes")
    parser.add_argument('--label_path', default="/root/data1/brats/val/label")
    # parser.add_argument('--cam_dir', default="F:/brats/val/cam")
    parser.add_argument('--video_dir', default="/root/data1/brats/val/valyes")
    parser.add_argument('--not-save', default=False, action='store_true',
                        help='If yes, only output log to terminal.')
    parser.add_argument('--work-dir', default='./work_dir',
                        help='the work folder for storing results')
    parser.add_argument('--out_path', default="/root/data1/brats/val/newcam")
    parser.add_argument('--video_out_path', default="/root/data1/brats/video_medsam_twosam_addcam_1018")
    parser.add_argument('--sam1-dir', default="/root/data1/brats/box_stack_sam1")
    parser.add_argument('--npz-dir', default="/root/data1/brats/twosam_addcam_1018")
    opt = parser.parse_args()
    main(opt)
