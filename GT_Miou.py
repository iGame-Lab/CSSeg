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


def show_mask_image(mask, ax, random_color=False, alpha=0.95):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
def calculate_iou(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个二值矩阵的IOU相似度"""
    intersection = np.sum((a == 1) & (b == 1))
    union = np.sum((a == 1) | (b == 1))
    return intersection / union if union != 0 else 0.0


def compute_similarity(label_list: list) -> float:
    """计算每个元素与相邻帧的IOU相似度"""
    n = len(label_list)
    if n <= 1:
        return 0.0  # 空列表或单元素直接返回0.0

    # 计算所有相邻帧对的IOU
    iou_list = [calculate_iou(label_list[i], label_list[i + 1]) for i in range(n - 1)]

    # 生成每个元素的相似度结果
    similarity = []
    for i in range(n):
        if i == 0:
            sim = iou_list[0]  # 首帧：仅与第二帧计算
        elif i == n - 1:
            sim = iou_list[-1]  # 末帧：仅与前一帧计算
        else:
            sim = (iou_list[i - 1] + iou_list[i]) / 2  # 中间帧：前后两对IOU的平均
        similarity.append(round(sim, 4))  # 保留4位小数
    return sum(similarity) / len(similarity) if similarity else 0.0
def main(opt, logger):

    device = "cuda:0"

    MedSAM_CKPT_PATH = "C:/Users/Administrator/Desktop/MedSAM-main/work_dir/MedSAM/medsam_vit_b.pth"
    medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
    # 初始化dice
    dice_caa = 0
    dice_sam_caa = 0
    dice_out = 0
    dice_in = 0
    pic_out_len = 0
    pic_in_len = 0

    # 初始化MedSAM
    bbox_prompt_demo = BboxPromptDemo(medsam_model)
    # 获得视频列表
    video_path = build_video_list(opt.video_dir)
    cnt = 0
    # 相似度列表
    similarity_label_list = []
    similarity_cam_list = []
    similarity_sam_list = []
    similarity_AMS_list = []
    dice_sam_list = []
    dice_cam_list =[]
    dice_AMS_list = []
    # 相似度列表
    parm_max = opt.hyperparameters
    dingwei_error=0
    for video_dir in tqdm(video_path, desc="Processing Videos"):  # 遍历所以的视频

        video_name = video_dir.split('\\')[-1]
        video_output_path = os.path.join(opt.video_out_path, video_dir.split('/')[-1])
        if not os.path.exists(video_output_path):
            os.makedirs(video_output_path)


        frame_names, frame = build_frame_list(video_dir)
        label_path = opt.label_path
        # 得到视频全部图片的label
        label = build_label_list(label_path, frame_names)
        sam_mask = build_sam_mask_list(opt.sam1_dir,frame_names)
        AMS_mask = build_sam_mask_list(opt.AMS_dir,frame_names)


        similarity_label = compute_similarity(label)
        similarity_label_list.append(similarity_label)

        similarity_sam = compute_similarity(sam_mask)
        similarity_sam_list.append(similarity_sam)

        similarity_AMS = compute_similarity(AMS_mask)
        similarity_AMS_list.append(similarity_AMS)


        cam_pro_mask=[]
        for img_name, label_data in zip(frame_names, label):
            grad_cam = np.load(os.path.join(opt.out_path, img_name.split('.')[0] + '.npz'))['original_cam']
            caa_grad_cam = np.load(os.path.join(opt.out_path, img_name.split('.')[0] + '.npz'))['caa_cam']
            original_grad_cam = np.array(caa_grad_cam).copy()
            original_grad_cam[original_grad_cam < parm_max] = 0
            original_grad_cam[grad_cam < (1 - parm_max)] = 0

            caa_grad_cam[caa_grad_cam < parm_max ] = 0
            caa_grad_cam[caa_grad_cam >= parm_max ] = 1
            # caa_grad_cam[grad_cam < (1 - parm_max)] = 0

            # caa_grad_cam = find_max_eras(caa_grad_cam)
            # 保存处理后的cam_mask
            cam_pro_mask.append(caa_grad_cam.copy())
        similarity_cam= compute_similarity(cam_pro_mask)
        similarity_cam_list.append(similarity_cam)

        dice_sam = []
        dice_cam = []
        dice_AMS = []
        for cam_mask_tmp,sam_mask_tmp,label_tmp,AMS_mask_tmp in zip(cam_pro_mask,sam_mask,label,AMS_mask):
            dice_sam.append(dice_coeff(sam_mask_tmp,label_tmp))
            dice_cam.append(dice_coeff(cam_mask_tmp, label_tmp))
            dice_AMS.append(dice_coeff(AMS_mask_tmp,label_tmp))
            if dice_coeff(sam_mask_tmp,label_tmp):
                dingwei_error+=1
        dice_sam_list.append(sum(dice_sam) / len(dice_sam) if dice_sam else 0.0)
        dice_cam_list.append(sum(dice_cam) / len(dice_cam) if dice_cam else 0.0)
        dice_AMS_list.append(sum(dice_AMS) / len(dice_AMS) if dice_AMS else 0.0)
        logging.info(f"{video_name}:"
                     f"similarity_label :{round(similarity_label, 4)},"
                     f"similarity_sam :{round(similarity_sam, 4)},"
                     f"similarity_cam :{round(similarity_cam, 4)},"
                     f"similarity_AMS :{round(similarity_AMS, 4)},"
                     f"sam_gt_dice :{round(sum(dice_sam) / len(dice_sam) if dice_sam else 0.0, 4)},"
                     f"cam_gt_dice :{round(sum(dice_cam) / len(dice_cam) if dice_cam else 0.0, 4)},"
                     f"AMS_gt_dice :{round(sum(dice_AMS) / len(dice_AMS) if dice_AMS else 0.0, 4)},")
    logging.info(f"similarity_label  :{round(sum(similarity_label_list) / len(similarity_label_list) if similarity_label_list else 0.0, 4)},"
                 f"similarity_cam  :{round(sum(similarity_cam_list) / len(similarity_cam_list) if similarity_cam_list else 0.0, 4)},"
                 f"similarity_sam  :{round(sum(similarity_sam_list) / len(similarity_sam_list) if similarity_sam_list else 0.0, 4)},"
                 f"similarity_AMS  :{round(sum(similarity_AMS_list) / len(similarity_AMS_list) if similarity_AMS_list else 0.0, 4)},"
                 f"dice_sam_gt  :{round(sum(dice_sam_list) / len(dice_sam_list) if dice_sam_list else 0.0, 4)},"
                 f"dice_cam_gt  :{round(sum(dice_cam_list) / len(dice_cam_list) if dice_cam_list else 0.0, 4)},"
                 f"dice_AMS_gt  :{round(sum(dice_AMS_list) / len(dice_AMS_list) if dice_AMS_list else 0.0, 4)},")
    logging.info(dingwei_error)


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
    parser.add_argument('--img_path', default="F:/brats/val/yes")
    parser.add_argument('--label_path', default="F:/brats/val/label")
    parser.add_argument('--video_dir', default="F:/brats/valyes")
    parser.add_argument('--not-save', default=False, action='store_true',
                        help='If yes, only output log to terminal.')
    parser.add_argument('--work-dir', default='./work_dir',
                        help='the work folder for storing results')
    parser.add_argument('--out_path', default="F:/brats/val/newcam")
    parser.add_argument('--video_out_path', default="F:/brats/val/2025_0514/video_result")
    parser.add_argument('--sam1-dir', default="F:/brats/val/2025消融实验0.7有2次SAM")
    parser.add_argument('--AMS-dir', default="F:/brats/val/AMS")
    parser.add_argument('--hyperparameters', default=0.8)
    parser.add_argument('--npz-dir', default="F:/brats/2025/result")
    opt = parser.parse_args()
    logger = loadLogger(opt)
    main(opt, logger)
