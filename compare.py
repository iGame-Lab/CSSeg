
from tqdm import tqdm

import os
import argparse
from CAM.main_vit import show_mask, dice_coeff
from Utils import *
import logging
import time

from skimage import measure



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


def show_mask_image(mask, ax, random_color=False, alpha=0.95):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def main(opt, logger):
    # 初始化dice
    dice_compareone = 0
    dice_comparetwo = 0


    # 获得视频列表
    video_path = build_video_list(opt.video_dir)
    cnt = 0

    for video_dir in tqdm(video_path, desc="Processing Videos"):  # 遍历所以的视频

        video_name = video_dir.split('\\')[-1]
        cam_list = []
        label_res_list = []
        cam_box_list = []
        SAM_list = []
        cam_box_dice = []
        cam_dice = []
        sam2_dice = []
        label_dice = []
        # 保存全部的box框
        box_list = []
        # 保存原始cam热力图
        cam_ori_mask = []
        cam_stack_mask = np.zeros((240, 240))
        # 保存处理后的cam热力图
        cam_pro_mask = []
        cam_pro_stack_mask = np.zeros((240, 240))
        # 保存box圈主的区域
        box_mask = []
        box_stack_mask = np.zeros((240, 240))
        # 保存label叠加
        label_mask = []
        label_stack_mask = np.zeros((240, 240))
        # 得到视频全部图片帧
        frame_names, frame = build_frame_list(video_dir)
        cam_path = opt.cam_dir
        label_path = opt.label_path
        # 得到视频全部图片的label
        label = build_label_list(label_path, frame_names)

        current_dice_compareone = 0
        current_dice_comparetwo = 0

        rescnt = -1
        for img_name, label_data in zip(frame_names, label):
            cnt+=1
            rescnt += 1
            cam_dice.append(0)
            img_data = np.load(os.path.join(opt.img_path, img_name.split('.')[0] + '.npz'))["arr_0"]

            iou = np.load(os.path.join(opt.compareone,img_name.split('.')[0] + '.npz'))["caa_sam1_stack_pred"]
            cam_pro_stack_mask = np.load(os.path.join(opt.comparetwo, img_name.split('.')[0] + '.npz'))["caa_sam1_stack_pred"]

            current_dice_compareone+=dice_coeff(iou,label_data)
            current_dice_comparetwo+=dice_coeff(cam_pro_stack_mask,label_data)
            dice_compareone+=dice_coeff(iou,label_data)
            dice_comparetwo+=dice_coeff(cam_pro_stack_mask,label_data)


        video_output_path = os.path.join(opt.video_out_path, video_dir.split('\\')[-1])
        if current_dice_comparetwo<current_dice_compareone:
            logging.info(
                f"{video_name}堆叠cam_pro：dice  {round(current_dice_compareone / (rescnt+1), 4)},iou为0的取上一帧box:dice  {round(current_dice_comparetwo / (rescnt+1), 4)}")


    logging.info(f"堆叠cam_pro：dice  {round(dice_compareone/cnt,4)},iou为0的取上一帧box:dice  {round(dice_comparetwo/cnt,4)}")


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
    parser.add_argument('--cam_dir', default="F:/brats/val/cam")
    parser.add_argument('--video_dir', default="F:/brats/valyes")
    parser.add_argument('--not-save', default=False, action='store_true',
                        help='If yes, only output log to terminal.')
    parser.add_argument('--work-dir', default='./work_dir',
                        help='the work folder for storing results')
    parser.add_argument('--out_path', default="F:/brats/val/cam")
    parser.add_argument('--video_out_path', default="F:/brats/video_medsam_stack_down02_adddistance")
    parser.add_argument('--sam1-dir', default="F:/brats/box_stack_sam1")
    parser.add_argument('--npz-dir', default="F:/brats/box_stack_sam1_stack/add_distance")
    parser.add_argument('--compareone', default="F:/brats/box_stack_sam1_stack/cam_mask_pro_down02")
    parser.add_argument('--comparetwo', default="F:/brats/box_stack_sam1_stack/与堆叠最大值求iou，若为0")
    opt = parser.parse_args()
    logger = loadLogger(opt)
    main(opt, logger)
