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


def main(opt, logger):

    device = "cuda:0"

    MedSAM_CKPT_PATH = "/root/data1/brats/medSAMpth/medsam_vit_b.pth"
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

    for video_dir in tqdm(video_path, desc="Processing Videos"):  # 遍历所以的视频

        video_name = video_dir.split('\\')[-1]

        video_output_path = os.path.join(opt.video_out_path, video_dir.split('\\')[-1])
        if not os.path.exists(video_output_path):
            os.makedirs(video_output_path)
        cam_list = []
        label_res_list = []
        cam_box_list = []
        SAM_list = []
        cam_box_dice = []
        cam_dice = []
        sam2_dice = []
        label_dice = []
        # 保存SAM_stack
        SAM_stack = []
        # 保存全部的box框
        box_list = []
        # 保存原始cam热力图
        cam_ori_mask = []
        cam_stack_mask = np.zeros((256, 256))
        # 保存处理后的cam热力图
        cam_pro_mask = []
        cam_pro_stack_mask = np.zeros((256, 256))
        # 保存sam处理图
        sam_pro_mask = []
        # 保存box圈主的区域
        box_mask = []
        box_stack_mask = np.zeros((256, 256))
        # 保存label叠加
        label_mask = []
        label_stack_mask = np.zeros((256, 256))
        # 得到视频全部图片帧
        frame_names, frame = build_frame_list(video_dir)
        label_path = opt.label_path
        # 得到视频全部图片的label
        label = build_label_list(label_path, frame_names)

        # 得到视频全部图片的cam热力图
        img_list = []
        img_data_list=[]
        for img_name, label_data in zip(frame_names, label):

            cnt += 1
            img_data = np.load(os.path.join(opt.img_path, img_name.split('.')[0] + '.npz'))["arr_0"]
            img_data = cv2.resize(np.array(img_data), (256, 256), interpolation=cv2.INTER_NEAREST)
            img_data_list.append(img_data.copy())
            grad_cam = np.load(os.path.join(opt.out_path, img_name.split('.')[0] + '.npz'))['original_cam']
            caa_grad_cam = np.load(os.path.join(opt.out_path, img_name.split('.')[0] + '.npz'))['caa_cam']
            sam_mask = np.load(os.path.join("/root/data1/chaos/computer_dice", img_name.split('.')[0] + '.npz'))[
                'caa_sam1_stack_pred']
            sam_pro_mask.append(sam_mask)
            # 保存原始的cam_mask
            cam_ori_mask.append(caa_grad_cam.copy())
            temp = []
            for _ in range(3):
                temp.append(img_data)
            tmpimage = np.array(temp)
            tmpimage = tmpimage.transpose(1, 2, 0)
            img_list.append(tmpimage)
            original_grad_cam = np.array(caa_grad_cam).copy()
            original_grad_cam[original_grad_cam < 0.7] = 0
            original_grad_cam[grad_cam < 0.3] = 0
            caa_grad_cam = preprossess(img_data, caa_grad_cam)

            caa_grad_cam[caa_grad_cam < 0.7 * caa_grad_cam.max()] = 0
            caa_grad_cam[caa_grad_cam >= 0.7 * caa_grad_cam.max()] = 1

            # 保存处理后的cam_mask
            cam_pro_mask.append(caa_grad_cam.copy())

            boxcaa, cntcaa = scoremap2bbox(scoremap=caa_grad_cam, threshold=0.4, multi_contour_eval=True)
            # 保存box圈主的区域
            zero_array = np.zeros_like(caa_grad_cam)
            # 使用box的坐标，圈住的区域内赋值为1
            x1, y1, x2, y2 = boxcaa[0]
            tempbox = []
            tempbox.append(boxcaa[0])
            # box_list.append(boxcaa[0])
            zero_array[y1:y2 + 1, x1:x2 + 1] = 1
            for j_ in range(1, cntcaa):
                x1, y1, x2, y2 = boxcaa[j_]
                tempbox.append(boxcaa[j_])
                zero_array[y1:y2 + 1, x1:x2 + 1] = 1
            box_list.append(np.array(tempbox))
            box_mask.append(zero_array)
            label_mask.append(cv2.resize(np.array(label_data), (256, 256), interpolation=cv2.INTER_NEAREST))

        SAM_MASK = sam_pro_mask.copy()

        label_stack_mask = normalize_sum_of_arrays(label_mask)
        cam_stack_mask = normalize_sum_of_arrays(cam_ori_mask)
        cam_pro_stack_mask = normalize_sum_of_arrays(cam_pro_mask)
        sam_pro_mask = normalize_sum_of_arrays(sam_pro_mask)
        SAM_MASK = normalize_sum_of_arrays(SAM_MASK)

        if not os.path.exists(os.path.join(video_output_path, "STACK")):
            os.makedirs(os.path.join(video_output_path, "STACK"))
        img_uint8 = np.uint8(
            255 * np.clip(show_cam_on_image(tmpimage, cam_pro_stack_mask, returncam=True, use_rgb=False), 0, 1))
        cv2.imwrite(os.path.join(video_output_path, "STACK", "cam_stack.png"), img_uint8)

        img_uint8 = np.uint8(
            255 * np.clip(show_cam_on_image(tmpimage, label_stack_mask, returncam=True, use_rgb=False), 0, 1))
        cv2.imwrite(os.path.join(video_output_path, "STACK", "label_stack.png"), img_uint8)

        original_cam_pro_stack_mask = cam_pro_stack_mask.copy()
        maxvalue = original_cam_pro_stack_mask.max()
        original_cam_pro_stack_mask[original_cam_pro_stack_mask == maxvalue] = 1
        original_cam_pro_stack_mask[original_cam_pro_stack_mask != maxvalue] = 0

        cam_pro_stack_mask[cam_pro_stack_mask >= 0.3] = 1
        cam_pro_stack_mask[cam_pro_stack_mask < 0.3] = 0
        # boxsum, cntsum = scoremap2bbox(scoremap=cam_pro_stack_mask, threshold=0.4, multi_contour_eval=True)

        original_sam_stack_mask = sam_pro_mask.copy()
        maxvalue = original_sam_stack_mask.max()
        original_sam_stack_mask[original_sam_stack_mask != maxvalue] = 0
        original_sam_stack_mask[original_sam_stack_mask == maxvalue] = 1

        sam_pro_mask[sam_pro_mask > 0] = 1
        sam_pro_mask[sam_pro_mask <= 0] = 0
        boxsum, cntsum = scoremap2bbox(scoremap=sam_pro_mask, threshold=0.4, multi_contour_eval=True)

        SAM_MASK[SAM_MASK > 0.3] = 1
        SAM_MASK[SAM_MASK <= 0.3] = 0
        SAM_MASK = find_max_eras(SAM_MASK)

        for index in np.arange(len(box_list)):
            cam_ori = cam_ori_mask[index].copy()
            maxvalue = cam_ori.max()
            cam_ori[cam_ori != maxvalue] = 0
            cam_ori[cam_ori == maxvalue] = 1
            if index == 0:  # 确保不是最后一个元素
                if not np.any(np.logical_and(cam_ori == 1, SAM_MASK == 1)):
                    box_list[index] = box_list[index + 1]  # 将当前的 box 替换为下一个 box 的值
                    cam_ori_mask[index] = cam_ori_mask[index + 1]
            else:
                if not np.any(np.logical_and(cam_ori == 1, SAM_MASK == 1)):
                    box_list[index] = box_list[index - 1]
                    cam_ori_mask[index] = cam_ori_mask[index - 1]

        for index in np.arange(len(box_list) - 1, -1, -1):  # 从最后一个元素开始反向遍历
            cam_ori = cam_ori_mask[index].copy()
            maxvalue = cam_ori.max()
            cam_ori[cam_ori != maxvalue] = 0
            cam_ori[cam_ori == maxvalue] = 1
            if index != len(box_list) - 1:  # 确保不是最后一个元素
                if not np.any(np.logical_and(cam_ori == 1, SAM_MASK == 1)):
                    box_list[index] = box_list[index + 1]  # 将当前的 box 替换为下一个 box 的值
                    cam_ori_mask[index] = cam_ori_mask[index + 1]
            else:
                if not np.any(np.logical_and(cam_ori == 1, SAM_MASK == 1)):
                    box_list[index] = box_list[index - 1]
                    cam_ori_mask[index] = cam_ori_mask[index - 1]

        # 将全部图片的框与最佳框进行取交集
        for i, box in enumerate(box_list):
            box_list_res = []
            for one_box in box:
                for sum_box in boxsum:
                    tmpbox = get_intersection(one_box, sum_box)
                    if tmpbox is not None:
                        box_list_res.append(tmpbox)
            if len(box_list_res) != 0:
                box_list[i] = np.array(box_list_res)
            else:
                if i == 0:
                    box_list[i] = box_list[i + 1]
                else:
                    box_list[i] = box_list[i - 1]
        box_list = fill_none_with_nearest(box_list)

        for index in np.arange(0, len(box_list)):
            if index != 0:
                box = box_list[index]
                zero_array = np.zeros_like(original_sam_stack_mask)
                for j_ in range(0, len(box)):
                    x1, y1, x2, y2 = box[j_]
                    zero_array[y1:y2 + 1, x1:x2 + 1] = 1
                if compute_iou(zero_array, original_sam_stack_mask) == 0 and process_image(img_data_list[index],box=box_list[index])<process_image(img_data_list[index],box=box_list[index-1]):
                    box_list[index] = box_list[index - 1]
            else:
                box = box_list[index]
                zero_array = np.zeros_like(original_sam_stack_mask)
                for j_ in range(0, len(box)):
                    x1, y1, x2, y2 = box[j_]
                    zero_array[y1:y2 + 1, x1:x2 + 1] = 1
                if compute_iou(zero_array, original_sam_stack_mask) == 0 and process_image(img_data_list[index],box=box_list[index])<process_image(img_data_list[index],box=box_list[index+1]):
                    box_list[index] = box_list[index + 1]

        for index in np.arange(len(box_list) - 1, -1, -1):  # 从最后一个元素开始反向遍历
            if index != len(box_list) - 1:  # 确保不是最后一个元素
                box = box_list[index]
                zero_array = np.zeros_like(original_sam_stack_mask)
                for j_ in range(0, len(box)):
                    x1, y1, x2, y2 = box[j_]
                    zero_array[y1:y2 + 1, x1:x2 + 1] = 1
                if compute_iou(zero_array, original_sam_stack_mask) == 0 and process_image(img_data_list[index],box=box_list[index])<process_image(img_data_list[index],box=box_list[index+1]):
                    box_list[index] = box_list[index + 1]  # 将当前的 box 替换为下一个 box 的值
            else:
                box = box_list[index]
                zero_array = np.zeros_like(original_sam_stack_mask)
                for j_ in range(0, len(box)):
                    x1, y1, x2, y2 = box[j_]
                    zero_array[y1:y2 + 1, x1:x2 + 1] = 1
                if compute_iou(zero_array, original_sam_stack_mask) == 0 and process_image(img_data_list[index],box=box_list[index])<process_image(img_data_list[index],box=box_list[index-1]):
                    box_list[index] = box_list[index - 1]  # 将最后一个元素替换为前一个元素的值

        rescnt = -1
        SAM_MASK_LIST=[]


        for img_name, label_data in zip(frame_names, label):
            rescnt += 1
            cam_dice.append(0)
            img_data = np.load(os.path.join(opt.img_path, img_name.split('.')[0] + '.npz'))["arr_0"]
            target_size = img_data.shape
            img_data = cv2.resize(np.array(img_data), (256, 256), interpolation=cv2.INTER_NEAREST)
            img_data_mask = img_data.copy()
            img_data_mask[img_data_mask > 0] = 1
            grad_cam = np.load(os.path.join(opt.out_path, img_name.split('.')[0] + '.npz'))['original_cam']
            caa_grad_cam = np.load(os.path.join(opt.out_path, img_name.split('.')[0] + '.npz'))['caa_cam']
            temp = []
            for _ in range(3):
                temp.append(img_data)
            tmpimage = np.array(temp)
            tmpimage = tmpimage.transpose(1, 2, 0)
            cam_list.append(show_cam_on_image(tmpimage, caa_grad_cam, returncam=True))

            original_grad_cam = np.array(caa_grad_cam).copy()

            original_grad_cam[original_grad_cam < 0.7] = 0
            original_grad_cam[grad_cam < 0.3] = 0
            caa_grad_cam = preprossess(img_data, caa_grad_cam)
            caa_grad_cam[caa_grad_cam < 0.7] = 0
            caa_grad_cam[caa_grad_cam >= 0.7] = 1
            caa_grad_cam[grad_cam < 0.3] = 0

            caa_grad_cam = find_max_eras(caa_grad_cam)
            zero_array = np.zeros_like(caa_grad_cam)
            for j_ in box_list[rescnt]:
                boxcaa = j_
                x1, y1, x2, y2 = boxcaa
                zero_array[y1:y2 + 1, x1:x2 + 1] = 1
            caa_grad_cam[zero_array == 0] = 0
            cam_box_list.append(show_cam_on_image(tmpimage.copy(), caa_grad_cam, box=boxcaa, returncam=True))
            cam_box_dice.append(dice_coeff(caa_grad_cam, cv2.resize(np.array(label_data), (256, 256), interpolation=cv2.INTER_NEAREST)))
            dice = dice_coeff(caa_grad_cam,cv2.resize(np.array(label_data), (256, 256), interpolation=cv2.INTER_NEAREST))
            dice_caa += dice

            label_res_list.append(show_cam_on_image(tmpimage.copy(), cv2.resize(np.array(label_data), (256, 256), interpolation=cv2.INTER_NEAREST), returncam=True))
            label_dice.append(1)
            SAM_mask_caa = bbox_prompt_demo.show(image_path=os.path.join(opt.img_path, img_name.split('.')[0] + '.npz'),
                                                 Box=scale_box_to_target(box_list[rescnt][0],target_size))
            for j_ in range(1, len(box_list[rescnt])):
                tmp_SAM_mask = bbox_prompt_demo.show(
                    image_path=os.path.join(opt.img_path, img_name.split('.')[0] + '.npz'), Box=scale_box_to_target(box_list[rescnt][j_],target_size))
                SAM_mask_caa = SAM_mask_caa + tmp_SAM_mask
            SAM_mask_caa[SAM_mask_caa > 0] = 1
            SAM_mask_caa = cv2.resize(np.array(SAM_mask_caa), (256, 256), interpolation=cv2.INTER_NEAREST)
            SAM_mask_caa = preprossess(img_data, SAM_mask_caa)
            SAM_MASK_LIST.append(SAM_mask_caa)



        SAM_MASK_LIST = connection_interval_processing(img_data_list,SAM_MASK_LIST)
        rescnt=-1
        for img_name, label_data in zip(frame_names, label):
            rescnt+=1
            dice_sam = dice_coeff(SAM_MASK_LIST[rescnt], cv2.resize(np.array(label_data), (256, 256), interpolation=cv2.INTER_NEAREST))
            dice_sam_caa += dice_sam
            sam2_dice.append(dice_sam)
            if not os.path.exists(opt.npz_dir):
                os.makedirs(opt.npz_dir)
            img_data = np.load(os.path.join(opt.img_path, img_name.split('.')[0] + '.npz'))["arr_0"]
            img_data = cv2.resize(np.array(img_data), (256, 256), interpolation=cv2.INTER_NEAREST)
            temp = []
            for _ in range(3):
                temp.append(img_data)
            tmpimage = np.array(temp)
            tmpimage = tmpimage.transpose(1, 2, 0)
            img_list.append(tmpimage)
            SAM_list.append(show_cam_on_image(tmpimage.copy(), SAM_MASK_LIST[rescnt], returncam=True))
            np.savez(os.path.join(opt.npz_dir, img_name.split('.')[0]),
                     caa_sam1_stack_pred=SAM_MASK_LIST[rescnt]
                     )
        logging.info(f"{video_name}:dice_cam:{round(np.array(cam_box_dice).sum() / cam_box_dice.__len__(), 6)}"
                     f"dice_sam:{round(np.array(sam2_dice).sum() / sam2_dice.__len__(), 6)},dice_cam_sum: {round(dice_caa / cnt, 6)} dice_sam_sum: {round(dice_sam_caa / cnt, 6)}")


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
    parser.add_argument('--img_path',default="/root/data1/chaos/yes")
    parser.add_argument('--label_path',default="/root/data1/chaos/label")
    # parser.add_argument('--cam_dir', default="F:/brats/val/cam")
    parser.add_argument('--video_dir', default="/root/data1/chaos/valyes")
    parser.add_argument('--not-save', default=False, action='store_true',
                        help='If yes, only output log to terminal.')
    parser.add_argument('--work-dir', default='./work_dir',
                        help='the work folder for storing results')
    parser.add_argument('--out_path',default="/root/data1/chaos/newcam")
    parser.add_argument('--video_out_path', default="/root/data1/chaos/video_twosam_addcam_1021")
    parser.add_argument('--sam1-dir', default="/root/data1/chaos/box_stack_sam1")
    parser.add_argument('--npz-dir', default="/root/data1/chaos/val/twosam_addcam_1021")
    opt = parser.parse_args()
    logger = loadLogger(opt)
    main(opt, logger)
