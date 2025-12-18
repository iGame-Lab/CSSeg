import numpy as np
from tqdm import tqdm
from MedSAM.segment_anything import sam_model_registry
from MedSAM.demo import BboxPromptDemo
import os
import argparse
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

    if max_index == 0:
        return img
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
    #初始化dice
    dice_caa=0
    dice_sam_caa=0
    dice_out=0
    dice_in=0
    pic_out_len=0
    pic_in_len=0
    #初始化MedSAM
    bbox_prompt_demo = BboxPromptDemo(medsam_model)
    #获得视频列表
    video_path = build_video_list(opt.video_dir)
    cnt=0
    for video_dir in tqdm(video_path, desc="Processing Videos"):  # 遍历所以的视频

        sam_iou_mask=[]
        cam_iou_mask=[]
        ori_iou=[]
        label_iou_mask=[]
        video_name = video_dir.split('\\')[-1]
        if video_name == 'BraTS2021_00071':
            print(1)
        else:
            continue
        cam_list = []
        label_res_list = []
        cam_box_list = []
        SAM_list = []
        cam_box_dice = []
        cam_dice = []
        sam2_dice = []
        label_dice = []
        # 得到视频全部图片帧
        frame_names, frame = build_frame_list(video_dir)
        cam_path = opt.cam_dir
        label_path = opt.label_path
        # 得到视频全部图片的label
        label = build_label_list(label_path, frame_names)
        # 得到视频全部图片的cam热力图
        for img_name, label_data in zip(frame_names, label):
            cnt+=1
            cam_dice.append(0)
            img_data = np.load(os.path.join(opt.img_path, img_name.split('.')[0]+'.npz'))["arr_0"]
            grad_cam = np.load(os.path.join(opt.out_path, img_name.split('.')[0]+'.npz'))['original_cam']
            caa_grad_cam = np.load(os.path.join(opt.out_path, img_name.split('.')[0]+'.npz'))['caa_cam']
            temp = []
            for _ in range(3):
                temp.append(img_data)
            tmpimage = np.array(temp)
            tmpimage = tmpimage.transpose(1, 2, 0)
            cam_list.append(show_cam_on_image(tmpimage , caa_grad_cam, returncam=True))

            original_grad_cam = np.array(caa_grad_cam).copy()
            original_grad_cam[original_grad_cam < 0.8] = 0
            original_grad_cam[grad_cam < 0.2] = 0
            caa_grad_cam[caa_grad_cam < 0.8] = 0
            caa_grad_cam[caa_grad_cam >= 0.8] = 1
            caa_grad_cam[grad_cam < 0.2] = 0
            caa_grad_cam = preprossess(img_data, caa_grad_cam)
            caa_grad_cam = find_max_eras(caa_grad_cam)
            dice = dice_coeff(caa_grad_cam, label_data)
            dice_caa += dice
            boxcaa, cntcaa = scoremap2bbox(scoremap=caa_grad_cam, threshold=0.4, multi_contour_eval=True)
            cam_box_list.append(show_cam_on_image(tmpimage.copy() , caa_grad_cam, box=boxcaa[0], returncam=True))
            cam_box_dice.append(dice_coeff(caa_grad_cam, label_data))
            cam_iou_mask.append(caa_grad_cam)
            label_res_list.append(show_cam_on_image(tmpimage.copy() , label_data, returncam=True))
            label_dice.append(1)
            SAM_mask_caa = np.load(os.path.join("F:/brats/box_stack_sam1",img_name.split('.')[0]+'.npz'))['caa_sam1_pred']
            SAM_mask_caa[SAM_mask_caa > 0] = 1
            dice_sam=dice_coeff(SAM_mask_caa, label_data)
            dice_sam_caa += dice_sam
            SAM_list.append(show_cam_on_image(tmpimage.copy(), SAM_mask_caa, returncam=True))
            sam2_dice.append(dice_sam)
            sam_iou_mask.append(SAM_mask_caa)
            label_iou_mask.append(label_data)
            ori_iou.append(0)
            if dice_coeff(caa_grad_cam, label_data) >= 0.1:
                dice_out+=dice_sam
                pic_out_len+=1
            else :
                dice_in+=dice_sam
                pic_in_len+=1

        sam_iou=compute_average_iou_for_stack(np.array(sam_iou_mask))
        cam_iou=compute_average_iou_for_stack(np.array(cam_iou_mask))
        label_iou=compute_average_iou_for_stack(np.array(label_iou_mask))
        video_output_path = os.path.join(opt.video_out_path, video_dir.split('\\')[-1])
        video_output_path = os.path.join(video_output_path,"process")
        if not os.path.exists(video_output_path):
            os.makedirs(video_output_path)
        # 保存原始cam为视频。
        save_images_as_video(cam_list, os.path.join(video_output_path, "原始cam.mp4"))
        # 保存处理后cam+box视频
        save_images_as_video(cam_box_list, os.path.join(video_output_path, "cam+box.mp4"))
        # 保存label视频
        save_images_as_video(label_res_list, os.path.join(video_output_path, "label.mp4"))
        #保存MedSAM为视频
        save_images_as_video(SAM_list, os.path.join(video_output_path, "SAM.mp4"))
        dice_list = np.array([cam_dice, cam_box_dice, label_dice, sam2_dice])
        iou_list  = np.array([ori_iou,cam_iou,label_iou,sam_iou])
        merge_videos([
            os.path.join(video_output_path, "原始cam.mp4"),
            os.path.join(video_output_path, "cam+box.mp4"),
            os.path.join(video_output_path, "label.mp4"),
            os.path.join(video_output_path, "SAM.mp4")
        ], output_path=os.path.join(video_output_path, "combine.mp4"), titles=["原始cam", "cam+box", "label","SAM1"],dice=dice_list)
        logging.info(f"{video_name}:dice_cam:{round(np.array(cam_box_dice).sum() / cam_box_dice.__len__(), 4)}"
                     f"dice_sam:{round(np.array(sam2_dice).sum() / sam2_dice.__len__(), 4)},dice_cam_sum: {round(dice_caa / cnt, 4)} dice_sam_sum: {round(dice_sam_caa / cnt, 4)}"
                     f"去掉0.1的dice：{round(dice_out/pic_out_len,4)},只看0.1以下的dice：{round(dice_in/pic_in_len,4)},数据集{pic_out_len}/{cnt}")


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
    parser.add_argument('--cam_dir', default="F:/brats/val/cam")
    parser.add_argument('--video_dir', default="F:/brats/valyes")
    parser.add_argument('--not-save', default=False, action='store_true',
                          help='If yes, only output log to terminal.')
    parser.add_argument('--work-dir', default='./work_dir',
                        help='the work folder for storing results')
    parser.add_argument('--out_path', default="F:/brats/val/cam")
    parser.add_argument('--video_out_path', default="F:/brats/video_medsam")
    parser.add_argument('--sam1-dir',default="F:/brats/box_stack_sam1")
    opt = parser.parse_args()
    logger = loadLogger(opt)
    main(opt,logger)

