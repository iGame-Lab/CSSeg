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

import os
import shutil




def main(opt):

    # 获得视频列表
    video_path = build_video_list(opt.video_dir)
    cnt = 0

    for video_dir in tqdm(video_path, desc="Processing Videos"):
        video_name = video_dir.split('\\')[-1]
        dir = os.path.join('F:/brats/BraTS2021_Training_Data',video_name)
        tardir = os.path.join('F:/brats/BraTS2021_val_Data',video_name)
        source_path = os.path.abspath(dir)
        target_path = os.path.abspath(tardir)


        shutil.copytree(source_path, target_path)


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
    parser.add_argument('--video_out_path', default="F:/brats/video_medsam_stackiou0")
    parser.add_argument('--sam1-dir', default="F:/brats/box_stack_sam1")
    parser.add_argument('--npz-dir', default="F:/brats/box_stack_sam1_stack/iou0")
    opt = parser.parse_args()
    main(opt)
