import numpy as np

from MedSAM.segment_anything import sam_model_registry
from MedSAM.demo import BboxPromptDemo
import os
from CAM.utils import GradCAM
from CAM.vit_model import vit_base_patch16_224_in21k as create_model
import argparse
import CAM.main_vit
from CAM.main_vit import show_mask,dice_coeff
import logging
import time
from CAM.utils import scoremap2bbox
from Utils import *

def main(opt,logger):
    device = "cuda:0"
    #CAM模型导入


    MedSAM_CKPT_PATH = "C:/Users/Administrator/Desktop/MedSAM-main/work_dir/MedSAM/medsam_vit_b.pth"
    medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()

    bbox_prompt_demo = BboxPromptDemo(medsam_model)
    for i,img_name in enumerate(os.listdir(opt.img_path),1):
        img_data = np.load(os.path.join(opt.img_path, img_name))["arr_0"]
        label_data = np.load(os.path.join(opt.label_path, img_name))["arr_0"]
        label_data[label_data > 0] = 1


        file_cam=np.load(os.path.join(opt.cam_path,img_name))["arr_0"]
        grad_cam=file_cam['original_cam']
        caa_grad_cam=file_cam['caa_cam']







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
    parser.add_argument('--cam_path',default="F:/brats/val/cam")
    parser.add_argument('--work-dir', default='./work_dir',
                        help='the work folder for storing results')
    opt = parser.parse_args()
    logger = loadLogger(opt)
    main(opt,logger)




