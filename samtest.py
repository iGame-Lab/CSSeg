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
from MedSAM.segment_anything import sam_model_registry
from MedSAM.demo import BboxPromptDemo
from CAM.utils import GradCAM, show_cam_on_image
def show_mask_image(mask, ax, random_color=False, alpha=0.95):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
def main(opt,logger):
    if not os.path.exists(opt.out_path):
        os.makedirs(opt.out_path)
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

        np.savez(os.path.join(opt.out_path,img_name),
            original_cam=grad_cam.numpy(),
            caa_cam=cam_refined_highres
        )
def dice_computer_cam(opt,logger):
    device = "cuda:0"
    MedSAM_CKPT_PATH = "C:/Users/Administrator/Desktop/MedSAM-main/work_dir/MedSAM/medsam_vit_b.pth"
    medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
    bbox_prompt_demo = BboxPromptDemo(medsam_model)
    dice_sum=0
    for i,img_name in enumerate(os.listdir(opt.img_path),1):
        img_data = np.load(os.path.join(opt.img_path, img_name))["arr_0"]
        label_data = np.load(os.path.join(opt.label_path, img_name))["arr_0"]
        label_data[label_data > 0] = 1
        temp = []
        for _ in range(3):
            temp.append(img_data)
        tmpimage = np.array(temp)
        tmpimage = tmpimage.transpose(1, 2, 0)
        # show_cam_on_image(tmpimage, label_data, use_rgb=True, dice=dice_coeff(label_data, label_data),
        #                   name="sam")

        box, cnt = scoremap2bbox(scoremap=label_data, threshold=0.4, multi_contour_eval=True)
        SAM_mask = bbox_prompt_demo.show(image_path=os.path.join(opt.img_path, img_name), Box=box[0])
        for j_ in range(1,cnt):
            tmp_SAM_mask = bbox_prompt_demo.show(image_path=os.path.join(opt.img_path, img_name), Box=box[j_])
            SAM_mask=SAM_mask+tmp_SAM_mask
        # show_cam_on_image(tmpimage, SAM_mask, use_rgb=True, dice=dice_coeff(SAM_mask, label_data),
        #                       name="sam")
        SAM_mask[SAM_mask>0]=1
        dice_sum+=dice_coeff(SAM_mask,label_data)
        if dice_coeff(SAM_mask,label_data)<0.6:
            logger.info("第{}张图片,dice_sam{}".format(i,dice_coeff(SAM_mask,label_data)))
    logger.info("\ndice_sum:{}".format(dice_sum/len(os.listdir(opt.img_path))))
    print(dice_sum)
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
    parser.add_argument('--out_path',default="F:/brats/val/cam")
    parser.add_argument('--work-dir', default='./work_dir',
                        help='the work folder for storing results')
    opt = parser.parse_args()
    logger = loadLogger(opt)
    # main(opt,logger)
    dice_computer_cam(opt,logger)

