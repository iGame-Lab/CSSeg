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
    weights_path = opt.weights_path

    CAMmodel.load_state_dict(torch.load(weights_path, map_location=device))
    target_layers = [CAMmodel.blocks[-1].norm1]
    cam = GradCAM(model=CAMmodel,
                  target_layers=target_layers,
                  use_cuda=True,
                  reshape_transform=CAM.main_vit.ReshapeTransform(CAMmodel))
    target_category = 1
    for i,img_name in enumerate(os.listdir(opt.img_path),1):
        img_data = np.load(os.path.join(opt.img_path, img_name))["arr_0"]
        label_data = np.load(os.path.join(opt.label_path, img_name))["arr_0"]

        label_data[label_data > 0] = 1
        grayscale_cam,attn_weight_list,res= show_mask(os.path.join(opt.img_path, img_name), cam, target_category, 0.5)
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
        grad_cam=torch.tensor(cv2.resize(np.array(grayscale_cam),(256,256)))
        grayscale_cam = torch.tensor(cv2.resize(np.array(grayscale_cam),(14,14)))
        cam_to_refine = torch.FloatTensor(grayscale_cam)
        cam_to_refine = cam_to_refine.view(-1, 1)
        cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(224 // 16, 224 // 16)
        cam_refined = cam_refined.cpu().numpy().astype(np.float32)
        cam_refined_highres = scale_cam_image([cam_refined], (256, 256))[0]

        np.savez(os.path.join(opt.out_path,img_name),
            original_cam=grad_cam.numpy(),
            caa_cam=cam_refined_highres
        )
def dice_computer_cam(opt,logger,max_dice):
    data_cache = {}
    for img_name in os.listdir(opt.img_path):
        img = np.load(os.path.join(opt.img_path, img_name))["arr_0"]
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
        label = np.load(os.path.join(opt.label_path, img_name))["arr_0"]
        label = cv2.resize((label > 0).astype(np.uint8), (256, 256), cv2.INTER_NEAREST)
        cam_data = np.load(os.path.join(opt.out_path, img_name))
        data_cache[img_name] = (img, label, cam_data['original_cam'], cam_data['caa_cam'])
    for threshold in np.arange(0.85,0.9,0.05).round(2):
        dice_grad_cam=0
        dice_caa_grad_cam=0
        pre_dice_caa_grad_cam=0
        pre_dice_grad_cam=0
        for i,img_name in enumerate(os.listdir(opt.img_path),1):
            dice_ori ,dice_caa = process_single_image(data_cache[img_name],threshold)
            dice_grad_cam+=dice_ori
            dice_caa_grad_cam +=dice_caa
        logger.info("以{}为阈值时，dice_cam:{},dice_caa_cam{},pre_dice_grad_cam:{},pre_dice_caa_grad_cam:{}".format(threshold,dice_grad_cam/i,dice_caa_grad_cam/i,pre_dice_grad_cam/i,pre_dice_caa_grad_cam/i))
        max_dice = max(max_dice,dice_grad_cam/i,dice_caa_grad_cam/i,pre_dice_grad_cam/i,pre_dice_caa_grad_cam/i)
    logger.info("                                         ")
    return  max_dice

def process_single_image(data, threshold):
    """单图像处理函数 (根据网页3[3](@ref)的Dice计算优化)"""
    img_data, label, original_cam, caa_cam = data
    binary_original = np.zeros(original_cam.shape)
    binary_caa = np.zeros(caa_cam.shape)
    # 阈值化处理
    binary_original[original_cam >= threshold] = 1
    binary_caa[caa_cam >= threshold] = 1
    binary_original = preprossess(img_data,binary_original)
    binary_caa = preprossess(img_data, binary_caa)

    # 后处理优化 (参考网页8[8](@ref)的向量化操作)
    binary_original = find_max_eras(binary_original)
    binary_caa = find_max_eras(binary_caa)

    dice_original = dice_coeff(label, binary_original)
    dice_caa = dice_coeff(label, binary_caa)
    return dice_original, dice_caa

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
    root_path = "/root/autodl-tmp/Task04_Hippocampus/Train"
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--img_path',default=os.path.join(root_path,"yes"))
    parser.add_argument('--label_path',default=os.path.join(root_path,"label"))
    parser.add_argument('--not-save', default=False, action='store_true',
                          help='If yes, only output log to terminal.')
    parser.add_argument('--out_path',default=os.path.join(root_path,"newcam"))
    parser.add_argument('--work-dir', default='./work_dir',
                        help='the work folder for storing results')
    parser.add_argument('--weights_path',default="/root/brats-class/vision_transformer/Task04_Hippocampus/model-5.pth")
    opt = parser.parse_args()
    logger = loadLogger(opt)
    max_dice = 0

    main(opt,logger)
    # max_dice = dice_computer_cam(opt,logger,max_dice)



