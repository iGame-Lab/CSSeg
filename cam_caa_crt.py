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
# import CRF
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
    # MedSAM_CKPT_PATH = "C:/Users/Administrator/Desktop/MedSAM-main/work_dir/MedSAM/medsam_vit_b.pth"
    # medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
    # medsam_model = medsam_model.to(device)
    # medsam_model.eval()

    dice_CAM_CRF=0
    dice_caa=0
    dice_sam_crf=0
    dice_sam_caa=0
    pic_len=0
    # bbox_prompt_demo = BboxPromptDemo(medsam_model)
    for i,img_name in enumerate(os.listdir(opt.img_path),1):
        if img_name!='BraTS2021_00528_57.npz':
            continue
        img_data = np.load(os.path.join(opt.img_path, img_name))["arr_0"]
        label_data = np.load(os.path.join(opt.label_path, img_name))["arr_0"]
        label_data[label_data > 0] = 1
        grad_cam = np.load(os.path.join(opt.out_path, img_name))['original_cam']
        caa_grad_cam = np.load(os.path.join(opt.out_path, img_name))['caa_cam']
        original_grad_cam = np.array(caa_grad_cam).copy()
        original_grad_cam[original_grad_cam<0.8]=0
        original_grad_cam[grad_cam<0.2]=0
        caa_grad_cam[caa_grad_cam < 0.8] = 0
        caa_grad_cam[caa_grad_cam >= 0.8] = 1
        caa_grad_cam[grad_cam<0.2]=0
        caa_grad_cam= preprossess(img_data,caa_grad_cam)
        caa_grad_cam = find_max_eras(caa_grad_cam)
        if dice_coeff(caa_grad_cam,label_data)>=0.1:
            continue
        else :
            pic_len+=1
        dice_caa+=dice_coeff(caa_grad_cam,label_data)
        temp = []
        for _ in range(3):
            temp.append(img_data)
        tmpimage = np.array(temp)
        tmpimage = tmpimage.transpose(1, 2, 0)
        # CRF_label = CRF.crf(n_jobs, tmpimage * 255, label_data, original_grad_cam ,
        #                     mean_bgr=(img_data.mean(), img_data.mean(), img_data.mean()))
        # if dice_coeff(CRF_label,label_data)>dice_coeff(caa_grad_cam,label_data):
        #     dice_CAM_CRF += dice_coeff(CRF_label, label_data)
        # else :
        #     dice_CAM_CRF += dice_coeff(caa_grad_cam, label_data)

        # boxcrf, cntcrf = scoremap2bbox(scoremap=CRF_label, threshold=0.4, multi_contour_eval=True)
        # boxcaa, cntcaa = scoremap2bbox(scoremap=caa_grad_cam, threshold=0.4, multi_contour_eval=True)

        # SAM_mask_crf = bbox_prompt_demo.show(image_path=os.path.join(opt.img_path, img_name), Box=boxcrf[0])
        # for j_ in range(1, cntcrf):
        #     tmp_SAM_mask = bbox_prompt_demo.show(image_path=os.path.join(opt.img_path, img_name), Box=boxcrf[j_])
        #     SAM_mask_crf = SAM_mask_crf + tmp_SAM_mask
        # show_cam_on_image(tmpimage, SAM_mask, use_rgb=True, dice=dice_coeff(SAM_mask, label_data),
        #                       name="sam")
        # SAM_mask_crf[SAM_mask_crf > 0] = 1

        # SAM_mask_caa = bbox_prompt_demo.show(image_path=os.path.join(opt.img_path, img_name), Box=boxcaa[0])
        # for j_ in range(1, cntcaa):
        #     tmp_SAM_mask = bbox_prompt_demo.show(image_path=os.path.join(opt.img_path, img_name), Box=boxcaa[j_])
        #     SAM_mask_caa = SAM_mask_caa + tmp_SAM_mask
        # show_cam_on_image(tmpimage, SAM_mask, use_rgb=True, dice=dice_coeff(SAM_mask, label_data),
        #                       name="sam")
        # SAM_mask_caa[SAM_mask_caa > 0] = 1
        # if dice_coeff(SAM_mask_crf, label_data) > dice_coeff(SAM_mask_caa, label_data):
        # dice_sam_crf += dice_coeff(SAM_mask_crf, label_data)
        # # else :
        # #     dice_sam_crf += dice_coeff(SAM_mask_caa, label_data)
        # # dice_sam_caa += dice_coeff(SAM_mask_caa, label_data)
        # np.savez(os.path.join("F:/brats/SAM1",img_name),
        #          caa_CRF_sam1_pred=CRF_label,
        #          # caa_sam1_pred=SAM_mask_caa
        #          )


        # if dice_coeff(SAM_mask_caa, label_data)<0.3:
            # show_cam_on_image(tmpimage, CRF_label, use_rgb=True, dice=dice_coeff(CRF_label, label_data), name="CRF")
            # show_cam_on_image(tmpimage, caa_grad_cam, use_rgb=True, dice=dice_coeff(caa_grad_cam, label_data),
            #                   name="caa_grad_cam")
            # show_cam_on_image(tmpimage, original_grad_cam, use_rgb=True,
            #                   name="original_grad_cam")
            # show_cam_on_image(tmpimage, grad_cam, use_rgb=True,
            #                   name="grad_cam")
            # show_cam_on_image(tmpimage, label_data, use_rgb=True, dice=dice_coeff(label_data, label_data), name="label_data")
            # show_cam_on_image(tmpimage, SAM_mask_crf, use_rgb=True,dice=dice_coeff(SAM_mask_crf,label_data),
            #                   name="SAM_mask_crf")
            # show_cam_on_image(tmpimage, SAM_mask_caa, use_rgb=True, dice=dice_coeff(SAM_mask_caa, label_data),
            #                   name="SAM_mask_caa")
    #     logger.info("图片:{},dice_sam_crf{},dice_caa_sam_cam:{}".format(img_name,dice_coeff(SAM_mask_crf, label_data),dice_coeff(SAM_mask_caa, label_data)))
    #
    #     if i%30==0:
    #         logger.info("\ndice_crf_sam_mean:{},dice_caa_sam_mean:{}".format(dice_sam_crf / pic_len,dice_sam_caa/pic_len))
    # logger.info("\ndice_sum:{}".format(dice_sam_crf / pic_len))
    # logger.info("\ndice_caa_sum:{}".format(dice_sam_caa / pic_len))
    # logger.info("\n数据集大小:{}".format(pic_len))



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
    parser.add_argument('--out_path', default="F:/brats/val/newcam")
    opt = parser.parse_args()
    logger = loadLogger(opt)
    main(opt,logger)

































