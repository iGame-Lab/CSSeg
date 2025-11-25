import os
import numpy as np
from PIL import Image
import cv2
from Utils import concatenate_saved_images

fold_path = os.listdir('F:/brats/chutu/add_iou')
for fold in fold_path:

    for path in os.listdir(os.path.join('F:/brats/chutu/add_iou',fold)):
        output_path = 'F:/brats/chutu/cot'
        img_path1 = os.path.join('F:/brats/chutu/add_iou',fold,path)
        img_path2 = os.path.join('F:/brats/chutu/add_iou_add_ts',fold,path)
        img_path3 = os.path.join('F:/brats/chutu/add_twosam',fold,path)
        img_path4 = os.path.join('F:/brats/chutu/valyes',fold,path)
        output_path = os.path.join(output_path,img_path1.split('\\')[1])
        img1 = np.array(Image.open(img_path1))
        img2 = np.array(Image.open(img_path2))
        img3 = np.array(Image.open(img_path3))
        img4 = np.array(Image.open(img_path4))
        img1_part1 = img1[:, :216, :]  # 第一张 216x288
        img1_part2 = img1[:, 216:432, :]  # 第二张 216x288
        img1_part3 = img1[:, 432:648, :]  # 第三张 216x288
        img1_part4 = img1[:, 648:, :]  # 第三张 216x288
        # 从 img2 提取第三张部分
        img2_part3 = img2[:, 432:648, :]  # 第三张 216x288

        # 从 img3 提取第三张部分
        img3_part3 = img3[:, 432:648, :]  # 第三张 216x288
        concatenated_image = np.concatenate([img1_part1,img4, img1_part2, img1_part3, img2_part3, img3_part3,img1_part4], axis=1)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        Image.fromarray(concatenated_image).save(os.path.join(output_path,img_path1.split('\\')[-1]))
