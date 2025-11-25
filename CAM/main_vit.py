import os
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from CAM.utils import GradCAM, show_cam_on_image, center_crop_img
from CAM.vit_model import vit_base_patch16_224
from CAM.vit_model import vit_base_patch16_224_in21k as create_model
import argparse
import math
import train_utils.distributed_utils as utils
from Utils import bounding_box
from torch.nn import Module


class ReshapeTransform:
    def __init__(self, model):
        input_size = model.patch_embed.img_size
        patch_size = model.patch_embed.patch_size
        self.h = input_size[0] // patch_size[0]
        self.w = input_size[1] // patch_size[1]

    def __call__(self, x):
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        result = x[:, 1:, :].reshape(x.size(0),
                                     self.h,
                                     self.w,
                                     x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result
def dice_coeff(pred,target):
    smooth = 1e-5
    m1=pred.flatten()
    m2=target.flatten()
    intersection=(m1*m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def main(args):
    target_category = 1  # pug, pug-dog
    dice_computer("F:/brats/val/yes","F:/brats/val/label")



def show_mask(root:str,cam,target_category,threshold):
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    img_path = root
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)

    img = np.load(img_path)["arr_0"]
    img = np.array(img)
    temp = []
    for i in range(3):
        temp.append(img)
    img = np.array(temp)
    img = img.transpose(1, 2, 0)
    img = center_crop_img(img, 224)
    img_tensor = data_transform(img).float()
    input_tensor = torch.unsqueeze(img_tensor, dim=0)
    grayscale_cam,attn_weight_list,res = cam(input_tensor=input_tensor, target_category=target_category)
    grayscale_cam = grayscale_cam[0]
    grayscale_cam = grayscale_cam[0]

    cam = grayscale_cam  # [1,768,14,14]
    return cam,attn_weight_list,res

# 阈值为大于最大值的20%
def dice_computer(img_path:str,label_path:str):
    model = create_model(num_classes=2, has_logits=False)
    weights_path = "F:/Brats-class/vision_transformer/weights/model-5.pth"
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    target_layers = [model.blocks[-1].norm1]
    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  use_cuda=False,
                  reshape_transform=ReshapeTransform(model))
    target_category = 1
    img_sum=len(os.listdir(img_path))
    dice=0

    for i,img_name in enumerate(os.listdir(img_path),1):
        img_data=np.load(os.path.join(img_path,img_name))["arr_0"]
        label_data=np.load(os.path.join(label_path,img_name))["arr_0"]
        label_data[label_data>0]=1
        mask=show_mask(os.path.join(img_path,img_name),cam,target_category,0.7)
        box=bounding_box(mask)
        dice+=dice_coeff(mask,label_data)
        if i%30==0:
            print("以第",{i},"张图片为止时 dice:",(dice/i))
    print(dice/img_sum)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--num_classes', type=int, default=2)
    opt = parser.parse_args()
    main(opt)















