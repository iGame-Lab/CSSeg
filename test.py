import os
import numpy as np
from PIL import Image

from Utils import concatenate_images_and_save

# def concatenate_images_vertically(image_paths):
#     # 加载所有图片
#     images = [Image.open(path) for path in image_paths]
#
#     # 获取单张图片的宽度和所有图片总高度
#     width = images[0].width
#     total_height = sum(image.height for image in images)
#
#     # 创建拼接后的空白图像
#     concatenated_image = Image.new('RGB', (width, total_height))
#
#     # 逐张图片粘贴到新图片上
#     y_offset = 0
#     for image in images:
#         concatenated_image.paste(image, (0, y_offset))
#         y_offset += image.height
#
#     return concatenated_image
# image_path=[]
# image_paths = os.listdir('F:/brats/Train_Sets/val/val/video_twosam_addcam_10223/10')
# for img in image_paths:
#     image_path.append(os.path.join('F:/brats/Train_Sets/val/val/video_twosam_addcam_10223/10',img))
# result_image = concatenate_images_vertically(image_path)
# result_image.save('F:/brats/Train_Sets/val/val/video_twosam_addcam_10223/10/concatenated_image.jpg')

#2,3,6,11
img = np.array(Image.open("F:/brats/Train_Sets/val/val/valyes/1/1_22.png"))
img = img[30:210,30:210]



# 将数组转换为Pillow图像对象
result_image = Image.fromarray(img.astype('uint8'))

# 保存并显示结果
result_image.save("F:/brats/Train_Sets/val/val/chutu/{4}.png")
# result_image.show()
print(1)
# concatenate_images_and_save(np.array(Image.open()),output_path=os.path.join("F:/brats/Train_Sets/val/val/chutu", f"{1}.png"))
# concatenate_images_and_save(np.array(Image.open("F:/brats/Train_Sets/val/val/valyes/10/10_22.png")),output_path=os.path.join("F:/brats/Train_Sets/val/val/chutu", f"{2}.png"))
# concatenate_images_and_save(np.array(Image.open("F:/brats/Train_Sets/val/val/valyes/1/1_21.png")),output_path=os.path.join("F:/brats/Train_Sets/val/val/chutu", f"{3}.png"))
# concatenate_images_and_save(np.array(Image.open("F:/brats/Train_Sets/val/val/valyes/1/1_22.png")),output_path=os.path.join("F:/brats/Train_Sets/val/val/chutu", f"{4}.png"))
# "F:/brats/Train_Sets/val/val/valyes/10/10_20.png"
"F:/brats/Train_Sets/val/val/valyes/10/10_22.png"
"F:/brats/Train_Sets/val/val/valyes/1/1_21.png"
"F:/brats/Train_Sets/val/val/valyes/1/1_22.png"
