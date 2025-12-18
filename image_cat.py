import os
from PIL import Image

def concat_images_vertically(images):
    widths, heights = zip(*(img.size for img in images))

    # 拼接后的宽度为最大宽度，高度为所有高度之和
    total_width = max(widths)
    total_height = sum(heights)

    # 创建一个支持透明度的新图像用于拼接
    new_image = Image.new("RGBA", (total_width, total_height))

    # 将每张图片按顺序粘贴到新图像中
    y_offset = 0
    for img in images:
        # 提取 alpha 通道作为透明度蒙版
        if img.mode == "RGBA":
            alpha = img.split()[-1]  # 提取 alpha 通道
            new_image.paste(img, (0, y_offset), mask=alpha)
        else:
            # 如果图像没有透明度，直接粘贴
            new_image.paste(img, (0, y_offset))
        y_offset += img.height

    return new_image

def process_folder(folder_path, output_folder):
    images = []
    # 遍历每个子文件夹
    count = 0
    cnt = 0
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)

        if os.path.isdir(subfolder_path):

            for filename in sorted(os.listdir(subfolder_path)):
                if filename.endswith('.png'):
                    img_path = os.path.join(subfolder_path, filename)
                    img = Image.open(img_path)
                    images.append(img)
                    # 每5张图片进行一次拼接
                    if len(images) == 5:
                        combined_image = concat_images_vertically(images)
                        # 保存拼接结果为 PNG 格式
                        output_path = os.path.join(output_folder, f"combined_{cnt}.png")
                        cnt+=1
                        combined_image.save(output_path, format="PNG")
                        print(f"拼接完成：{output_path}")
                        # 清空列表并增加计数
                        images.clear()
                        count += 1

    if images:
        # 拼接图像
        combined_image = concat_images_vertically(images)
        # 保存拼接结果为 PNG 格式
        output_path = os.path.join(output_folder, f"combined{cnt}.png")
        combined_image.save(output_path, format="PNG")
        print(f"拼接完成：{output_path}")

# 设置主文件夹路径和输出路径
main_folder = "F:/brats/chutu/valyes"
output_folder = "F:/brats/chutu/cat"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

process_folder(main_folder, output_folder)
