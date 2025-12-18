import re


# 计算 dice_sam 平均值的函数，尝试不同编码格式
def calculate_average_dice_sam(file_path):
    # 尝试使用不同编码读取文件
    encodings = ['utf-8', 'ISO-8859-1', 'GBK']

    for encoding in encodings:
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding=encoding) as file:
                data = file.read()
            print(f"文件成功使用 {encoding} 编码读取")
            break
        except UnicodeDecodeError:
            print(f"使用 {encoding} 编码读取失败，尝试下一个编码...")
    else:
        print("所有编码读取失败。")
        return None

    # 提取 dice_sam 值
    dice_sam_values = [float(x) for x in re.findall(r'dice_sam:([\d\.]+)', data)]

    # 计算 dice_sam 的平均值
    if dice_sam_values:
        average_dice_sam = sum(dice_sam_values) / len(dice_sam_values)
        return average_dice_sam
    else:
        return None


# 使用示例，替换为你的文件路径
file_path = '/root/data1/cam-MEDSAM/work_dir/有二次SAM，0.9/log.txt'  # 将 'your_file.txt' 替换为你的实际文件路径
average_dice_sam = calculate_average_dice_sam(file_path)

if average_dice_sam is not None:
    print(f"文件中 dice_sam 的平均值是: {average_dice_sam:.4f}")
else:
    print("文件中没有找到 dice_sam 值。")
