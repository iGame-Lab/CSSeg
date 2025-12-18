# 读取文件内容并进行处理
input_file = 'F:/new_cam-MEDSAM/work_dir/sam3d.txt'  # 输入文件的路径
output_file = 'F:/new_cam-MEDSAM/work_dir/sam3d_sort.txt'  # 输出文件的路径

# 从 txt 文件中读取数据
with open(input_file, 'r') as file:
    data = file.readlines()

# 将每行的编号提取并排序
sorted_lines = sorted(data, key=lambda x: int(x.split('_')[1]))

# 将排序后的结果写入新的 txt 文件
with open(output_file, 'w') as file:
    file.writelines(sorted_lines)

print(f"排序后的结果已保存到 {output_file}")
