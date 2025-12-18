import re
import pandas as pd


def log_to_excel(input_txt, output_xlsx=None):
    """
    将视频处理日志转换为Excel表格
    参数:
        input_txt: 输入日志文件路径
        output_xlsx: 输出Excel路径(默认同路径同名文件)
    """
    # 定义正则表达式模式（网页3、网页7的最佳实践）
    pattern = r'(\w+):similarity_label :([\d.]+),similarity_sam :([\d.]+),' \
              r'similarity_cam :([\d.]+),sam_gt_dice :([\d.]+),cam_gt_dice :([\d.]+)'

    # 自动生成输出路径（网页5的实现逻辑）
    if not output_xlsx:
        output_xlsx = input_txt.rsplit('.', 1)[0] + '_metrics.xlsx'

    data = []
    try:
        # 读取日志文件（网页1推荐的with语句和编码处理）
        with open(input_txt, 'r', encoding='utf-8') as f:
            for line in f:
                # 使用正则表达式匹配（网页3的解析方法）
                match = re.search(pattern, line.strip())
                if match:
                    # 提取并转换数据类型（网页7的数据清洗实践）
                    row = [match.group(1)] + [round(float(x), 4) for x in match.groups()[1:]]
                    data.append(row)

        # 创建DataFrame（网页2推荐的Pandas方法）
        df = pd.DataFrame(data, columns=[
            'ID',
            'similarity_label',
            'similarity_cam',
            'similarity_sam',
            'sam_gt_dice',
            'cam_gt_dice'
        ])

        # 导出Excel（网页4的保存方法）
        df.to_excel(output_xlsx, index=False, engine='openpyxl')
        print(f"成功转换 {len(data)} 条记录到 {output_xlsx}")

    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_txt}")
    except Exception as e:
        print(f"转换失败：{str(e)}")


# 使用示例
if __name__ == "__main__":
    log_to_excel("F:/new_cam-MEDSAM/work_dir/Miou_dice/Miou_dice.txt", "F:/new_cam-MEDSAM/work_dir/Miou_dice/output_metrics.xlsx")