import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# 读取数据并清洗
df = pd.read_excel("F:/new_cam-MEDSAM/work_dir/Miou_dice/output_metrics.xlsx")
df.dropna(inplace=True)

# 设置坐标轴数据
x = df['cam_gt_dice'].values
y = df['sam_gt_dice'].values

# 创建基础散点图
plt.figure(figsize=(10, 8))
scatter = plt.scatter(x, y,
                     c='violet',
                     s=60,
                     alpha=0.6,
                     edgecolor='w',
                     linewidth=1.2)
# 在绘制散点图之后，坐标轴设置之前添加：
plt.plot([0, 1], [0, 1],
        color='steelblue',  # 红色系增强对比[6,7](@ref)
        # linestyle='--',   # 虚线样式[1,3](@ref)
        linewidth=2.5,
        alpha=0.6,
        label='y = x')    # 图例标签[7](@ref)
# 二次多项式拟合（网页3/网页6/网页7）
degree = 5  # 可修改为3进行三次拟合
coeffs = np.polyfit(x, y, degree)
y_fit = np.polyval(coeffs, x)  # 生成拟合值

# 绘制新拟合线（网页1/网页3）
plt.plot(np.sort(x), np.polyval(coeffs, np.sort(x)),
         color='violet',
         linestyle='--',
         label=f'Poly Fit (d={degree})')
# 坐标轴装饰
plt.xlabel('CAM_GT_Dice', fontsize=18, fontweight='bold')
plt.ylabel('SAM_GT_Dice', fontsize=18, fontweight='bold')
plt.title('SAM vs CAM Dice', fontsize=20, pad=15)
plt.grid(True, alpha=0.3, linestyle=':')

# ========== 新增坐标轴范围设置 ==========
plt.xlim(0.15, 0.9)  # 强制x轴范围[1,3](@ref)
plt.ylim(0.15, 0.9)  # 强制y轴范围[1,3](@ref)
# 或使用 plt.axis([0, 1, 0, 1])[4](@ref)

plt.tight_layout()
plt.savefig('SAM_CAM_scatter.png',
           dpi=600,
           bbox_inches='tight',
           facecolor='#F8F9F9')
plt.show()