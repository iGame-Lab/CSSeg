import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_excel("F:/new_cam-MEDSAM/work_dir/Miou_dice/ddd_metrics.xlsx")
df.dropna(inplace=True)
x = df['ID'].values
y_columns = ['similarity_label', 'similarity_sam', 'similarity_AMS','similarity_cam']

# 创建组合画布（网页6、网页8的子图布局方法）
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(1, 2, width_ratios=[5, 1], wspace=0.03)
ax_main = fig.add_subplot(gs[0])
ax_kde = fig.add_subplot(gs[1])

# ========== 主图区域 ==========
colors = ['c', 'violet','blue','orangered']
markers = ['o', 's', 'D','s']
labels = ['GT', 'CCSeg', 'AMS','initial CAM']
x_smooth = np.linspace(min(x), max(x), 300)

# 绘制散点+拟合曲线（网页1的拟合方法）
for idx, col in enumerate(y_columns):
    ax_main.scatter(x, df[col], color=colors[idx], marker=markers[idx],
                   s=45, alpha=0.5, edgecolors='w', label=labels[idx])
    coeffs = np.polyfit(x, df[col], deg=3)
    poly_func = np.poly1d(coeffs)
    ax_main.plot(x_smooth, poly_func(x_smooth), color=colors[idx],
                linestyle='--', linewidth=3, alpha=0.9)

# 主图装饰（网页8的格式优化）
ax_main.set_xlabel('3D Sample', fontsize=25, fontweight='bold', labelpad=15)
ax_main.set_ylabel('Similarity', fontsize=25, fontweight='bold', labelpad=15)
ax_main.set_title('The similarity of adjacent frames', fontsize=25, pad=20)
ax_main.grid(True, alpha=0.3, linestyle=':')
ax_main.set_ylim(0.55, 0.95)
ax_main.tick_params(axis='both', labelsize=18)

for idx, col in enumerate(y_columns):
    sns.kdeplot(y=df[col],  # 关键修改点：删除vertical=True参数
               ax=ax_kde,
               fill=True,
               alpha=0.2,
               color=colors[idx],
               linewidth=2.5,
                zorder=3 - idx,
                )

# 坐标轴设置（网页4的轴控制方法）

ax_kde.yaxis.set_visible(False)  # 隐藏重复的y轴
ax_kde.xaxis.set_visible(False)  # 隐藏重复的y轴
ax_kde.spines['top'].set_visible(False)  # 网页2、网页3
ax_kde.spines['bottom'].set_visible(False)  # 网页2、网页3
# ax_kde.tick_params(axis='x', labelsize=18, rotation=0)
ax_kde.spines['left'].set_visible(False)
ax_kde.spines['right'].set_visible(False)
ax_kde.set_xlim(0, 11)  # 根据实际密度范围调整
ax_kde.set_ylim(1, 0.5)
# ========== 全局优化 ==========
# 图例设置（网页7的视觉优化）
handles, labels = ax_main.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper right',
#           bbox_to_anchor=(0.92, 0.88),
#           frameon=True, shadow=True,
#           fontsize=14, borderpad=1)
ax_main.legend(
    handles=handles,
    labels=labels,
    loc='upper right',
    frameon=True,
    shadow=True,
    fontsize=13,  # 与主图标题字号比例协调
    borderpad=0.4,
    framealpha=1,
    # edgecolor='#2E86C1',  # 与主图配色呼应[3](@ref)
    # facecolor='#F8F9F9'   # 浅灰色背景
)


# 保存输出（网页6的高清设置）
plt.savefig('F:/new_cam-MEDSAM/work_dir/Miou_dice/组合图表+AMS.png',
           dpi=1200, bbox_inches='tight', facecolor='white')
plt.show()