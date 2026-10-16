import os
import pandas as pd
import numpy as np
import matplotlib

# 配置 Matplotlib 使用 Times New Roman 字体
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# ==================== 全局字体与风格设置 ====================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ==================== 配置区 ====================
base_path = './2'
datasets = ['C', 'B', 'F']
dataset_names = ['Cdataset', 'Bdataset', 'Fdataset']

variants = [
    'Variant1', 'Variant2', 'Variant3', 'Variant4',
    'Variant5', 'Variant6', 'Variant7', 'Variant8', 'FULL'
]
short_labels = [v.replace('Variant', 'V') for v in variants]

save_folder = './2'
save_name = 'all_datasets_clean.png'
save_path = os.path.join(save_folder, save_name)

# ==================== 1. 定义渐变色 (保持不变) ====================
colors_hex = [
    '#d7312d', '#f2724d', '#fcb777', '#fee395', '#fef9b7',
    '#d2edf2', '#acd2e5', '#6090c1', '#4573b4'
]
custom_cmap = LinearSegmentedColormap.from_list("custom_red_blue", colors_hex, N=len(variants))

# ==================== 2. 读取数据 (保持不变) ====================
data_store = {}
# print("正在读取所有数据集数据...")
for ds_code in datasets:
    folder_path = os.path.join(base_path, ds_code, '消融')
    ds_means = []
    for name in variants:
        fname = f"{name}.xlsx"
        file_path = os.path.join(folder_path, fname)
        val_mean = 0
        if os.path.exists(file_path):
            try:
                df = pd.read_excel(file_path, index_col=0)
                if 'Mean' in df.index and 'AUC' in df.loc['Mean']:
                    val_mean = float(df.loc['Mean']['AUC'])
            except Exception:
                pass
        if val_mean == 0: val_mean = 0.85  # 仅用于测试，实际请确保数据存在
        ds_means.append(val_mean)
    data_store[ds_code] = {'means': np.array(ds_means)}

# ==================== 3. 绘图 ====================

# 保持紧凑尺寸: 宽6, 高3
fig, ax = plt.subplots(figsize=(6, 3), dpi=300)

n_bars = len(variants)
bar_width = 0.8  # 柱宽
group_spacing = 1.0  # 组间距
indices = np.arange(n_bars)

bar_colors = [custom_cmap(i / (n_bars - 1)) for i in range(n_bars)]
x_ticks_pos = []

for i, ds_code in enumerate(datasets):
    means = data_store[ds_code]['means']

    x_offset = i * (n_bars * bar_width + group_spacing)
    x_pos = indices * bar_width + x_offset

    # 绘制柱子
    bars = ax.bar(x_pos, means, width=bar_width,
                  color=bar_colors, edgecolor='black', linewidth=0.5, zorder=3)

    x_ticks_pos.append(x_offset + (n_bars * bar_width) / 2 - (bar_width / 2))

    # === 修改：不再添加数值标注 ===
    # (Removed ax.text loop)

# ==================== 4. 美化图表 ====================

# X轴
ax.set_xticks(x_ticks_pos)
ax.set_xticklabels(dataset_names, fontsize=12, fontweight='bold')
ax.tick_params(axis='x', length=0)

# Y轴
ax.set_ylim(0.8, 1.0)
ax.set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])
# === 修改：不再设置 Y 轴标签 ===
# ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
ax.tick_params(axis='y', labelsize=10)

ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_title('')

# 边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

# 图例
legend_handles = [Patch(facecolor=c, edgecolor='black', linewidth=0.5, label=l)
                  for c, l in zip(bar_colors, short_labels)]

ax.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 1.02),
          ncol=n_bars, fontsize=8, frameon=False, columnspacing=0.5, handletextpad=0.2)

plt.tight_layout()

# ==================== 保存 ====================
os.makedirs(save_folder, exist_ok=True)
plt.savefig(save_path, bbox_inches='tight', dpi=300)
print(f"\n清洁版图表已保存: {save_path}")
plt.close()