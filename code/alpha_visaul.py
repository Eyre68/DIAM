import os
import pandas as pd
import numpy as np
import matplotlib
from matplotlib.ticker import FormatStrFormatter
# 强制使用 Agg 后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==================== 配置区 ====================
param_name = 'alpha'  # 参数名
base_path = './2'  # 数据集所在的根目录
datasets = ['B', 'C', 'F']  # 要对比的三个数据集

# 绘图配置
xlabel = r'$\alpha$'  # X轴标签 (支持 LaTeX 格式，显示为希腊字母 alpha)
# ylabel        = 'AUC Score'               # 已移除
figsize = (10, 7)  # 大画布
dpi = 150
save_folder = './combined_results'  # 图片保存位置
save_filename = f'combined_{param_name}_auc_large.png'

# 字体大小配置 (保持大字体)
FONT_SIZE_LABEL = 30
FONT_SIZE_TICK = 30
FONT_SIZE_LEGEND = 25

# 线条配置
LINE_WIDTH = 1.5  # 细线
MARKER_SIZE = 10  # 大点

# 样式配置
styles = {
    'B': {'color': '#4E79A7', 'marker': 'o', 'label': 'Bdataset'},  # 蓝色
    'C': {'color': '#F28E2B', 'marker': 's', 'label': 'Cdataset'},  # 橙色
    'F': {'color': '#B5CE4E', 'marker': '^', 'label': 'Fdataset'}  # 红色
}

# ================================================

# ---------- 读取数据 ----------
# 结构: all_data = { 'B': [(alpha1, auc1), (alpha2, auc2)...], ... }
all_data = {}

print(f"准备读取数据集: {datasets}，参数: {param_name}")

for ds in datasets:
    folder_path = os.path.join(base_path, ds, param_name)
    xy_points = []  # 存储 (x, y) 元组

    if not os.path.exists(folder_path):
        print(f"  [缺失] 文件夹不存在: {folder_path}")
        all_data[ds] = []
        continue

    # 遍历文件夹下的所有文件
    for fname in os.listdir(folder_path):
        if not fname.endswith('.xlsx'):
            continue

        # 解析文件名获取 alpha 值
        try:
            alpha_val = float(fname.replace('.xlsx', ''))
        except ValueError:
            continue  # 跳过无法转为数字的文件

        file_path = os.path.join(folder_path, fname)

        try:
            df = pd.read_excel(file_path, index_col=0)

            if 'Mean' not in df.index:
                continue

            row = df.loc['Mean']
            if 'AUC' not in row:
                continue

            auc_val = float(row['AUC'])
            xy_points.append((alpha_val, auc_val))

        except Exception as e:
            print(f"  [读取失败] {ds}/{fname}: {e}")

    # *** 关键步骤：按 alpha 值从小到大排序 ***
    # 如果不排序，连线会乱七八糟
    xy_points.sort(key=lambda x: x[0])

    all_data[ds] = xy_points
    print(f"数据集 {ds} 读取完毕，共 {len(xy_points)} 个点。")

# ---------- 计算 Y 轴范围 ----------
all_auc_values = []
for points in all_data.values():
    for _, y in points:
        all_auc_values.append(y)

if not all_auc_values:
    print("警告：没有读取到任何数据，使用默认范围。")
    y_min, y_max = 0.5, 1.0
else:
    y_min = max(min(all_auc_values) - 0.01, 0.0)
    y_max = min(max(all_auc_values) + 0.01, 1.0)

# ---------- 绘图 ----------
plt.figure(figsize=figsize, dpi=dpi)

for ds in datasets:
    points = all_data[ds]
    if not points:
        continue

    # 解包 x 和 y
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]

    style = styles[ds]

    plt.plot(x_vals, y_vals,
             marker=style['marker'],
             color=style['color'],
             linewidth=LINE_WIDTH,
             markersize=MARKER_SIZE,
             label=style['label'])

# 装饰
# plt.title(...) # 已移除
plt.xlabel(xlabel, fontsize=FONT_SIZE_LABEL)
# plt.ylabel(...) # 已移除

plt.xticks(fontsize=FONT_SIZE_TICK)
plt.yticks(fontsize=FONT_SIZE_TICK)
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.ylim(y_min, y_max)
# 如果需要限制 X 轴范围，可以在这里加 plt.xlim(...)

plt.legend(fontsize=FONT_SIZE_LEGEND)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# ---------- 保存 ----------
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder, save_filename)
plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
print(f"\n图已保存: {save_path}")
plt.close()