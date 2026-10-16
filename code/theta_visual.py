import os
import pandas as pd
import numpy as np
import matplotlib

# 强制使用 Agg 后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==================== 配置区 ====================
param_name = 'neg'  # 参数文件夹名称
base_path = './2'  # 数据集所在的根目录
datasets = ['B', 'C', 'F']  # 要对比的三个数据集

# 必须的阈值列表（X轴）
required_values = [50, 60, 70, 80, 90]

# 绘图配置
xlabel = r'$\theta_{neg}$' # X轴标签
figsize = (10, 7)  # 大画布
dpi = 150
save_folder = './combined_results'  # 图片保存位置
save_filename = f'combined_{param_name}_auc.png'

# 字体大小配置 (大字体)
FONT_SIZE_LABEL = 30
FONT_SIZE_TICK = 30
FONT_SIZE_LEGEND = 25

# 线条配置
LINE_WIDTH = 1.5  # 细线
MARKER_SIZE = 10  # 大标记点

# 样式配置 (B/C/F 固定配色)
styles = {
    'B': {'color': '#4E79A7', 'marker': 'o', 'label': 'Bdataset'},  # 蓝色
    'C': {'color': '#F28E2B', 'marker': 's', 'label': 'Cdataset'},  # 橙色
    'F': {'color': '#B5CE4E', 'marker': '^', 'label': 'Fdataset'}  # 红色
}

# ================================================

# ---------- 读取数据 ----------
# 结构: all_data = { 'B': [auc_50, auc_60...], ... }
all_data = {}

print(f"准备读取数据集: {datasets}，参数: {param_name}")

for ds in datasets:
    folder_path = os.path.join(base_path, ds, param_name)
    auc_list = []

    # 遍历必须的阈值
    for val in required_values:
        fname = f"{val}.xlsx"
        file_path = os.path.join(folder_path, fname)

        if not os.path.exists(file_path):
            auc_list.append(np.nan)  # 占位
            continue

        try:
            df = pd.read_excel(file_path, index_col=0)

            if 'Mean' not in df.index:
                auc_list.append(np.nan)
                continue

            row = df.loc['Mean']
            if 'AUC' not in row:
                auc_list.append(np.nan)
                continue

            val_auc = float(row['AUC'])
            auc_list.append(val_auc)

        except Exception as e:
            print(f"  [读取失败] {ds}/{fname}: {e}")
            auc_list.append(np.nan)

    all_data[ds] = auc_list
    print(f"数据集 {ds} 读取完毕。")

# ---------- 准备绘图数据 ----------
# X轴位置：使用等距位置 (0, 1, 2, 3, 4)
positions = np.arange(len(required_values))
xticklabels = [str(v) for v in required_values]

# 计算 Y 轴范围
all_values = []
for v_list in all_data.values():
    all_values.extend([x for x in v_list if not np.isnan(x)])

if not all_values:
    print("警告：未读取到任何有效数据，使用默认范围。")
    y_min, y_max = 0.5, 1.0
else:
    y_min = max(min(all_values) - 0.01, 0.0)
    y_max = min(max(all_values) + 0.01, 1.0)

# ---------- 绘图 ----------
plt.figure(figsize=figsize, dpi=dpi)

for ds in datasets:
    y_data = all_data[ds]
    style = styles[ds]

    # 绘制折线
    plt.plot(positions, y_data,
             marker=style['marker'],
             color=style['color'],
             linewidth=LINE_WIDTH,
             markersize=MARKER_SIZE,
             label=style['label'])

# 装饰
# plt.title(...) # 已移除标题
plt.xlabel(xlabel, fontsize=FONT_SIZE_LABEL)
# plt.ylabel(...) # 已移除纵坐标标题

# 设置刻度
plt.xticks(positions, xticklabels, fontsize=FONT_SIZE_TICK)
plt.yticks(fontsize=FONT_SIZE_TICK)

plt.xlim(-0.5, len(positions) - 0.5)
plt.ylim(y_min, y_max)

plt.legend(fontsize=FONT_SIZE_LEGEND)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# ---------- 保存 ----------
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder, save_filename)
plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
print(f"\n图已保存: {save_path}")
plt.close()