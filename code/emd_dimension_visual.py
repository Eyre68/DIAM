import os
import pandas as pd
import numpy as np
import matplotlib

# 如果在服务器运行且没有图形界面，保持这一行；如果在本地 IDE (如 PyCharm/Jupyter) 可注释掉
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==================== 配置区 ====================
param_name = 'hgt_out_dim'  # 固定参数名
base_path = './2'  # 数据集所在的根目录
datasets = ['B', 'C', 'F']  # 要对比的三个数据集
# dataset_names = {'B': 'Bdataset', 'C': 'Cdataset', 'F': 'Fdataset'} # (此处未使用)

# 必须的维度列表（X轴）
required_dims = [32, 64, 128, 256, 512]

# 绘图配置
# title         = 'Impact of Embedding Dimension (AUC)' # 已移除
xlabel = 'd'
# ylabel        = 'AUC Score'                           # 已移除
figsize = (10, 7)  # 稍微调大画布以适应大字体
dpi = 150
save_folder = './combined_results'  # 图片保存位置
save_filename = f'combined_{param_name}_auc_large_font.png'

# 字体大小配置 (约为原来的2倍)
FONT_SIZE_LABEL = 30
FONT_SIZE_TICK = 30
FONT_SIZE_LEGEND = 25

# 线条配置
LINE_WIDTH = 1.5  # 更细的线
MARKER_SIZE = 10  # 稍微大一点的标记点以匹配大字体

# 样式配置：分别为 B, C, F 指定颜色和标记
styles = {
    'B': {'color': '#4E79A7', 'marker': 'o', 'label': 'Bdataset'},  # 蓝色
    'C': {'color': '#F28E2B', 'marker': 's', 'label': 'Cdataset'},  # 橙色
    'F': {'color': '#B5CE4E', 'marker': '^', 'label': 'Fdataset'}  # 红色
}

# ================================================

# ---------- 读取数据 (保持不变) ----------
# 结构: all_data = { 'B': [auc_32, auc_64...], 'C': [...], ... }
all_data = {}

print(f"准备读取数据集: {datasets}")

# 1. 遍历每个数据集
for ds in datasets:
    folder_path = os.path.join(base_path, ds, param_name)
    auc_list = []

    # print(f"\n--- 正在处理数据集: {ds} (路径: {folder_path}) ---")

    # 2. 遍历维度
    for dim in required_dims:
        fname = f"{dim}.xlsx"
        file_path = os.path.join(folder_path, fname)

        if not os.path.exists(file_path):
            # print(f"  [缺失] 文件未找到: {fname}")
            auc_list.append(np.nan)  # 用 nan 占位
            continue

        try:
            df = pd.read_excel(file_path, index_col=0)

            if 'Mean' not in df.index:
                # print(f"  [格式错误] {fname} 无 'Mean' 行")
                auc_list.append(np.nan)
                continue

            row = df.loc['Mean']
            if 'AUC' not in row:
                # print(f"  [格式错误] {fname} Mean行缺少 AUC 列")
                auc_list.append(np.nan)
                continue

            val = float(row['AUC'])
            auc_list.append(val)

        except Exception as e:
            print(f"  [读取失败] {ds}/{fname}: {e}")
            auc_list.append(np.nan)

    all_data[ds] = auc_list
    print(f"数据集 {ds} 读取完毕。")

# ---------- 准备绘图数据 ----------
# X轴位置：等距
positions = np.arange(len(required_dims))
xticklabels = [str(d) for d in required_dims]

# 计算 Y 轴范围 (为了美观自动调整)
all_values = []
for v_list in all_data.values():
    all_values.extend([x for x in v_list if not np.isnan(x)])

if not all_values:
    print("警告：未读取到任何有效 AUC 数据，将使用默认坐标轴范围。")
    y_min, y_max = 0.5, 1.0
else:
    y_min = max(min(all_values) - 0.01, 0.0)
    y_max = min(max(all_values) + 0.01, 1.0)

# ---------- 绘图 ----------
plt.figure(figsize=figsize, dpi=dpi)

for ds in datasets:
    y_data = all_data[ds]
    style = styles[ds]

    plt.plot(positions, y_data,
             marker=style['marker'],
             color=style['color'],
             linewidth=LINE_WIDTH,  # 修改：更细的线
             markersize=MARKER_SIZE,
             label=style['label'])

# 装饰
# plt.title(...) # 修改：已移除标题
plt.xlabel(xlabel, fontsize=FONT_SIZE_LABEL)  # 修改：大字体 X 轴标签
# plt.ylabel(...) # 修改：已移除 Y 轴标签

# 设置刻度字体大小
plt.xticks(positions, xticklabels, fontsize=FONT_SIZE_TICK)  # 修改：大字体 X 刻度
plt.yticks(fontsize=FONT_SIZE_TICK)  # 修改：大字体 Y 刻度

plt.xlim(-0.5, len(positions) - 0.5)
plt.ylim(y_min, y_max)

plt.legend(fontsize=FONT_SIZE_LEGEND)  # 修改：大字体图例
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# ---------- 保存 ----------
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder, save_filename)
plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
print(f"\n汇总图已保存 (无标题/大字体): {save_path}")
plt.close()