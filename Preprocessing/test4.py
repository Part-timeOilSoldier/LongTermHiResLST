import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from osgeo import gdal

# 文件夹路径
folder_path = 'F:/MyProjects/MachineLearning/Data/Final_Data/Predict_Data'

# 日期和R²值
dates = ['01/15', '02/24', '03/12', '04/24', '05/24', '06/16', '07/02', '08/14', '09/19', '10/22', '11/06', '12/24']
r_squared = [0.80, 0.74, 0.85, 0.85, 0.80, 0.86, 0.78, 0.65, 0.74, 0.81, 0.81, 0.72]

# 获取文件夹中的文件列表
file_list = sorted(os.listdir(folder_path))

# 创建图表
fig, axes = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)
cmap = plt.get_cmap('Spectral_r')
norm = plt.Normalize(vmin=265, vmax=335)

# 遍历文件并绘制
for i, file in enumerate(file_list):
    if file.endswith('.tif'):
        file_path = os.path.join(folder_path, file)

        # 打开LST数据文件
        dataset = gdal.Open(file_path)
        band = dataset.GetRasterBand(1)
        lst_data = band.ReadAsArray()

        # 计算行列索引
        row = i // 4
        col = i % 4

        # 绘制LST数据
        ax = axes[row, col]
        cax = ax.imshow(lst_data, cmap=cmap, norm=norm)
        ax.set_title(f"{dates[i]} $R^2$={r_squared[i]:.2f}", fontsize=24, fontname='Times New Roman')
        ax.axis('off')

# 添加一个颜色条
cbar = fig.colorbar(cax, ax=axes, orientation='vertical', fraction=0.02, pad=0.04, aspect=40)

# 设置颜色条的标签字体
ticks = np.linspace(265, 335, num=8)
cbar.ax.yaxis.set_major_locator(FixedLocator(ticks))
cbar.ax.set_yticklabels([f'{tick:.0f}' for tick in ticks], fontname='Times New Roman', fontsize=24, weight='bold')

# 为颜色条的每个值添加黑色的线并加粗线条
for tick in ticks:
    cbar.ax.axhline(tick, color='black', linewidth=2)

# 加粗颜色条
cbar.outline.set_linewidth(2)

# 保存图表
plt.savefig('F:/MyProjects/MachineLearning/Data/Final_Data/LST_Monthly_Plot.png', dpi=1000)

# 显示图表
plt.show()
