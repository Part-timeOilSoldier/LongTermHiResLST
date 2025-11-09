import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FixedLocator
from PIL import Image

# 输入文件夹和输出文件夹
lst_folder = 'F:/MyProjects/MachineLearning/Data/Usable_Data/LST_30m'
output_folder = 'F:/论文/论文基础/超大字体图片/LST'
final_output_path = 'F:/论文/论文基础/超大字体图片/LST_combined.png'

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取LST文件列表
lst_files = sorted([f for f in os.listdir(lst_folder) if f.endswith('.tif')])

# 定义色彩条的上下阈值
vmin = 265
vmax = 335

# 创建自定义颜色映射，将0设置为白色
cmap = plt.get_cmap('RdYlGn_r')
colors = cmap(np.arange(cmap.N))
custom_cmap = ListedColormap(colors)

# 日期和月份名称
dates = ['01/15', '02/24', '03/12', '04/24', '05/24', '06/16', '07/02', '08/14', '09/19', '10/22', '11/06', '12/24']

# 遍历LST文件并生成彩色图片
for idx, lst_file in enumerate(lst_files):
    lst_path = os.path.join(lst_folder, lst_file)
    output_path = os.path.join(output_folder, f'{os.path.splitext(lst_file)[0]}.png')

    with rasterio.open(lst_path) as src:
        lst_data = src.read(1)

    # 绘制LST数据
    plt.figure(figsize=(10, 8))
    masked_data = np.ma.masked_where(lst_data == 0, lst_data)
    plt.imshow(masked_data, cmap=custom_cmap, vmin=vmin, vmax=vmax)  # 使用自定义配色方案
    cbar = plt.colorbar(extend='both', extendfrac=0.05, fraction=0.046, pad=0.04)  # 修改颜色条的extend样式和长度

    # 设置色条的字体和标题字体
    tick_locator = FixedLocator(np.arange(vmin, vmax + 1, 10))
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.yaxis.set_tick_params(labelsize=18)  # 增大色条数值字体
    cbar.ax.set_yticklabels(cbar.ax.get_yticks(), fontname='Times New Roman', fontsize=22)  # 增大色条数值字体

    # 删除色条标识
    cbar.set_label('')

    # 为所有值添加黑色直线
    for tick in cbar.ax.get_yticks():
        cbar.ax.axhline(tick, color='black', linewidth=3)

    # 设置标题为对应的月份和日期并设置为新罗马字体
    plt.title(f'{dates[idx]}', fontsize=30, fontname='Times New Roman')  # 增大标题字体

    plt.axis('off')

    # 保存图像
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

print("LST images have been generated and saved to the output folder.")

# 拼接图片
images = [Image.open(os.path.join(output_folder, f'{os.path.splitext(lst_file)[0]}.png')) for lst_file in lst_files]

# 每行的图片数量和每列的图片数量
rows, cols = 3, 4

# 每张图片的尺寸
width, height = images[0].size

# 创建新图像用于拼接
combined_image = Image.new('RGB', (cols * width, rows * height))

# 将图片拼接到新图像中
for idx, image in enumerate(images):
    row, col = divmod(idx, cols)
    combined_image.paste(image, (col * width, row * height))

# 保存拼接后的图像
combined_image.save(final_output_path)

print("Combined image has been generated and saved.")
