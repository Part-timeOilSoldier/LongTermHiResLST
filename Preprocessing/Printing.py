import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from typing import Dict, Literal, Sequence, Tuple
import matplotlib.patches as mpatches

# 指定支持中文的字体，这里以微软雅黑为例
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['font.size'] = 10  # 可以调整字体大小
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号


def read_tiff_with_rasterio(path):
    """使用rasterio库读取TIFF文件并返回数据数组"""
    with rasterio.open(path) as src:
        array = src.read(1)
    return array


def plot_lst_30m(array, save_path):
    """根据数组绘制地表温度图，并保存到指定路径"""
    # 定义颜色映射
    ColorMapDict = Dict[Literal['red', 'green', 'blue'], Sequence[Tuple[float, float, float]]]

    cdict: ColorMapDict = {
        'red': ((0.0, 0.0, 0.0), (0.25, 0.0, 0.0), (0.75, 0.8, 0.8), (1.0, 0.5, 0.5)),
        'green': ((0.0, 0.0, 0.0), (0.25, 0.5, 0.5), (0.75, 0.8, 0.8), (1.0, 0.0, 0.0)),
        'blue': ((0.0, 0.2, 0.2), (0.25, 0.5, 0.5), (0.75, 0.0, 0.0), (1.0, 0.0, 0.0))
    }
    custom_cmap = mcolors.LinearSegmentedColormap('custom_cmap', cdict)

    # 绘制地表温度图
    plt.figure(figsize=(8, 6))
    img = plt.imshow(array, cmap=custom_cmap, vmin=np.min(array), vmax=335)
    plt.colorbar(img, label='单位(K)')
    plt.title('30m研究区地表温度图')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_lst_990m_30m(array, save_path):
    """根据数组绘制地表温度图，并保存到指定路径"""
    # 定义颜色映射
    ColorMapDict = Dict[Literal['red', 'green', 'blue'], Sequence[Tuple[float, float, float]]]

    cdict: ColorMapDict = {
        'red': ((0.0, 0.0, 0.0), (0.25, 0.0, 0.0), (0.75, 0.8, 0.8), (1.0, 0.5, 0.5)),
        'green': ((0.0, 0.0, 0.0), (0.25, 0.5, 0.5), (0.75, 0.8, 0.8), (1.0, 0.0, 0.0)),
        'blue': ((0.0, 0.2, 0.2), (0.25, 0.5, 0.5), (0.75, 0.0, 0.0), (1.0, 0.0, 0.0))
    }
    custom_cmap = mcolors.LinearSegmentedColormap('custom_cmap', cdict)

    # 绘制地表温度图
    plt.figure(figsize=(8, 6))
    img = plt.imshow(array, cmap=custom_cmap, vmin=np.min(array), vmax=320)
    plt.colorbar(img, label='单位(K)')
    plt.title('990m研究区地表温度图')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_ndvi(array, save_path):
    # 使用matplotlib预定义的反转颜色映射
    reversed_colormap = plt.cm.RdYlGn.reversed()

    # 绘制NDVI图
    plt.figure(figsize=(8, 6))
    img = plt.imshow(array, cmap=reversed_colormap, vmin=0, vmax=1)  # 使用反转的colormap
    plt.colorbar(img, label='Normalized NDVI Value')
    plt.title('NDVI Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig(save_path, dpi=300)
    plt.close()  # 关闭plt对象，以防止内存泄漏


def plot_type(land_use_array, output_path):
    """
        绘制土地利用类型图，直接根据类型映射颜色，并在图像旁边添加图例。将“其他”更改为“建筑”，并使用更鲜明的颜色。

        :param land_use_array: 二维数组，表示土地利用类型。
        :param output_path: 图像保存路径。
        """
    # 更新颜色映射，使用更易区分的颜色
    color_map = {1: 'limegreen',  # 田地
                 2: 'forestgreen',  # 森林
                 5: 'royalblue',  # 水域
                 8: 'orangered'}  # 建筑

    nrows, ncols = land_use_array.shape
    color_array = np.empty((nrows, ncols, 3), dtype=float)

    for key, color in color_map.items():
        color_array[land_use_array == key] = mcolors.to_rgb(color)

    # 绘制图像
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.imshow(color_array)

    # 创建图例，将“其他”更改为“建筑”
    legend_labels = {'田地': 'limegreen', '森林': 'forestgreen', '水域': 'royalblue', '建筑': 'orangered'}
    patches = [mpatches.Patch(color=color, label=label) for label, color in legend_labels.items()]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)


def main(tiff_path_need, save_path):
    # 读取数据
    temperature_data = read_tiff_with_rasterio(tiff_path_need)

    # 判断save_path，并调用相应的绘图函数
    if 'LST_30m' in save_path:
        plot_lst_30m(temperature_data, save_path)
    elif 'LST_990m_30m' in save_path:
        plot_lst_990m_30m(temperature_data, save_path)
    elif 'NDVI_30m' in save_path:
        plot_ndvi(temperature_data, save_path)
    elif 'Type_30m' in save_path:
        plot_type(temperature_data, save_path)
    else:
        print("Invalid save path provided. No plot will be generated.")


tiff_path1 = ('F:/MyProjects/MachineLearning/Data/Usable_Data/LST_30m'
              '/LC09_L2SP_121038_20220616_20230411_02_T1_ST_B10_processed.tif')
tiff_path2 = 'F:/MyProjects/MachineLearning/Data/Usable_Data/LST_30m_990m/LST_Day_1km_20220616.tif'
tiff_path3 = 'F:/MyProjects/MachineLearning/Data/Usable_Data/NDVI_30m/NDVI_20230411.tif'
tiff_path4 = 'F:/MyProjects/MachineLearning/Data/Usable_Data/Type_30m/anhuiwgs84.tif'
output_path1 = 'F:/MyProjects/MachineLearning/Data/Final_Data/Pictures/LST_30m'
output_path2 = 'F:/MyProjects/MachineLearning/Data/Final_Data/Pictures/LST_990m_30m'
output_path3 = 'F:/MyProjects/MachineLearning/Data/Final_Data/Pictures/NDVI_30m'
output_path4 = 'F:/MyProjects/MachineLearning/Data/Final_Data/Pictures/Type_30m'
main(tiff_path1, output_path1)
main(tiff_path2, output_path2)
main(tiff_path3, output_path3)
main(tiff_path4, output_path4)
