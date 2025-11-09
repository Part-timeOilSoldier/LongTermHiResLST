import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from typing import Dict, Literal, Sequence, Tuple

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
    """根据数组绘制地表温度图，并保存到指定路径，使用灰度色阶显示"""
    plt.figure(figsize=(8, 6))
    img = plt.imshow(array, cmap='gray', vmin=np.min(array), vmax=335)
    plt.colorbar(img, label='单位(K)')
    plt.title('30m研究区地表温度图')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_lst_990m_30m(array, save_path):
    """根据数组绘制地表温度图，并保存到指定路径，使用灰度色阶显示"""
    plt.figure(figsize=(8, 6))
    img = plt.imshow(array, cmap='gray', vmin=np.min(array), vmax=320)
    plt.colorbar(img, label='单位(K)')
    plt.title('990m研究区地表温度图')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_ndvi(array, save_path):
    """根据数组绘制NDVI图，并保存到指定路径，使用反转灰度色阶显示"""
    plt.figure(figsize=(8, 6))
    img = plt.imshow(array, cmap='gray', vmin=0, vmax=1)
    plt.colorbar(img, label='Normalized NDVI Value')
    plt.title('NDVI Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_type(land_use_array, output_path):
    """绘制土地利用类型图，使用四种灰度级别，并在图像旁边添加图例"""
    # Define four shades of gray
    gray_shades = {1: '#dddddd', 2: '#aaaaaa', 5: '#777777', 8: '#333333'}
    nrows, ncols = land_use_array.shape
    color_array = np.empty((nrows, ncols, 3), dtype=float)

    for key, color in gray_shades.items():
        color_array[land_use_array == key] = matplotlib.colors.to_rgb(color)

    # Plot the image
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.imshow(color_array)

    # Create a legend
    patches = [matplotlib.patches.Patch(color=color, label=label) for label, color in gray_shades.items()]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)


def main(tiff_path_need, save_path):
    # Read data
    temperature_data = read_tiff_with_rasterio(tiff_path_need)

    # Call the appropriate plotting function based on the save_path
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


# Define paths and execute the main function
tiff_path1 = ('F:/MyProjects/MachineLearning/Data/Usable_Data/LST_30m'
              '/LC08_L2SP_121038_20220616_20230411_02_T1_ST_B10_processed.tif')
tiff_path2 = 'F:/MyProjects/MachineLearning/Data/Usable_Data/LST_30m_990m/LST_Day_1km_20220616.tif'
tiff_path3 = 'F:/MyProjects/MachineLearning/Data/Usable_Data/NDVI_30m/NDVI_20220616.tif'
tiff_path4 = 'F:/MyProjects/MachineLearning/Data/Usable_Data/Type_30m/anhuiwgs84_6.tif'
output_path1 = 'F:/MyProjects/MachineLearning/Data/Final_Data/Pictures/GrayLST_30m'
output_path2 = 'F:/MyProjects/MachineLearning/Data/Final_Data/Pictures/GrayLST_990m_30m'
output_path3 = 'F:/MyProjects/MachineLearning/Data/Final_Data/Pictures/GrayNDVI_30m'
output_path4 = 'F:/MyProjects/MachineLearning/Data/Final_Data/Pictures/GrayType_30m'
main(tiff_path1, output_path1)
main(tiff_path2, output_path2)
main(tiff_path3, output_path3)
main(tiff_path4, output_path4)
