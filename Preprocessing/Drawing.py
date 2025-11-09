import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
"""
绘制实验用12张每月LST
"""

def read_tiff_with_rasterio(tiff_path):
    """使用rasterio读取TIFF文件并返回numpy数组。"""
    with rasterio.open(tiff_path) as dataset:
        return dataset.read(1)


def plot_temperature(tiff_path, output_folder, vmin=None, vmax=None):
    """
    读取地表温度TIFF文件，应用颜色映射，并根据文件名设置正确的月份并保存结果图像。

    :param tiff_path: TIFF文件的路径。
    :param output_folder: 输出JPEG文件的文件夹。
    :param vmin: 色条的最小值。
    :param vmax: 色条的最大值。
    """
    temperature_data = read_tiff_with_rasterio(tiff_path)
    # 从文件名解析月份
    month_str = os.path.basename(tiff_path)[-6:-4]  # 提取月份
    year_month = f"2022-{month_str} Surface temperature"

    # 绘制图像
    plt.figure(figsize=(8, 6))
    reversed_colormap = plt.cm.RdYlGn.reversed()
    plt.imshow(temperature_data, aspect='auto', cmap=reversed_colormap, vmin=vmin, vmax=vmax)
    plt.colorbar(label='Temperature Value')
    plt.title(year_month)

    # 保存图像
    output_path = os.path.join(output_folder, f"Surface_Temperature_{year_month.replace(' ', '_')}.jpg")
    plt.savefig(output_path, dpi=300)
    plt.close()


def process_folder(input_folder, output_folder, vmin=None, vmax=None):
    """
    处理文件夹中的所有TIFF文件，并根据文件名中的月份信息调整图像标题。

    :param input_folder: 包含TIFF文件的输入文件夹路径。
    :param output_folder: 用于存放输出JPEG文件的文件夹路径。
    :param vmin: 全局色条的最小值。
    :param vmax: 全局色条的最大值。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.startswith('Predict_LST_') and (filename.endswith('.tif') or filename.endswith('.tiff')):
            file_path = os.path.join(input_folder, filename)
            plot_temperature(file_path, output_folder, vmin=vmin, vmax=vmax)


# 用法示例
input_folder = 'F:/MyProjects/MachineLearning/Data/Final_Data/Predict_Data'
output_folder = 'F:/MyProjects/MachineLearning/Data/Final_Data/Pictures/LST'
process_folder(input_folder, output_folder, vmin=None, vmax=None)  # 这里设置色条的最大最小值
