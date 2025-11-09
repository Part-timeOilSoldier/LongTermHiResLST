import os
import numpy as np
from osgeo import gdal

# 文件夹路径
ndvi_folder = r'F:\MyProjects\MachineLearning\Data\Usable_Data\NDVI_30m'
lst_folder = r'F:\MyProjects\MachineLearning\Data\Usable_Data\LST_30m'
lcd_file = r'F:\MyProjects\MachineLearning\Data\Usable_Data\Type_30m\anhuiwgs84_1.tif'

# 获取文件列表
ndvi_files = [os.path.join(ndvi_folder, f) for f in os.listdir(ndvi_folder) if f.endswith('.tif')]
lst_files = [os.path.join(lst_folder, f) for f in os.listdir(lst_folder) if f.endswith('.tif')]

# 排序以确保月份对应
ndvi_files.sort()
lst_files.sort()

# 读取土地利用类型数据
def read_lcd_data(file_path):
    ds = gdal.Open(file_path)
    if ds is None:
        raise FileNotFoundError(f"Failed to open file: {file_path}")

    arr = ds.ReadAsArray()
    ds = None  # 释放资源
    return arr

# 计算两个数组的线性回归斜率
def compute_slope(arr1, arr2):
    mask = np.isfinite(arr1) & np.isfinite(arr2)  # 排除无效值（NaN和inf）
    if np.sum(mask) < 2:
        return np.nan  # 如果有效值少于2个，返回NaN

    x = arr1[mask]
    y = arr2[mask]
    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m  # 返回斜率

# 主循环
for ndvi_file, lst_file in zip(ndvi_files, lst_files):
    # 提取月份和年份
    ndvi_month = ndvi_file.split('_')[-1][4:6]  # 从文件名中提取月份
    lst_month = lst_file.split('_')[-7][4:6]  # 从文件名中提取月份

    if ndvi_month != lst_month:
        raise ValueError("NDVI and LST files do not match for the same month.")

    # 读取NDVI和LST数据
    ndvi_data = gdal.Open(ndvi_file).ReadAsArray().astype(float)
    lst_data = gdal.Open(lst_file).ReadAsArray().astype(float)

    # 读取土地利用类型数据
    lcd_data = read_lcd_data(lcd_file)

    # 获取不同土地利用类型的统计信息
    unique_labels = np.unique(lcd_data)
    for label in unique_labels:
        # 只处理土地利用类型为1、2、5、8
        if label in [1, 2, 5, 8]:
            ndvi_masked = np.where(lcd_data == label, ndvi_data, np.nan)
            lst_masked = np.where(lcd_data == label, lst_data, np.nan)

            # 计算线性回归斜率
            slope = compute_slope(lst_masked, ndvi_masked)

            # 打印结果
            print(f"Month: {ndvi_month}, Land Use Type: {label}, Slope of Regression Line: {slope}")
