"""
这是一个在项目中自己使用的Tool库
目的是将与主体无关的函数放入其中
使得主题脚本逻辑结构更加清晰
"""
import glob
import os
import shutil
from multiprocessing import Pool

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import cupy as cp
from matplotlib import pyplot as plt
from osgeo import gdal, osr
from datetime import datetime, timedelta
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def convert_day(year, doy):
    """
    将年份和年内日序转换为标准日期格式。
    :param year: 年份，如2022。
    :param doy: 年内日序，如335。
    :return: 标准日期字符串，格式为YYYY-MM-DD。
    """
    date = datetime(year, 1, 1) + timedelta(doy - 1)
    return date.strftime('%Y%m%d')


def extract_tiff(hdf4_file_path, dataset_name, output_folder, date_str):
    """
    从HDF4文件中提取特定数据集并保存为TIFF格式。
    :param hdf4_file_path: HDF4文件的路径。
    :param dataset_name: 要提取的数据集名称。
    :param output_folder: TIFF文件的输出目录。
    :param date_str: 提取数据的日期字符串。
    :return: None
    """
    try:
        # 打开传入的HDF4文件
        hdf_file = gdal.Open(hdf4_file_path)

        # 找到HDF4文件中特定的数据集(如LST和QC)
        sub_datasets = hdf_file.GetSubDatasets()  # GetSubDatasets()函数会打开HDF4文件的并将文件内的所有子文件(LST,QC等)赋值给sub_datasets
        target_dataset = None  # 初始化一个空变量方便后面存储遍历获得的需要数据
        for sub_dataset in sub_datasets:  # 遍历sub_datasets将每次数据存在sub_dataset中，这里的sub_dataset只有一个波段
            if dataset_name in sub_dataset[0]:  # 判断要找的名称是否在sub_dataset的名称中，由于只有一个波段所以索引为0
                target_dataset = sub_dataset[0]  # 将数据赋值给空变量target_dataset
                break  # 获得想要的值之后退出遍历

        if target_dataset is None:  # 如果没有找到含有相关名称的数据输出未找到
            raise ValueError(f"没有在HDF file中找到{dataset_name}")

        # 打开目标数据，将值赋给sub_dataset
        sub_dataset = gdal.Open(target_dataset)

        # 构建输出文件的路径、名称，命名方式为(需要的名称_日期.tif)
        output_file_path = os.path.join(output_folder, f"{dataset_name}_{date_str}.tif")

        # 确保输出目录存在
        os.makedirs(output_folder, exist_ok=True)

        # 将数据集转换为TIFF格式并保存
        gdal.Translate(output_file_path, sub_dataset)

        # 打印保存成功语句以判断文件是否被正常提取且保存
        print(f"数据 {dataset_name} 已经被保存为 {output_file_path}")

    # 报错打印
    except Exception as e:
        print(f"Error extracting dataset {dataset_name}: {e}")


def process_hdf_files(hdf_folder_path, lst_output_folder):
    """
    处理HDF文件夹内的所有HDF文件，提取LST和QC数据并保存为TIFF格式。
    :param hdf_folder_path: HDF文件夹的路径。
    :param lst_output_folder: LST数据的输出文件夹。
    :return: None
    """
    # 使用glob模块获取HDF文件夹内的所有HDF文件
    hdf_files = glob.glob(os.path.join(hdf_folder_path, "*.hdf"))

    # 遍历每个HDF文件
    for hdf_file in hdf_files:
        # 从文件名中提取年份和年内日序
        basename = os.path.basename(hdf_file)
        parts = basename.split('.')
        year = int(parts[1][1:5])
        doy = int(parts[1][5:])
        # 从文件名中提取网格编号
        grid = int(parts[2][1:3])

        # 转换为标准日期格式
        date_str = convert_day(year, doy)
        # 添加网格编号到日期字符串
        date_str = f"{date_str}_{grid}"

        # 提取LST_Day_1km数据并保存
        extract_tiff(hdf_file, "LST_Day_1km", lst_output_folder, date_str)


def clean_tiff_data(folder_path):
    """
    遍历指定文件夹中的所有 TIFF 文件，计算每个 TIFF 文件中像素值为0的占比，
    并删除那些零值像素占比超过5%的文件。
    :param folder_path: 要清理的 TIFF 文件所在的文件夹路径。
    """
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件扩展名是否为 .tif 或 .tiff，以确定目标文件
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            # 构造文件的完整路径
            file_path = os.path.join(folder_path, filename)

            # 使用 rasterio 库打开并读取 TIFF 文件
            with rasterio.open(file_path, 'r') as src:
                # 仅读取第一个波段的数据
                data = src.read(1)  # 假设我们只对第一个波段感兴趣

            # 计算数据中像素值为0的比例
            zero_pixel_ratio = np.sum(data == 0) / data.size

            # 打印当前文件的零值像素比例
            print(f"{filename}0值占比: {zero_pixel_ratio:.2%}")

            # 判断零值像素的比例是否大于5%
            if zero_pixel_ratio > 0.07:
                # 如果是，则删除该文件
                os.remove(file_path)


def process_modis_lst_data(data_path):
    """
    处理指定路径下的所有LST TIFF数据，将像元数值乘以0.02并覆盖原始文件。
    :param data_path: 包含LST TIFF文件的文件夹路径。
    :return: None
    """
    modis_LST_Day_list = glob.glob(os.path.join(data_path, '*.tif'))

    for lst_file in modis_LST_Day_list:
        with rasterio.open(lst_file) as sds_1:
            sds_1_np = sds_1.read(1, masked=True) * 0.02

            # 创建一个临时文件
            temp_filename = lst_file.replace('.tif', '_temp.tif')
            with rasterio.open(
                    temp_filename,
                    'w',
                    driver='GTiff',
                    height=sds_1.shape[0],
                    width=sds_1.shape[1],
                    count=1,
                    dtype=sds_1_np.dtype,
                    crs=sds_1.crs,
                    transform=sds_1.transform
            ) as dst:
                dst.write(sds_1_np, 1)

        # 删除原文件
        os.remove(lst_file)
        # 将临时文件重命名为原文件名
        shutil.move(temp_filename, lst_file)

    print("LST_990m系数乘法完成。")


def reproject_tiff(input_tiff_path, output_tiff_path):
    """
    将TIFF数据从原始投影转换为UTM Zone 50N WGS84，并将输出数据的分辨率设置为1000米。
    :param input_tiff_path: 输入TIFF文件的路径。
    :param output_tiff_path: 输出TIFF文件的路径。
    :return: None
    """
    # 打开原始TIFF文件
    src_ds = gdal.Open(input_tiff_path)

    # 设置目标投影为UTM Zone 50N WGS84
    dst_proj = osr.SpatialReference()
    dst_proj.SetWellKnownGeogCS("WGS84")
    dst_proj.SetUTM(50, True)

    # 使用gdal.Warp进行重投影，设置输出分辨率为990米
    gdal.Warp(output_tiff_path, src_ds, dstSRS=dst_proj.ExportToWkt(), xRes=990, yRes=990)

    print(f"文件 {input_tiff_path} 已重投影到 {output_tiff_path}，输出分辨率为990米")


def reproject_folder(folder_path):
    """
    将指定文件夹内所有TIFF文件的投影转换为UTM Zone 50N WGS84，在原地更新。
    :param folder_path: 包含TIFF文件的文件夹路径。
    :return: None
    """
    # 获取文件夹中的所有TIFF文件
    tiff_files = glob.glob(os.path.join(folder_path, '*.tif'))

    for input_tiff_path in tiff_files:
        temp_tiff_path = input_tiff_path.replace('.tif', '_temp.tif')

        # 对每个文件进行重投影到临时文件
        reproject_tiff(input_tiff_path, temp_tiff_path)

        # 替换原文件
        os.remove(input_tiff_path)
        shutil.move(temp_tiff_path, input_tiff_path)


def merge_lst_tiff_by_date(folder_path):
    """
    合并同一日期但不同网格编号的TIFF文件。
    :param folder_path: 包含TIFF文件的文件夹路径。
    :return: None
    """
    # 获取所有TIFF文件
    all_tiffs = glob.glob(os.path.join(folder_path, '*.tif'))

    # 按日期对文件进行分组
    grouped_files = {}
    for file in all_tiffs:
        # 从文件名中提取日期
        date = os.path.basename(file).split('_')[3]
        if date not in grouped_files:
            grouped_files[date] = []
        grouped_files[date].append(file)

    # 遍历每个日期的组，合并相应的文件
    for date, files in grouped_files.items():
        # 读取每个文件并添加到待合并列表
        src_files_to_mosaic = [rasterio.open(file) for file in files]

        # 合并当前日期的文件
        mosaic, out_trans = merge(src_files_to_mosaic)

        # 关闭打开的文件
        for src in src_files_to_mosaic:
            src.close()

        # 输出文件名（移除网格编号）
        out_filename = f'LST_Day_1km_{date}.tif'
        out_path = os.path.join(folder_path, out_filename)

        # 写入合并后的文件
        out_meta = src_files_to_mosaic[0].meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": mosaic.shape[1],
                         "width": mosaic.shape[2],
                         "transform": out_trans})

        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(mosaic)

        # 删除原文件
        for file in files:
            os.remove(file)

        print(f"文件 {out_filename} 已合并并保存。")

    print("所有同日LST_990m数据已拼接完成。")


def calculate_ndvi(red_band, nir_band):
    """计算NDVI值，处理零除问题"""
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir_band - red_band) / (nir_band + red_band)
        ndvi[np.isnan(ndvi)] = 0  # 将NaN值替换为0
    return ndvi


def find_landsat_files(folder_path, date, band_pattern):
    """根据提供的日期和波段模式查找Landsat文件"""
    pattern = os.path.join(folder_path, f"*_{date}_*_{band_pattern}.TIF")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found for pattern {pattern}")
    return files[0]


def process_landsat_data(input_path, output_path, date):
    """处理Landsat数据并计算NDVI"""
    b4_path = find_landsat_files(input_path, date, "B4")
    b5_path = find_landsat_files(input_path, date, "B5")

    # 读取数据
    with rasterio.open(b4_path) as red_src, rasterio.open(b5_path) as nir_src:
        red = red_src.read(1, resampling=Resampling.bilinear)
        nir = nir_src.read(1, resampling=Resampling.bilinear)

        # 计算NDVI
        ndvi = calculate_ndvi(red, nir)

        # 设置输出文件的元数据
        ndvi_meta = red_src.meta
        ndvi_meta.update(dtype=rasterio.float32)

        # 写入NDVI数据
        ndvi_output_path = os.path.join(output_path, f"NDVI_{date}.tif")
        with rasterio.open(ndvi_output_path, 'w', **ndvi_meta) as dst:
            dst.write(ndvi.astype(rasterio.float32), 1)

    print(f"NDVI 已成功计算并保存至 {ndvi_output_path}")


def plot_tif(folder_path, i):
    """
    读取指定文件夹中的第一个.tif文件的第一个波段并绘图。
    :param i: 需要打印的图像索引
    :param folder_path: 包含.tif文件的文件夹路径。
    """
    tif_list = glob.glob(folder_path + "/*.tif")

    if tif_list:
        print(f"在文件夹 '{folder_path}' 中找到 {len(tif_list)} 个.tif文件。")
        first_tif = tif_list[i]
        with rasterio.open(first_tif) as dataset:
            band1 = dataset.read(1)
            plt.imshow(band1, vmin=0, vmax=0.4)
            plt.title(f"First Band: {first_tif}")
            plt.colorbar(label='NDVI')
            plt.show()
    else:
        print(f"在文件夹 '{folder_path}' 中未找到.tif文件。")


def group_landsat_files(input_folder):
    """将 Landsat 文件按日期分组并移动到相应的文件夹中"""

    # 获取所有 TIFF 文件
    tiff_files = glob.glob(os.path.join(input_folder, "*.TIF"))

    for file in tiff_files:
        # 从文件名中提取日期
        parts = file.split('_')
        if len(parts) > 3:
            date = parts[4]
            if len(date) == 8:  # 确保日期格式正确
                # 创建日期文件夹（如果尚不存在）
                date_folder = os.path.join(input_folder, date)
                if not os.path.exists(date_folder):
                    os.makedirs(date_folder)

                # 移动文件到日期文件夹
                shutil.move(file, os.path.join(date_folder, os.path.basename(file)))

    print(f"文件已按日期分组并移动到相应文件夹。")


def process_all_landsat(parent_folder, output_path):
    """遍历父文件夹中的所有子文件夹，并调用 process_landsat_data 函数"""

    # 检查父文件夹是否存在
    if not os.path.exists(parent_folder):
        print(f"父文件夹 {parent_folder} 不存在。")
        return

    # 遍历父文件夹中的所有子目录
    for subdir in os.listdir(parent_folder):
        subdir_path = os.path.join(parent_folder, subdir)

        # 确保是一个目录
        if os.path.isdir(subdir_path):
            # 提取子目录名称作为日期
            date = subdir.split('_')[-1]

            # 调用已定义的 process_landsat_data 函数
            process_landsat_data(subdir_path, output_path, date)
            print(f"已处理子文件夹：{subdir}")


def crop_raster(input_folder, output_folder, utm_coords, size):
    """
    裁切指定的栅格数据文件，并将裁切后的图像保存到输出文件夹。

    该函数遍历输入文件夹中的所有.tif栅格数据文件，根据给定的UTM坐标和裁切大小，
    对每个文件进行裁切操作，并将裁切得到的图像保存到输出文件夹中，文件名与原始文件相同。

    参数:
    - input_folder (str): 输入文件夹的路径，该文件夹包含需要被裁切的.tif文件。
    - output_folder (str): 输出文件夹的路径，裁切后的文件将被保存在这里。
    - utm_coords (tuple): UTM坐标 (x, y)，表示裁切窗口的左上角在原始图像中的位置。
    - size (tuple): 裁切窗口的大小，形式为 (width, height)，单位为像素。

    返回:
    - 无。函数执行后，裁切后的文件将直接保存在指定的输出文件夹中。
    """
    # 使用glob.glob来获取输入文件夹中所有的.tif文件。
    files = glob.glob(os.path.join(input_folder, '*.tif'))

    # 遍历找到的所有.tif文件
    for file in files:
        # 打开当前遍历的.tif文件
        with rasterio.open(file) as src:
            # 将UTM坐标转换为当前栅格数据的像元行列号（即在图像矩阵中的位置）
            row, col = src.index(*utm_coords)

            # 基于行列号和指定的大小创建一个裁切窗口，用于定义要裁切的区域
            window = Window(col, row, size[0], size[1])

            # 从源文件中读取指定窗口的数据，实现裁切功能
            cropped_image = src.read(window=window)

            # 获取原栅格文件的元数据（如坐标系、分辨率等信息），并准备更新这些元数据
            meta = src.meta
            meta.update({
                # 设置输出文件的格式为GeoTIFF
                'driver': 'GTiff',
                # 更新元数据中的高度和宽度为裁切窗口的高度和宽度
                'height': window.height,
                'width': window.width,
                # 更新坐标转换信息，以确保裁切后的栅格数据的空间参考与原始数据一致
                'transform': rasterio.windows.transform(window, src.transform)
            })

            # 构建输出文件的完整路径
            output_path = os.path.join(output_folder, os.path.basename(file))

            # 使用更新后的元数据创建新的栅格文件，并写入裁切后的数据
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(cropped_image)


def resample_ndvi(input_folder, output_folder):
    """
    该函数对指定文件夹内的所有单波段NDVI TIFF图像进行重采样。

    参数:
    input_folder: 包含输入TIFF文件的文件夹路径。
    output_folder: 用于存放重采样后TIFF文件的文件夹路径。
    """
    new_height, new_width = 64, 71  # 重采样后的图像尺寸

    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):  # 确保处理的是TIFF文件
            with rasterio.open(os.path.join(input_folder, filename)) as src:
                # 读取图像数据
                data = src.read(1)  # 读取第一个波段

                # 计算变换因子
                height_factor, width_factor = data.shape[0] / new_height, data.shape[1] / new_width

                # 初始化重采样后的数据数组
                resampled_data = np.zeros((new_height, new_width))

                # 重采样过程
                for i in range(new_height):
                    for j in range(new_width):
                        start_i, start_j = int(i * height_factor), int(j * width_factor)
                        end_i, end_j = int((i + 1) * height_factor), int((j + 1) * width_factor)
                        # 计算新像素的NDVI值，即原像素区域NDVI值的平均
                        resampled_data[i, j] = np.mean(data[start_i:end_i, start_j:end_j])

                # 保存重采样后的图像
                new_filename = filename.replace('.tif', '_resampled.tif')
                new_filepath = os.path.join(output_folder, new_filename)

                # 计算新图像的变换参数
                new_transform = src.transform * src.transform.scale(
                    (src.width / resampled_data.shape[1]),
                    (src.height / resampled_data.shape[0])
                )

                # 写入新文件
                with rasterio.open(
                        new_filepath, 'w', driver='GTiff',
                        height=new_height, width=new_width,
                        count=1, dtype='float32',
                        crs=src.crs, transform=new_transform
                ) as dst:
                    dst.write(resampled_data, 1)

    output_files = [f for f in os.listdir(output_folder) if f.endswith('_resampled.tif')]
    if output_files:
        first_resampled_file = output_files[0]
        with rasterio.open(os.path.join(output_folder, first_resampled_file)) as src:
            print(f"文件名: {first_resampled_file}")
            print(f"投影: {src.crs}")
            print(f"宽度: {src.width}")
            print(f"高度: {src.height}")
            print(f"单个像元分辨率: {src.res[0]}m (在x方向), {src.res[1]}m (在y方向)")
    else:
        print("输出文件夹中没有找到重采样后的TIFF文件。")


def resample_lst_to_30m(input_folder, output_folder):
    """
    重采样输入文件夹中的单波段地表温度数据 TIFF 文件，
    从990m分辨率重采样到30m分辨率，并保存到输出文件夹中。

    参数:
    input_folder (str): 包含需要处理的TIFF文件的输入文件夹路径。
    output_folder (str): 保存重采样后TIFF文件的输出文件夹路径。
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with rasterio.open(input_path) as src:
                # 计算新的分辨率对应的变换矩阵和新的宽高
                new_transform = src.transform * src.transform.scale(
                    (src.width / 2343),
                    (src.height / 2112)
                )

                # 设置重采样后的新宽高
                new_width = 2343
                new_height = 2112

                # 重采样数据
                resampled_data = src.read(
                    1,
                    out_shape=(new_height, new_width),
                    resampling=Resampling.nearest
                )

                # 写入重采样后的数据到新文件
                with rasterio.open(
                        output_path, 'w',
                        driver='GTiff',
                        height=new_height,
                        width=new_width,
                        count=1,
                        dtype=resampled_data.dtype,
                        crs=src.crs,
                        transform=new_transform
                ) as dst:
                    dst.write(resampled_data, 1)

                print(f"重采样完成并保存到 {output_path}")


def preprocessing_B10folder(input_folder, output_folder, utm_coords, size):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.TIF') or filename.endswith('.tiff'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + "_processed.tif")

            with rasterio.open(input_path) as src:
                row, col = src.index(*utm_coords)
                window = Window(col, row, size[0], size[1])
                cropped_image = src.read(window=window, out_dtype='float32')  # Ensure the data is read as float32
                meta = src.meta.copy()
                meta.update({
                    'dtype': 'float32',  # Ensure the metadata specifies float32 type
                    'height': window.height,
                    'width': window.width,
                    'transform': rasterio.windows.transform(window, src.transform)
                })

                # Continue with your processing
                cropped_image = 0.00341802 * cropped_image + 149

                valid_pixels = cropped_image[cropped_image >= 270]
                mean_value = valid_pixels.mean() if valid_pixels.size > 0 else 0
                cropped_image[(cropped_image < 270) & (cropped_image > 0)] = mean_value
                cropped_image[cropped_image == 0] = mean_value

                # Save the processed image
                with rasterio.open(output_path, 'w', **meta) as dst:
                    dst.write(cropped_image)

    print(f"All files processed and saved in {output_folder}")


def crop_type_data(input_folder, output_folder, utm_coords, size):
    """
    裁切指定的栅格数据文件，并将裁切后的图像保存到输出文件夹。

    该函数遍历输入文件夹中的所有.tif栅格数据文件，根据给定的UTM坐标和裁切大小，
    对每个文件进行裁切操作，并将裁切得到的图像保存到输出文件夹中，文件名与原始文件相同。

    参数:
    - input_folder (str): 输入文件夹的路径，该文件夹包含需要被裁切的.tif文件。
    - output_folder (str): 输出文件夹的路径，裁切后的文件将被保存在这里。
    - utm_coords (tuple): UTM坐标 (x, y)，表示裁切窗口的左上角在原始图像中的位置。
    - size (tuple): 裁切窗口的大小，形式为 (width, height)，单位为像素。

    返回:
    - 无。函数执行后，裁切后的文件将直接保存在指定的输出文件夹中。
    """
    # 使用glob.glob来获取输入文件夹中所有的.tif文件。
    files = glob.glob(os.path.join(input_folder, '*.tif'))

    # 遍历找到的所有.tif文件
    for file in files:
        # 打开当前遍历的.tif文件
        with rasterio.open(file) as src:
            # 将UTM坐标转换为当前栅格数据的像元行列号（即在图像矩阵中的位置）
            row, col = src.index(*utm_coords)

            # 基于行列号和指定的大小创建一个裁切窗口，用于定义要裁切的区域
            window = Window(col, row, size[0], size[1])

            # 从源文件中读取指定窗口的数据，实现裁切功能
            cropped_image = src.read(window=window)

            # 获取原栅格文件的元数据（如坐标系、分辨率等信息），并准备更新这些元数据
            meta = src.meta
            meta.update({
                # 设置输出文件的格式为GeoTIFF
                'driver': 'GTiff',
                # 更新元数据中的高度和宽度为裁切窗口的高度和宽度
                'height': window.height,
                'width': window.width,
                # 保留原始数据的其它内容
                'count': 1,  # 保留原始图像的波段数
                'dtype': src.dtypes[0],
                'crs': src.crs,
                # 更新坐标转换信息，以确保裁切后的栅格数据的空间参考与原始数据一致
                'transform': rasterio.windows.transform(window, src.transform)
            })

            # 构建输出文件的完整路径
            output_path = os.path.join(output_folder, os.path.basename(file))

            # 使用更新后的元数据创建新的栅格文件，并写入裁切后的数据
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(cropped_image)


def print_model_R2(y_test, y_pred, model_name, output_path, r2, mse, mae, density_save_path,
                   degree=2, n_jobs=8,):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用支持更多字符集的字体
    plt.rcParams['axes.unicode_minus'] = False

    # 准备模型评估指标数据，构建一个字典
    summary_data = {
        'Metric': ['R²', 'MAE', 'MSE'],  # 定义指标名称
        'Value': [r2, mae, mse]  # 对应的指标值
    }
    # 将字典转换为DataFrame以便更好的展示
    summary_df = pd.DataFrame(summary_data)

    # 打印模型概要信息
    print("Model Summary：")
    print(summary_df.to_string(index=False))  # 以表格形式打印出来，不显示索引

    # 使用多项式拟合预测值和测试值，degree表示多项式的度
    weights = np.polyfit(y_test, y_pred, degree)
    # 通过拟合得到的系数构建多项式模型
    model = np.poly1d(weights)

    # 构建并打印拟合曲线方程
    equation = "Fit Curve Equation: " + " + ".join([f"{temp:.4f} x^{i}" for i, temp in enumerate(weights[::-1])])
    print(equation)

    z = get_density_data(model_name, y_test, y_pred, 'R2', density_save_path, n_jobs)

    # 使用Seaborn的颜色映射进行上色
    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)

    # 创建一个图形
    plt.figure(figsize=(10, 6))
    # 绘制散点图，颜色由密度决定
    scatter = plt.scatter(y_test, y_pred, c=z, cmap=cmap, s=1, edgecolor='none')
    # 绘制拟合曲线
    plt.plot(np.sort(y_test), model(np.sort(y_test)), color='#000000', label=equation.replace("Curve Equation: ", ""))
    # 添加颜色条
    plt.colorbar(scatter, label='Density', fontsize=25)

    # 设置图表的标签和标题
    plt.xlabel('Actual Value', fontsize=25)
    plt.ylabel('Predicted Values', fontsize=25)
    plt.title(f'{model_name} Model', fontsize=25)

    # 在图中添加文本框以显示各项指标的值
    text_str = '\n'.join([f"{row['Metric']}: {row['Value']:.2f}" for _, row in summary_df.iterrows()])
    plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes, verticalalignment='top')

    # 添加图例并保存图表为jpeg格式
    plt.legend(loc='lower right', fontsize=25)
    plt.savefig(f"{output_path}/{model_name}_R2.png", format='png', dpi=1000)
    # 显示图表
    plt.show()


def preprocess_data(paths, clip_min=None, clip_max=None):
    vectorized_data = []
    for path in paths:
        # 直接在这里读取TIFF文件
        with rasterio.open(path) as src:
            data = src.read(1)  # 不再需要 meta，所以这里不将其赋值

        # 接下来是处理数据的部分
        if clip_min is not None or clip_max is not None:
            mean_value = np.mean(data)  # 计算平均值

            if clip_min is not None:
                data[data < clip_min] = mean_value  # 替换小于 clip_min 的值
            if clip_max is not None:
                data[data > clip_max] = mean_value  # 替换大于 clip_max 的值

        vectorized_data.append(data.ravel())

    return np.concatenate(vectorized_data)


def preprocess_single_data(folder_path, file_index=0, clip_min=None, clip_max=None):
    vectorized_data = []
    # 使用glob模块查找文件夹中所有的TIFF文件
    paths = glob.glob(os.path.join(folder_path, '*.tif'))

    # 确保 file_index 在合理的范围内
    if file_index < 0 or file_index >= len(paths):
        raise IndexError("file_index is out of the range of the number of files available.")

    # 选择指定索引的文件
    selected_path = paths[file_index]

    # 打开选定的TIFF文件
    with rasterio.open(selected_path) as src:
        data = src.read(1)  # 读取第一个波段，可以根据需要修改波段编号

    # 接下来是处理数据的部分
    if clip_min is not None or clip_max is not None:
        mean_value = np.mean(data)  # 计算平均值

        if clip_min is not None:
            data[data < clip_min] = mean_value  # 替换小于 clip_min 的值
        if clip_max is not None:
            data[data > clip_max] = mean_value  # 替换大于 clip_max 的值

    vectorized_data.append(data.ravel())

    return np.concatenate(vectorized_data)


def model_accuracy(y_test, y_pred, n_parameters, model_name, output_path):
    """
    通过计算几个预测准确性指标来评估模型性能。

    :param output_path: 图片保存路径
    :param model_name: 模型名称
    :param y_test: 类数组, 真实值
    :param y_pred: 类数组, 预测值
    :param n_parameters: int, 模型中参数的数量（用于调整后的R-squared）
    :return: dict, 包含所有评估的指标
    """

    # 平均绝对百分比误差
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # 均方根误差
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # 平均绝对误差
    mae = mean_absolute_error(y_test, y_pred)

    # R-squared（决定系数）
    r2 = r2_score(y_test, y_pred)

    # 调整后的R-squared
    n = len(y_test)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_parameters - 1)

    # 均方误差
    mse = mean_squared_error(y_test, y_pred)

    density_save_path = 'F:/MyProjects/MachineLearning/Data/Final_Data/Density'
    # 传入参数打印图表
    print_model_R2(y_test, y_pred, model_name, output_path, r2, mse, mape, rmse, mae, adj_r2, density_save_path,
                   degree=2, n_jobs=8,)
    plot_mse(y_test, y_pred, model_name, output_path, density_save_path, n_jobs=8)
    plot_mae(y_test, y_pred, model_name, output_path, density_save_path, n_jobs=8)


def compute_kde(segment):
    # 在这个例子中，segment是包含(y_test, y_pred)对的一部分
    xy = np.vstack(segment)
    kde = gaussian_kde(xy)(xy)
    return kde


def parallel_kde(y_test, y_pred, n_jobs):
    # 将数据拆分成n_jobs个部分进行并行处理
    segments = np.array_split(np.vstack([y_test, y_pred]), n_jobs, axis=1)
    pool = Pool(n_jobs)
    results = pool.map(compute_kde, segments)
    # 将结果合并
    return np.concatenate(results)


def save_density_data(z, save_path):
    np.save(save_path, z)


def load_density_data(save_path):
    return np.load(save_path)


def plot_mae(y_true, y_pred, model_name, output_path, density_save_path, n_jobs=8):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    if len(y_true) != len(y_pred):
        raise ValueError("实际值和预测值的长度必须相同。")

    errors = np.abs(y_true - y_pred)

    z = get_density_data(model_name, y_true, y_pred, 'MAE', density_save_path, n_jobs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 散点图
    scatter = ax1.scatter(range(len(errors)), errors, c=z, s=1, edgecolor='none', cmap='RdYlGn_r')
    ax1.set_title(f'{model_name} MAE')
    ax1.set_xlabel('Data Point Index')
    ax1.set_ylabel('Error Squared')
    ax1.grid(True)
    plt.colorbar(scatter, ax=ax1, label='Density')

    # 饼状图
    intervals = [0, 1, 10, np.max(errors)]
    labels = ['0-1', '1-10', f'10-{np.max(errors):.1f}']
    counts = [np.sum((errors >= intervals[i]) & (errors < intervals[i + 1])) for i in range(len(intervals) - 1)]
    ax2.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140,
            colors=['#ff6347', '#4682b4', '#32cd32'])  # 番茄红，钢蓝，酸橙绿
    ax2.set_title('Error Interval Proportion')

    plt.tight_layout()
    plt.savefig(f"{output_path}/{model_name}_MAE.png", format='png', dpi=1000)
    plt.show()


def plot_mse(y_true, y_pred, model_name, output_path, density_save_path, n_jobs=8):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    if len(y_true) != len(y_pred):
        raise ValueError("实际值和预测值的长度必须相同。")

    errors = (y_true - y_pred) ** 2

    # 简化间隔定义，便于与 plot_mae 保持一致
    intervals = [(0, 1), (1, 3), (3, 10), (10, np.max(errors))]
    labels = ['0-1', '1-3', '3-10', f'10-{np.max(errors):.1f}']
    counts = [np.sum((errors >= interval[0]) & (errors < interval[1])) for interval in intervals]

    z = get_density_data(model_name, y_true, y_pred, 'MSE', density_save_path, n_jobs)

    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 散点图
    scatter = ax1.scatter(range(len(errors)), errors, c=z, cmap=cmap, s=1, edgecolor='none')
    ax1.set_title(f'{model_name} MSE')
    ax1.set_xlabel('Data Point Index')
    ax1.set_ylabel('Error Squared')
    ax1.grid(True)
    plt.colorbar(scatter, ax=ax1, label='Density')

    # 饼状图
    ax2.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140,
            colors=['#ffa500', '#ff4500', '#2e8b57', '#1e90ff', '#da70d6', '#ee82ee'])  # 橙色，橘红色，海洋绿，道奇蓝，兰花紫，紫罗兰
    ax2.set_title('Error Interval Proportion')

    plt.tight_layout()
    plt.savefig(f"{output_path}/{model_name}_MSE.png", format='png', dpi=1000)
    plt.show()


def get_density_data(model_name, y_test, y_pred, density_data_id, density_save_path=None, n_jobs=8):
    """
    获取或计算给定模型的密度数据。

    :param model_name: 模型的名称，用于区分不同模型的密度数据。
    :param y_test: 测试集的实际值。
    :param y_pred: 模型对测试集的预测值。
    :param density_data_id: 用于进一步区分同一模型不同情况的密度数据标识符。
    :param density_save_path: 保存密度数据的路径。如果提供，将在这个路径下保存和查找密度数据。
    :param n_jobs: 用于并行计算密度的作业数。
    :return: 密度数据数组。
    """
    density_file_name = f"{model_name}_{density_data_id}_density_data.npy"
    full_density_path = os.path.join(density_save_path, density_file_name)

    # 检查是否已有密度数据文件
    if os.path.exists(full_density_path):
        z = load_density_data(full_density_path)
    else:
        # 计算密度数据并保存
        z = parallel_kde(y_test, y_pred, n_jobs)
        save_density_data(z, full_density_path)

    return z


def simple_gpu_kde(x_values, y_values, bandwidth=0.1):
    """
    使用CuPy在GPU上进行简单的KDE计算，并直接在数据点上计算而非网格上。
    同时提供计算进度的输出。

    参数:
    x_values (numpy.ndarray): X坐标的值。
    y_values (numpy.ndarray): Y坐标的值。
    bandwidth (float): KDE的带宽。

    返回:
    numpy.ndarray: 计算得到的KDE值。
    """

    # 将输入数据转移到GPU
    data = cp.asarray(np.vstack([x_values, y_values]))
    n_points = data.shape[1]

    # 初始化KDE结果数组
    kde_values = cp.zeros(n_points)

    # 对于每个数据点计算KDE并输出进度
    for i in range(n_points):
        diff = data - data[:, i:i+1]
        kde_values[i] = cp.sum(cp.exp(-cp.sum(diff**2, axis=0) / (2 * bandwidth**2)))

        # 每计算一定数量的点后输出一次进度
        if i % 10000 == 0:  # 每1000个点报告一次，这个数字可以根据需要调整
            print(f"Processed {i} of {n_points} points ({100. * i / n_points:.2f}%)")

    kde_values /= (n_points * (2 * cp.pi * bandwidth**2))

    # 将结果转换回NumPy数组以进行可视化
    return cp.asnumpy(kde_values)


# 加载并组合输入数据的函数
def load_and_stack(*paths):
    arrays = []
    for path in paths:
        with rasterio.open(path) as src:
            arrays.append(src.read(1))  # 读取第一个波段
    return np.stack(arrays, axis=-1)


def predict_and_save(model_path, input_paths, output_path, qa_path):
    # 加载模型
    model = joblib.load(model_path)

    # 读取输入数据
    X = load_and_stack(*input_paths)
    X_reshaped = X.reshape(-1, X.shape[-1])  # 重塑为 (n_samples, n_features)

    # 读取 QA 数据
    with rasterio.open(qa_path) as src:
        QA = src.read(1).flatten()  # 确保 QA 数据与 X 的形状一致

    # 应用 QA 数据过滤
    if X_reshaped.shape[0] != QA.size:
        raise ValueError("QA 数据与输入数据尺寸不匹配")
    valid_indices = QA == 1
    X_filtered = X_reshaped[valid_indices]

    # 检查X中的无穷大值
    inf_count = np.isinf(X_filtered).sum()
    if inf_count > 0:
        # 使用 X_reshaped.shape[1] 确保我们获取正确列数
        median_values = np.array(
            [np.median(X_filtered[~np.isinf(X_filtered[:, col]), col]) for col in range(X_filtered.shape[1])])

        # 替换无穷大值
        inf_positions = np.where(np.isinf(X_filtered))
        for pos in zip(*inf_positions):
            # 替换为对应列的中位数
            X_filtered[pos] = median_values[pos[1]]

    y_pred = model.predict(X_filtered)
    # 将预测结果填充回原始形状
    y_pred_full = np.full(QA.shape, np.nan)  # 初始化全 NaN 数组
    y_pred_full[valid_indices] = y_pred  # 将预测结果填充到相应位置
    y_pred_reshaped = y_pred_full.reshape(X.shape[:-1])  # 二维重塑

    # 保存预测结果
    with rasterio.open(input_paths[0]) as src:
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(y_pred_reshaped.astype(rasterio.float32), 1)

    # 读取并显示预测结果
    with rasterio.open(output_path) as src:
        raster = src.read(1)
        fig, ax = plt.subplots()
        cax = ax.imshow(raster, cmap='RdYlGn_r')
        cbar = fig.colorbar(cax, ax=ax)  # 创建色彩条
        plt.savefig(f"{output_path}_preview.png", format='png', bbox_inches='tight', dpi=300)  # 保存图像与色彩条
        plt.show()


def preprocessing_QA_folder(input_folder, output_folder, utm_coords, size):
    for QA_filename in os.listdir(input_folder):
        if QA_filename.lower().endswith('.tif'):
            QA_input_path = os.path.join(input_folder, QA_filename)  # 导入数据
            # 构建数据路径
            output_filename = QA_filename.split('_')[3] + "_QA_Usable.tif"
            output_path = os.path.join(output_folder, output_filename)

            # 检查输出文件是否已存在
            if os.path.exists(output_path):
                print(f"文件 {output_filename} 已存在，跳过处理。")
                continue  # 跳过当前循环的其余部分

            with rasterio.open(QA_input_path) as src:
                row, col = src.index(*utm_coords)
                window = Window(col, row, size[0], size[1])
                cropped_image = src.read(window=window, out_dtype='uint16')  # 窗口裁切，确保数据格式为uint16
                meta = src.meta.copy()
                meta.update({
                    'dtype': 'uint16',  # 确保数据格式为uint16
                    'height': window.height,
                    'width': window.width,
                    'transform': rasterio.windows.transform(window, src.transform)
                })

                # 属于reclassification_values的设置为0，否则设置为1
                reclassification_values = [55052, 54596, 54852, 24472, 24344, 24088, 22280, 23888, 23952, 24216]
                mask = np.isin(cropped_image, reclassification_values)
                cropped_image[mask] = 0
                cropped_image[~mask] = 1

                # 保存处理后的图像
                with rasterio.open(output_path, 'w', **meta) as dst:
                    dst.write(cropped_image)

    print(f"QA波段数据处理完成并保存至 {output_folder}")
