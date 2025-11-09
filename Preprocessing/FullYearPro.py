import os
import glob
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from osgeo import gdal, osr
import shutil
from datetime import datetime
import joblib

# 文件夹路径
model_folder = r"F:\MyProjects\MachineLearning\Data\Final_Data\Models\2022"
modis_data_folder = r"F:\MyProjects\MachineLearning\Data\FullYearPre\Row"
temp_data_folder = r"F:\MyProjects\MachineLearning\Data\FullYearPre\TempData"
lst_temp1_folder = os.path.join(temp_data_folder, "LST_Temp1")
qc_temp1_folder = os.path.join(temp_data_folder, "QC_Temp1")
lst_temp2_folder = os.path.join(temp_data_folder, "LST_Temp2")
qc_temp2_folder = os.path.join(temp_data_folder, "QC_Temp2")
lst_mouth_folder = os.path.join(temp_data_folder, "LSTmouth")
qc_mouth_folder = os.path.join(temp_data_folder, "QCmouth")
temp_folder = os.path.join(temp_data_folder, "Temp")

# 已创建的文件夹无需再创建
os.makedirs(lst_mouth_folder, exist_ok=True)
os.makedirs(qc_mouth_folder, exist_ok=True)
os.makedirs(temp_folder, exist_ok=True)


# 提取LST和QC波段并重命名
def extract_bands():
    for file in glob.glob(os.path.join(modis_data_folder, "*.hdf")):
        basename = os.path.basename(file)
        parts = basename.split('.')
        date_str = parts[1][1:]
        tile_str = parts[2]

        lst_band = f'LST_Day_1km'
        qc_band = f'QC_Day'

        lst_output = os.path.join(lst_temp1_folder, f"LST{date_str}{tile_str}.tif")
        qc_output = os.path.join(qc_temp1_folder, f"QC{date_str}{tile_str}.tif")

        print(f"提取LST波段: {lst_output}")
        gdal.Translate(lst_output, f'HDF4_EOS:EOS_GRID:"{file}":MODIS_Grid_Daily_1km_LST:{lst_band}')

        print(f"提取QC波段: {qc_output}")
        gdal.Translate(qc_output, f'HDF4_EOS:EOS_GRID:"{file}":MODIS_Grid_Daily_1km_LST:{qc_band}')


# 计算LST波段
def calculate_lst_band():
    for file in glob.glob(os.path.join(lst_temp1_folder, "*.tif")):
        print(f"计算LST波段: {file}")
        with rasterio.open(file, 'r+') as src:
            data = src.read(1) * 0.02
            src.write(data, 1)


# 拼接LST和QC数据
def merge_tiles(input_folder, output_folder, prefix):
    for date_str in set([os.path.basename(f)[3:10] for f in glob.glob(os.path.join(input_folder, "*.tif"))]):
        tile_files = glob.glob(os.path.join(input_folder, f"{prefix}{date_str}*.tif"))
        if len(tile_files) == 0:
            print(f"没有找到可用的源文件进行拼接: {date_str}")
            continue

        temp_output_file = os.path.join(temp_folder, f"{prefix}{date_str}.tif")
        final_output_file = os.path.join(output_folder, f"{prefix}{date_str}.tif")

        print(f"拼接文件: {tile_files}")

        gdal.Warp(temp_output_file, tile_files)

        if os.path.exists(temp_output_file):
            print(f"临时文件已生成: {temp_output_file}")
            shutil.move(temp_output_file, final_output_file)
        else:
            print(f"未找到临时文件: {temp_output_file}")


# 拼接QC数据时需使用不同的命名规则
def merge_qc_tiles(input_folder, output_folder, prefix):
    for date_str in set([os.path.basename(f)[2:10] for f in glob.glob(os.path.join(input_folder, "*.tif"))]):
        tile_files = glob.glob(os.path.join(input_folder, f"{prefix}{date_str}*.tif"))
        if len(tile_files) == 0:
            print(f"没有找到可用的源文件进行拼接: {date_str}")
            continue

        temp_output_file = os.path.join(temp_folder, f"{prefix}{date_str}.tif")
        final_output_file = os.path.join(output_folder, f"{prefix}{date_str}.tif")

        print(f"拼接文件: {tile_files}")

        gdal.Warp(temp_output_file, tile_files)

        if os.path.exists(temp_output_file):
            print(f"临时文件已生成: {temp_output_file}")
            shutil.move(temp_output_file, final_output_file)
        else:
            print(f"未找到临时文件: {temp_output_file}")


# 重投影
def reproject_rasters(folder):
    for file in glob.glob(os.path.join(folder, "*.tif")):
        src_ds = gdal.Open(file)
        dst_proj = osr.SpatialReference()
        dst_proj.SetWellKnownGeogCS("WGS84")
        dst_proj.SetUTM(50, True)

        temp_output_file = os.path.join(temp_folder, os.path.basename(file))
        print(f"重投影文件: {file} 到 {temp_output_file}")
        gdal.Warp(temp_output_file, src_ds, dstSRS=dst_proj.ExportToWkt(), xRes=990, yRes=990)

        if os.path.exists(temp_output_file):
            shutil.move(temp_output_file, file)
        else:
            print(f"未找到临时文件: {temp_output_file}")


# 切割
def crop_raster(input_folder, utm_coords, size):
    files = glob.glob(os.path.join(input_folder, '*.tif'))
    for file in files:
        with rasterio.open(file) as src:
            row, col = src.index(*utm_coords)
            window = Window(col, row, size[0], size[1])
            cropped_image = src.read(window=window)
            meta = src.meta
            meta.update({
                'driver': 'GTiff',
                'height': window.height,
                'width': window.width,
                'transform': rasterio.windows.transform(window, src.transform)
            })

        temp_output_path = os.path.join(temp_folder, os.path.basename(file))
        final_output_path = os.path.join(input_folder, os.path.basename(file))
        print(f"裁切文件: {file} 到 {temp_output_path}")
        with rasterio.open(temp_output_path, 'w', **meta) as dst:
            dst.write(cropped_image)

        if os.path.exists(temp_output_path):
            shutil.move(temp_output_path, final_output_path)
        else:
            print(f"未找到临时文件: {temp_output_path}")


# 重采样
def resample_lst_to_30m(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            input_path = os.path.join(input_folder, filename)
            temp_output_path = os.path.join(temp_folder, filename)
            final_output_path = os.path.join(output_folder, filename)
            with rasterio.open(input_path) as src:
                new_transform = src.transform * src.transform.scale(
                    (src.width / 2343),
                    (src.height / 2112)
                )
                new_width = 2343
                new_height = 2112
                resampled_data = src.read(
                    1,
                    out_shape=(new_height, new_width),
                    resampling=Resampling.nearest
                )
                print(f"重采样文件: {input_path} 到 {temp_output_path}")
                with rasterio.open(
                        temp_output_path, 'w',
                        driver='GTiff',
                        height=new_height,
                        width=new_width,
                        count=1,
                        dtype=resampled_data.dtype,
                        crs=src.crs,
                        transform=new_transform
                ) as dst:
                    dst.write(resampled_data, 1)

            if os.path.exists(temp_output_path):
                shutil.move(temp_output_path, final_output_path)
            else:
                print(f"未找到临时文件: {temp_output_path}")


# 将数据按月份分类
def rename_files():
    def rename_file(file, prefix):
        # 提取文件名中的日期编号部分
        basename = os.path.basename(file)
        if prefix == 'QC':
            date_number = basename.split('.')[0][len(prefix):-1]  # 删除最后的h字符
        else:
            date_number = basename.split('.')[0][len(prefix):]

        year = date_number[:4]
        day_of_year = int(date_number[4:])

        # 将日期编号转为年月日
        date = datetime.strptime(f"{year}{day_of_year:03}", "%Y%j").strftime("%Y%m%d")

        # 重命名文件
        new_date = f"{date[:4]}{date[4:6]}{date[6:]}"
        new_file_name = f"{prefix}{new_date}.tif"

        new_file_path = os.path.join(os.path.dirname(file), new_file_name)
        return new_file_path

    # 处理LST文件
    for file in glob.glob(os.path.join(lst_temp2_folder, "LST*.tif")):
        new_file_path = rename_file(file, 'LST')
        os.rename(file, new_file_path)

    # 处理QC文件
    for file in glob.glob(os.path.join(qc_temp2_folder, "QC*.tif")):
        new_file_path = rename_file(file, 'QC')
        os.rename(file, new_file_path)


def organize_by_month():
    # 创建月份文件夹
    for i in range(1, 13):
        month_folder_lst = os.path.join(lst_mouth_folder, f"{i:02}")
        month_folder_qc = os.path.join(qc_mouth_folder, f"{i:02}")
        os.makedirs(month_folder_lst, exist_ok=True)
        os.makedirs(month_folder_qc, exist_ok=True)

    # 按月份分类LST文件
    for file in glob.glob(os.path.join(lst_temp2_folder, "LST*.tif")):
        basename = os.path.basename(file)
        month = basename[7:9]  # 获取月份
        month_folder_lst = os.path.join(lst_mouth_folder, month)
        dest_file = os.path.join(month_folder_lst, basename)
        if os.path.exists(dest_file):
            os.remove(dest_file)
        os.rename(file, dest_file)

    # 按月份分类QC文件
    for file in glob.glob(os.path.join(qc_temp2_folder, "QC*.tif")):
        basename = os.path.basename(file)
        month = basename[6:8]  # 获取月份
        month_folder_qc = os.path.join(qc_mouth_folder, month)
        dest_file = os.path.join(month_folder_qc, basename)
        if os.path.exists(dest_file):
            os.remove(dest_file)
        os.rename(file, dest_file)


# 执行过程
extract_bands()
calculate_lst_band()
merge_tiles(lst_temp1_folder, lst_temp2_folder, "LST")
merge_qc_tiles(qc_temp1_folder, qc_temp2_folder, "QC")
reproject_rasters(lst_temp2_folder)
reproject_rasters(qc_temp2_folder)
crop_raster(lst_temp2_folder, (505509, 3554444), (71, 64))
crop_raster(qc_temp2_folder, (505509, 3554444), (71, 64))
resample_lst_to_30m(lst_temp2_folder, lst_temp2_folder)
resample_lst_to_30m(qc_temp2_folder, qc_temp2_folder)
rename_files()
organize_by_month()

print("完成所有操作")
