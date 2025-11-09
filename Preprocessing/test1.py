import os
import re
import numpy as np
import rasterio
from scipy.stats import pearsonr

# 输入文件夹和土地覆盖类型文件
lst_folder = 'F:/MyProjects/MachineLearning/Data/Usable_Data/LST_30m'
ndvi_folder = 'F:/MyProjects/MachineLearning/Data/Usable_Data/NDVI_30m'
land_cover_path = 'F:/MyProjects/MachineLearning/Data/Usable_Data/Type_30m/anhuiwgs84_1.tif'

# 读取土地覆盖类型数据
with rasterio.open(land_cover_path) as src:
    land_cover = src.read(1)

# 定义土地覆盖类型
land_cover_types = [1, 2, 4, 5, 7, 8]

# 初始化存储相关性系数的矩阵
correlation_matrix = np.zeros((len(land_cover_types), 12))

# 获取文件列表
lst_files = sorted([f for f in os.listdir(lst_folder) if f.endswith('.tif')],
                   key=lambda x: int(re.search(r'_([0-9]{8})_', x).group(1)[4:6]))
ndvi_files = sorted([f for f in os.listdir(ndvi_folder) if f.endswith('.tif')],
                    key=lambda x: int(re.search(r'_([0-9]{8})\.', x).group(1)[4:6]))

# 遍历每个月份
for month in range(12):
    # 读取LST和NDVI数据
    lst_path = os.path.join(lst_folder, lst_files[month])
    ndvi_path = os.path.join(ndvi_folder, ndvi_files[month])

    with rasterio.open(lst_path) as src:
        lst_data = src.read(1)

    with rasterio.open(ndvi_path) as src:
        ndvi_data = src.read(1)

    # 初始化6个变量存储LST和NDVI数据
    lst_data_by_type = {lt: [] for lt in land_cover_types}
    ndvi_data_by_type = {lt: [] for lt in land_cover_types}

    # 根据土地利用类型索引存储LST和NDVI数据
    rows, cols = lst_data.shape
    for r in range(rows):
        for c in range(cols):
            land_type = land_cover[r, c]
            if land_type in land_cover_types:
                lst_value = lst_data[r, c]
                ndvi_value = ndvi_data[r, c]
                if not np.isnan(lst_value) and not np.isnan(ndvi_value):
                    lst_data_by_type[land_type].append(lst_value)
                    ndvi_data_by_type[land_type].append(ndvi_value)

    # 计算相关性系数
    for i, land_type in enumerate(land_cover_types):
        if lst_data_by_type[land_type] and ndvi_data_by_type[land_type]:
            # 将数据转换为numpy数组，并过滤NaN和Inf值
            lst_array = np.array(lst_data_by_type[land_type])
            ndvi_array = np.array(ndvi_data_by_type[land_type])
            valid_mask = ~np.isnan(lst_array) & ~np.isnan(ndvi_array) & np.isfinite(lst_array) & np.isfinite(ndvi_array)
            lst_clean = lst_array[valid_mask]
            ndvi_clean = ndvi_array[valid_mask]

            if len(lst_clean) > 0 and len(ndvi_clean) > 0:
                correlation = pearsonr(lst_clean, ndvi_clean)[0]
                correlation_matrix[i, month] = correlation
            else:
                correlation_matrix[i, month] = np.nan
        else:
            correlation_matrix[i, month] = np.nan

# 打印相关性系数矩阵
print(correlation_matrix)
