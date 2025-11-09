import numpy as np
import pandas as pd
import rasterio
import plotly.express as px

# 假设数据存储在以下路径
high_res_lst_path = ('F:/MyProjects/MachineLearning/Data/Usable_Data/LST_30m'
                     '/LC08_L2SP_121038_20220616_20220411_02_T1_ST_B10_processed.tif')  # 高分辨率LST数据路径
low_res_lst_path = 'F:/MyProjects/MachineLearning/Data/Usable_Data/LST_30m_990m/LST_Day_1km_20220616.tif'  # 低分辨率LST数据路径
ndvi_path = 'F:/MyProjects/MachineLearning/Data/Usable_Data/NDVI_30m/NDVI_20220616.tif'  # NDVI数据路径
land_use_type_path = 'F:/MyProjects/MachineLearning/Data/Usable_Data/Type_30m/anhuiwgs84_6.tif'  # 土地利用类型数据路径

# 定义一个函数来读取tif文件并将其展平为一维数组
def read_and_flatten_tif(file_path):
    with rasterio.open(file_path) as src:
        array = src.read(1)  # 读取第一个波段
    return array.flatten()

# 加载数据并展平为一维数组
high_res_lst_flat = read_and_flatten_tif(high_res_lst_path)  # 加载并展平高分辨率LST数据
low_res_lst_flat = read_and_flatten_tif(low_res_lst_path)  # 加载并展平低分辨率LST数据
ndvi_flat = read_and_flatten_tif(ndvi_path)  # 加载并展平NDVI数据
land_use_type_flat = read_and_flatten_tif(land_use_type_path)  # 加载并展平土地利用类型数据

# 创建DataFrame
data = {
    'LST30M': high_res_lst_flat,  # 高分辨率LST
    'LST990M': low_res_lst_flat,  # 低分辨率LST
    'NDVI30M': ndvi_flat,  # NDVI
    'LCD30M': land_use_type_flat  # 土地利用类型
}

df = pd.DataFrame(data).dropna()  # 创建DataFrame

# 计算相关矩阵
correlation_matrix = df.corr()  # 计算相关矩阵

# 使用 Plotly 绘制热图
fig = px.imshow(correlation_matrix, text_auto=True, color_continuous_scale='viridis')
fig.update_layout(title='Correlation Analysis', width=800, height=600)
fig.show()
