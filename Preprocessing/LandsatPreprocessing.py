from LST_Tools import Tool1

Landsat_B4B5_path = 'F:/MyProjects/MachineLearning/Data/Raw_Data/LandsatB4B5'  # 设置输入文件夹，这个文件夹存放Landsat的相对应日期的B4和B5波段数据
Landsat_B10_path = 'F:/MyProjects/MachineLearning/Data/Raw_Data/LandsatB10'  # 设置输入文件夹，这个文件夹存放Landsat的相对应日期的B10波段数据
NDVI_data_path = 'F:/MyProjects/MachineLearning/Data/Process_Data/NDVI'  # 设置输出文件夹，这个文件中会存放对应日期的NDVI数据
NDVI_usable_data_path = 'F:/MyProjects/MachineLearning/Data/Usable_Data/NDVI_30m'  # 设置裁切后的30m级NDVI文件输出文件夹
LST_30_path = 'F:/MyProjects/MachineLearning/Data/Usable_Data/LST_30m'  # 设置最终可用的30m地表温度数据文件夹
utm_coords = (505300, 3555000)  # 裁切起始左上角UTM坐标
size = (2343, 2112)  # 要裁切的区域的大小，以像素为单位

# 1.对输入文件夹进行整理，将同一日的LandsatB4B5数据创建并放在同一日期文件夹下
Tool1.group_landsat_files(Landsat_B4B5_path)

# 2.访问Landsat_B4B5_path文件夹，计算其文件夹中的对应日期的NDVI数据并保存至输出文件夹
Tool1.process_all_landsat(Landsat_B4B5_path, NDVI_data_path)

# 3.访问NDVI_data_path对文件夹中的数据逐个进行裁切，裁切以起始坐标和像元长宽为基准，保证数据在地理空间上的匹配
Tool1.crop_raster(NDVI_data_path, NDVI_usable_data_path, utm_coords, size)

#  4.访问Landsat_B10_path，对数据进行基本计算和裁切
Tool1.preprocessing_B10folder(Landsat_B10_path, LST_30_path, utm_coords, size)

#  5.访问输出文件夹,绘制出其中的第i张以供检查
Tool1.plot_tif(NDVI_usable_data_path, 0)
Tool1.plot_tif(LST_30_path, 0)
