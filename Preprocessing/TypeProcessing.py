from LST_Tools import Tool1


# TIFF文件的路径
input_path = 'F:/MyProjects/MachineLearning/Data/Raw_Data/Type2022'  # 土地利用类型数据的路径
output_path = 'F:/MyProjects/MachineLearning/Data/Usable_Data/Type_30m'
utm_coords = (505300, 3555000)  # 特定的UTM坐标
size = (2343, 2112)  # 要裁切的区域的大小，以像素为单位

Tool1.crop_type_data(input_path, output_path, utm_coords, size)
