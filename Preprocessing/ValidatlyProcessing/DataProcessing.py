from LST_Tools import Tool1

# lst_process_folder = 'F:/MyProjects/MachineLearning/Data/Validation_Data/Row/Temp990LST'  # 设置LST文件，将存储提取、拼接，重投影后的LST文件
# lst_usable_folder = 'F:/MyProjects/MachineLearning/Data/Validation_Data/Usable/LST990'  # 设置最后可用的990m级别LST存储文件夹
# lst_resample_folder = 'F:/MyProjects/MachineLearning/Data/Validation_Data/Usable/LST30990'  # 设置重采样的LST_30m数据文件夹
# utm_coords = (632604, 3554049)  # 裁切起始左上角UTM坐标 632604.3776,3554049.7737
# size = (71, 64)  # 要裁切的区域的大小，以像素为单位

# 5. 使用crop_raster函数对重投影后的lst_output_folder数据进行裁切
# Tool1.crop_raster(lst_process_folder, lst_usable_folder, utm_coords, size)
# Tool1.resample_lst_to_30m(lst_usable_folder, lst_resample_folder)


input_path = 'F:/MyProjects/MachineLearning/Data/Validation_Data/Row/LCD'  # 土地利用类型数据的路径
output_path = 'F:/MyProjects/MachineLearning/Data/Validation_Data/Usable/LCD30'
utm_coords = (631920, 3554444)  # 裁切起始左上角UTM坐标 631920.4614,3570251.0962
size = (2343, 2112)  # 要裁切的区域的大小，以像素为单位

Tool1.crop_type_data(input_path, output_path, utm_coords, size)
