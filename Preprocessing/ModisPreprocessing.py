from LST_Tools import Tool1


hdf_folder_path = 'F:/MyProjects/MachineLearning/Data/Raw_Data/MOD11A1'  # 设置HDF文件输入路径，这个文件夹存储着所有待提取的HDF文件
lst_process_folder = 'F:/MyProjects/MachineLearning/Data/Process_Data/LST_temp_990m'  # 设置LST文件，将存储提取、拼接，重投影后的LST文件
lst_usable_folder = 'F:/MyProjects/MachineLearning/Data/Process_Data/LST_990m'  # 设置最后可用的990m级别LST存储文件夹
lst_resample_folder = 'F:/MyProjects/MachineLearning/Data/Usable_Data/LST_30m_990m'  # 设置重采样的LST_30m数据文件夹
utm_coords = (505509, 3554444)  # 裁切起始左上角UTM坐标
size = (71, 64)  # 要裁切的区域的大小，以像素为单位

# 1. 使用process_hdf_files函数提取LST数据
Tool1.process_hdf_files(hdf_folder_path, lst_process_folder)

# 2. 使用process_modis_lst_data函数处理LST数据，即乘以像元系数0.02得到单位开尔文
Tool1.process_modis_lst_data(lst_process_folder)

# 3. 使用merge_lst_tiff_by_date函数对同日LST数据进行拼接
Tool1.merge_lst_tiff_by_date(lst_process_folder)

# 4. 使用reproject_folder函数将LST投影与Landsat8数据对齐
Tool1.reproject_folder(lst_process_folder)

# 5. 使用crop_raster函数对重投影后的lst_output_folder数据进行裁切
Tool1.crop_raster(lst_process_folder, lst_usable_folder, utm_coords, size)

# 6. 使用clean_tiff_data函数将0值大于0.02的数据剔除
Tool1.clean_tiff_data(lst_usable_folder)

# 7.使用resample_lst_to_30m函数将处理好的LST_990m重采样成30m
Tool1.resample_lst_to_30m(lst_usable_folder, lst_resample_folder)

# 8. 打印告知脚本运行结束
print("ModisPreprocessing.py所有步骤完成。")
