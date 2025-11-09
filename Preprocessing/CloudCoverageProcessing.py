from LST_Tools import Tool1

Landsat_QA_path = 'F:/MyProjects/MachineLearning/Data/Raw_Data/LandsatQA'
QA_30_path = 'F:/MyProjects/MachineLearning/Data/Usable_Data/QA_30m'
utm_coords = (505300, 3555000)  # 裁切起始左上角UTM坐标
size = (2343, 2112)  # 要裁切的区域的大小，以像素为单位

Tool1.preprocessing_QA_folder(Landsat_QA_path, QA_30_path, utm_coords, size)
