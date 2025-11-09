import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import matplotlib.dates as mdates

"""
此脚本用于统计2022年每日云盖率和地表温度
"""

# 输入文件夹路径
qc_data_folder = 'F:/MyProjects/MachineLearning/Data/FullYearPre/TempData/QCmouth'
lst_data_folder = 'F:/MyProjects/MachineLearning/Data/FullYearPre/FinalData'
output_folder = 'F:/MyProjects/MachineLearning/Data/Final_Data/Pictures/PerformanceMetrics'

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 准备存储结果的列表
cloud_coverages = []
lst_means = []
lst_dates = []

# 获取所有LST数据文件
lst_files = sorted(glob.glob(os.path.join(lst_data_folder, 'Predict_LST_LST*.tif')))

# 处理每个月的QC数据文件
for month in range(1, 13):
    month_str = f'{month:02d}'

    # 获取QC数据文件夹路径
    qc_month_folder = os.path.join(qc_data_folder, month_str)

    # 获取当月所有QC数据文件
    qc_files = sorted(glob.glob(os.path.join(qc_month_folder, '*.tif')))

    # 遍历每一天的QC文件
    for day in range(1, 32):
        date_str = f'2022{month_str}{day:02d}'
        qc_file = os.path.join(qc_month_folder, f'QC{date_str}.tif')
        lst_file = os.path.join(lst_data_folder, f'Predict_LST_LST{date_str}.tif')

        # 检查文件是否存在
        if os.path.exists(qc_file) and os.path.exists(lst_file):
            # 读取LST数据并计算平均值
            with rasterio.open(lst_file) as src:
                lst_data = src.read(1)
                total_pixels = lst_data.size

                # 将 nodata 值替换为 np.nan 以便计算有效数据的平均值
                lst_data = np.where(lst_data == src.nodata, np.nan, lst_data)

                # 计算有效数据的平均值
                valid_count = np.count_nonzero(~np.isnan(lst_data))
                if valid_count > 0:
                    lst_mean = np.nanmean(lst_data)
                else:
                    lst_mean = np.nan

                lst_means.append(lst_mean)
                lst_dates.append(date_str)

            # 读取QC数据并计算无效像元值百分比
            with rasterio.open(qc_file) as src:
                qc_data = src.read(1)
                invalid_data_count = np.sum(qc_data == 2)
                invalid_data_percentage = (invalid_data_count / total_pixels) * 100
                cloud_coverage = 100 - invalid_data_percentage
                cloud_coverages.append(cloud_coverage)
        else:
            # 如果文件缺失，则使用None填充
            lst_means.append(None)
            cloud_coverages.append(None)
            lst_dates.append(date_str)

# 创建一个新的图表，包含两个Y轴
fig, ax1 = plt.subplots(figsize=(12, 6))

# 绘制云覆盖率
ax1.plot(lst_dates, cloud_coverages, 'b-', label='Cloud Coverage')
ax1.set_xlabel('Days')
ax1.set_ylabel('Cloud Coverage (%)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_title('2022 Cloud Coverage and LST Mean')
ax1.legend(loc='upper left')

# 添加第二个Y轴，用于绘制LST平均值
ax2 = ax1.twinx()
ax2.plot(lst_dates, lst_means, 'r-', label='LST Mean')
ax2.set_ylabel('LST Mean (°C)', color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.legend(loc='upper right')

# 设置X轴日期格式
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

# 显示网格和旋转X轴刻度标签
fig.autofmt_xdate()
ax1.grid(True)

# 调整布局并保存图表
fig.tight_layout()
plt.savefig(os.path.join(output_folder, 'Cloud_Coverage_and_LST_Mean.png'), dpi=1000)
plt.close()

# 保存云覆盖率和LST平均值到txt文件
output_txt_file = os.path.join(output_folder, 'Cloud_Coverage_and_LST_Mean.txt')
with open(output_txt_file, 'w') as f:
    f.write('Date, Cloud Coverage (%), LST Mean (°C)\n')
    for date, coverage, mean in zip(lst_dates, cloud_coverages, lst_means):
        if coverage is not None and mean is not None:
            f.write(f'{date}, {coverage:.2f}, {mean:.2f}\n')
        else:
            f.write(f'{date}, NaN, NaN\n')





