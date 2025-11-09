import os.path
import matplotlib.pyplot as plt

"""
根据数据绘制每月云盖率和地表温度曲线图
"""

# 文件路径
input_file = 'F:/MyProjects/MachineLearning/Data/Final_Data/Pictures/PerformanceMetrics/Cloud_Coverage_and_LST_Mean.txt'
output_folder = 'F:/MyProjects/MachineLearning/Data/Final_Data/Pictures/PerformanceMetrics/2022Cloud_LST'

# 初始化列表
dates = []
cloud_coverages = []
lst_means = []

# 读取文件
with open(input_file, 'r') as file:
    next(file)  # 跳过标题行
    for line in file:
        date, cloud_coverage, lst_mean = line.strip().split(', ')
        dates.append(date)
        cloud_coverages.append(float(cloud_coverage))
        lst_means.append(float(lst_mean))

# 遍历每个月份，提取数据并绘图
for month in range(1, 13):
    month_str = f'{month:02d}'

    # 初始化当前月份的数据列表
    month_dates = []
    month_days = []
    month_cloud_coverages = []
    month_lst_means = []

    # 提取当前月份的数据
    for date, coverage, mean in zip(dates, cloud_coverages, lst_means):
        if date[4:6] == month_str:  # 检查日期字符串中的月份部分
            month_dates.append(date)
            month_days.append(date[6:8])  # 提取日期的最后两位
            month_cloud_coverages.append(coverage)
            month_lst_means.append(mean)

    # 如果当前月份没有数据，跳过绘图
    if not month_dates:
        continue

    # 创建折线图
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制温度折线图
    ax1.plot(month_days, month_lst_means, 'r-', label='LST Mean (°C)')
    ax1.set_xlabel(f'Day in {month_str}')
    ax1.set_ylabel('LST Mean (°C)', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.set_title(f'{month_str} 2022: LST Mean and Cloud Coverage')

    # 添加第二个Y轴，用于绘制云覆盖率折线图
    ax2 = ax1.twinx()
    ax2.plot(month_days, month_cloud_coverages, 'b-', label='Cloud Coverage (%)')
    ax2.set_ylabel('Cloud Coverage (%)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    # 显示图例
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    ax1.grid(True)

    # 调整布局并保存图表
    fig.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{month_str}_LST_Mean_and_Cloud_Coverage.png'), dpi=1000)
    plt.close()

    print(f"Line plot has been saved as '{month_str}_LST_Mean_and_Cloud_Coverage.png'")
