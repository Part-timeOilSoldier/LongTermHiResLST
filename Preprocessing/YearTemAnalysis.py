import os
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from meteostat import Daily, Stations
from scipy.stats import linregress

# 文件夹路径
folder_path = 'F:/MyProjects/MachineLearning/Data/FullYearPre/FinalData'  # 替换为你的文件夹路径

# 查找最近的气象站
stations = Stations()
stations = stations.nearby(31.8639, 117.2808)  # 合肥的经纬度
station = stations.fetch(1)  # 获取最近的站点

# 设置时间范围
start = datetime(2022, 1, 1)
end = datetime(2022, 12, 31)

# 获取气象数据
weather_data = Daily(station, start, end)
weather_data = weather_data.fetch()

# 初始化地表温度数据字典
lst_data = {}

# 遍历文件夹中的tif文件
for file in os.listdir(folder_path):
    if file.endswith('.tif') and 'Predict_LST_LST2022' in file:
        date = datetime.strptime(file[-12:-4], '%Y%m%d')  # 提取日期并转换为datetime对象
        with rasterio.open(os.path.join(folder_path, file)) as src:
            array = src.read(1)
            valid_data = array[array != src.nodata]  # 排除无数据值
            valid_data = valid_data[np.isfinite(valid_data)]  # 过滤非数值和无穷大值
            if valid_data.size > 0:
                mean_temp = valid_data.mean() - 273.15  # 计算平均值并转换为摄氏度
                lst_data[date] = mean_temp

# 创建DataFrame
df = pd.DataFrame(index=pd.date_range(start, end))
df.index.name = 'Date'
df['AT'] = weather_data['tavg']
df['LST'] = pd.Series(lst_data, index=lst_data.keys()).astype(float)  # 将日期作为索引添加LST数据

# 打印表格
print(df)

# 设置字体大小
plt.rcParams.update({'font.size': 14})

# 绘图
plt.figure(figsize=(16, 6), dpi=500)
plt.plot(df.index, df['AT'], label='Average Temperature', color='green')
plt.plot(df.index, df['LST'], label='Average Land Surface Temperature', color='blue')
plt.xlabel('Day', fontsize=16)  # 增大x轴说明的字体
plt.ylabel('Temperature (°C)', fontsize=16)  # 增大y轴说明的字体
plt.legend(fontsize=14)  # 增大图例的字体
plt.grid(True)

# 添加垂直虚线
highlight_dates = ['01/15', '02/24', '03/12', '04/24', '05/24', '06/16', '07/02', '08/14', '09/19', '10/25', '11/06', '12/24']
highlight_dates = [datetime.strptime('2022/' + date, '%Y/%m/%d') for date in highlight_dates]
for date in highlight_dates:
    plt.axvline(date, color='red', linestyle='--', linewidth=1, label='Date of NDVI Image Capture')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=14)  # 增大图例的字体

plt.savefig('F:/论文/论文基础/图片/2022LSTandAverageTemperature.png', dpi=1000)
plt.show()



# 其他代码保持不变，直到绘制散点图后

# Remove rows with NaN values for clean fitting
clean_df = df.dropna(subset=['AT', 'LST'])

# Fit a linear model using scipy.stats.linregress for detailed regression results
regression_result = linregress(clean_df['AT'], clean_df['LST'])
slope = regression_result.slope
intercept = regression_result.intercept
r_value = regression_result.rvalue  # This is the correlation coefficient
r_squared = r_value**2  # Calculate R-squared value

# Create a line of best fit for plotting
x_values = np.array([clean_df['AT'].min(), clean_df['AT'].max()])
y_values = slope * x_values + intercept

# Scatter plot
plt.figure(figsize=(16, 6), dpi=500)
plt.scatter(clean_df['AT'], clean_df['LST'], color='green', alpha=0.5, label='Data Points')

# Highlight NDVI Acquisition Dates
valid_highlight_dates = [date for date in highlight_dates if date in clean_df.index]
highlight_df = clean_df.loc[valid_highlight_dates]
plt.scatter(highlight_df['AT'], highlight_df['LST'], color='red', label='Date of NDVI Image Capture')

# Add the straight line of best fit with label including line break for R²
plt.plot(x_values, y_values, color='red', label=f'Linear Fit:y = {slope:.2f}x + {intercept:.2f}\nR² = {r_squared:.4f}')

# Add labels, legend, and title
plt.xlabel('Average Temperature (°C)', fontsize=16)  # 増大x轴说明的字体
plt.ylabel('Land Surface Temperature (°C)', fontsize=16)  # 増大y轴说明的字体
plt.legend(fontsize=14)  # 増大图例的字体
plt.grid(True)

plt.savefig('F:/论文/论文基础/图片/2022Scatter.png', dpi=1000)
plt.show()
