import os
import glob
import numpy as np
import pandas as pd
import rasterio
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

# 输入文件夹路径
qc_data_folder = 'F:/MyProjects/MachineLearning/Data/Usable_Data/QA_30m'

# 创建输出文件夹（如果不存在）
output_folder = 'F:/MyProjects/MachineLearning/Data/Final_Data/Pictures/PerformanceMetrics'
os.makedirs(output_folder, exist_ok=True)

# 准备存储结果的字典
zero_percentages = []

# 每月的R²值
r_squared_values = {
    '01': 0.79724,
    '02': 0.73527,
    '03': 0.84932,
    '04': 0.84532,
    '05': 0.80462,
    '06': 0.85560,
    '07': 0.78118,
    '08': 0.65446,
    '09': 0.74097,
    '10': 0.80872,
    '11': 0.80972,
    '12': 0.71674
}

# 获取所有QC数据文件
qc_files = sorted(glob.glob(os.path.join(qc_data_folder, '*_QA_Usable.tif')))

# 遍历每个月的QC文件
for qc_file in qc_files:
    with rasterio.open(qc_file) as src:
        qc_data = src.read(1)
        total_pixels = qc_data.size
        zero_count = np.sum(qc_data == 0)
        zero_percentage = (zero_count / total_pixels) * 100
        zero_percentages.append(zero_percentage)

# 保存结果到txt文件
output_txt_file = os.path.join(output_folder, 'zero_percentages.txt')
with open(output_txt_file, 'w') as f:
    f.write('Month, Zero Percentage (%), R2\n')
    for qc_file, percentage in zip(qc_files, zero_percentages):
        month = os.path.basename(qc_file)[4:6]
        r_squared = r_squared_values[month]
        f.write(f'{month}, {percentage:.2f}, {r_squared:.5f}\n')

# 创建数据框
data = {
    'Month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
    'Zero Percentage (%)': [0.00, 1.99, 0.45, 0.00, 0.00, 0.00, 0.40, 12.28, 0.00, 0.00, 0.00, 0.00],
    'R²': [0.79724, 0.73527, 0.84932, 0.84532, 0.80462, 0.85560, 0.78118, 0.65446, 0.74097, 0.80872, 0.80972, 0.71674]
}

df = pd.DataFrame(data)

# 提取自变量和因变量
X = df[['Zero Percentage (%)']]
y = df['R²']

# 创建线性回归模型并拟合
model = LinearRegression()
model.fit(X, y)

# 预测值
y_pred = model.predict(X)

# 绘制散点图和线性回归直线
plt.figure(figsize=(10, 6))
plt.scatter(df['Zero Percentage (%)'], df['R²'], edgecolors='blue', facecolors='lightblue', label='Point Data')
plt.plot(df['Zero Percentage (%)'], y_pred, color='red', label=f'Regression Line\ny = {model.coef_[0]:.5f}x + {model.intercept_:.5f}\n$R^2$ = {model.score(X, y):.5f}')
plt.xlabel('Cloudiness (%)', fontsize=18, fontname='Times New Roman')
plt.ylabel('R²', fontsize=18, fontname='Times New Roman')
plt.xticks(fontsize=20, fontname='Times New Roman')
plt.yticks(fontsize=20, fontname='Times New Roman')
plt.legend(fontsize=14, loc='best', frameon=True)
plt.grid(True)

# 保存图片
plt.savefig('F:/论文/论文基础/图片/linear_regression_fit.png', dpi=1000)
plt.show()
