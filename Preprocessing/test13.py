import matplotlib.pyplot as plt

# 读取数据
output_folder = 'F:/论文/论文基础/超大字体图片/chazhi.png'  # 更改为png格式
data_file = 'F:/MyProjects/Data/txt/LCD差值.txt'
months = []
farmland_diff = []
forest_diff = []
water_diff = []

with open(data_file, 'r') as file:
    lines = file.readlines()
    for line in lines[1:]:  # 跳过第一行标题
        parts = line.split()
        months.append(parts[0])
        farmland_diff.append(float(parts[1]))
        forest_diff.append(float(parts[2]))
        water_diff.append(float(parts[3]))

# 创建绘图对象
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制折线图
ax.plot(months, water_diff, label='Water', marker='o')
ax.plot(months, farmland_diff, label='Farmland', marker='o')
ax.plot(months, forest_diff, label='Forest', marker='o')

# 添加图例和标签，设置字体和大小
ax.set_xlabel('Month', fontsize=20, fontname='Times New Roman')
ax.set_ylabel('Average Temperature Difference', fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='both', which='major', labelsize=16, labelcolor='black', direction='in', length=6, width=2)
ax.legend(fontsize=20, prop={'family': 'Times New Roman'})  # 调整图例字体大小

# 显示网格
ax.grid(True)

# 保存并显示图表
plt.savefig(output_folder, dpi=1000)
plt.show()
