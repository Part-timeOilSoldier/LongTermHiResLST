import numpy as np
import matplotlib.pyplot as plt

# 相关性矩阵数据
correlation_matrix = np.array([
[-0.88407105, -0.04252764, -1.41523259, -2.42376383, -0.61088118, -2.61502179, -0.46523756, -8.76298733, -0.02120453, -1.93884762, -1.58872499, -0.70326398],
 [ 2.33925637,  0.43062793, -0.31536176, -8.55903897,  7.28734306, -20.73947889, -2.5656298,  18.71409538, -0.04090051, -3.0796701,  -0.33507752,  2.05566425],
 [-1.14379193, -1.28701567, -2.10670275, -1.79473366, -1.37969051, -1.79107469, -0.40566547,  0.59843051, -0.78620398, -0.91571559, -1.57061531, -0.67777853],
 [ 0.10052962, -0.30968032, -1.68688416, -8.20407368, -0.65976205, -5.44575275, -0.13873817, -15.04257462, -0.57364465, -1.31151885, -0.711017,   -0.47312345]
]
)

# 定义月份和土地利用类型标签
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
land_cover_labels = ['Farmland', 'Forest', 'Water', 'Building']

# 创建图表
fig, ax = plt.subplots(figsize=(12, 6))
cax = ax.matshow(correlation_matrix, cmap='coolwarm')

# 添加颜色条，设置在底部
cbar = fig.colorbar(cax, orientation='horizontal', pad=0.2)

# 设置轴标签
ax.set_xticks(np.arange(len(months)))
ax.set_yticks(np.arange(len(land_cover_labels)))
ax.set_xticklabels(months, fontname='Times New Roman', fontsize=15)
ax.set_yticklabels(land_cover_labels, fontname='Times New Roman', fontsize=15)

# 旋转x轴标签
plt.xticks(rotation=45)

# 在方框中打印具体数值
for i in range(correlation_matrix.shape[0]):
    for j in range(correlation_matrix.shape[1]):
        text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                       ha='center', va='center', color='black', fontname='Times New Roman', fontsize=15)

# 添加标题

plt.savefig('F:/论文/论文基础/超大字体图片/LofN.png', dpi=1000)
# 显示图表
plt.show()
