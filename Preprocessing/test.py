import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# Function to read data from the file
def read_data(filename):
    data = {}
    with open(filename, 'r') as file:
        for line in file:
            if line.strip().isdigit():
                month = int(line.strip())
                data[month] = {}
            elif "average" in line:
                parts = line.split()
                lcd_type = parts[0]
                average = float(parts[3].split('=')[1])
                data[month][lcd_type] = average
    return data


# Function to plot data
def plot_data(data, save_path):
    labels = ["Farmland", "Forest", "Water", "Building"]
    lcd_map = {"LCD1": "Farmland", "LCD2": "Forest", "LCD5": "Water", "LCD8": "Building"}
    lcd_colors = {
        "LCD1": ('#FFB74D', '#F57C00'),  # Slightly darker peach fill, Dark peach edge
        "LCD2": ('#C0CA33', '#388E3C'),  # Slightly darker green fill, Dark green edge
        "LCD5": ('#90CAF9', '#1976D2'),  # Slightly darker blue fill, Dark blue edge
        "LCD8": ('#BA68C8', '#8E24AA')  # Slightly darker orange fill, Dark orange edge
    }

    months = list(data.keys())
    x = np.arange(len(months))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, lcd in enumerate(["LCD1", "LCD2", "LCD5", "LCD8"]):
        values = [data[month].get(lcd, np.nan) for month in months]
        rects = ax.bar(x + i * width, values, width, label=lcd_map[lcd],
                       color=lcd_colors[lcd][0], edgecolor=lcd_colors[lcd][1], linewidth=2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Month', fontname='Times New Roman', fontsize=20)
    ax.set_ylabel('Average Temperature (K)', fontname='Times New Roman', fontsize=20)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([str(month).zfill(2) for month in months], fontname='Times New Roman', fontsize=20)
    ax.set_ylim(270, 330)
    ax.set_yticks(np.arange(270, 340, 10))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis='y', linestyle='--', linewidth=1.5)

    # Add legend
    legend = ax.legend(prop={'size': 18, 'family': 'Times New Roman'})  # 调整图例字体大小

    # Set font for y-axis and x-axis values
    ax.tick_params(axis='y', which='major', labelsize=20, labelcolor='black', direction='in', length=6, width=2)
    ax.tick_params(axis='x', which='major', labelsize=24, labelcolor='black', direction='in', length=6, width=2)

    fig.tight_layout()

    # Save the figure
    plt.savefig(save_path, dpi=1000)

    plt.show()


# Read the data from the input file
data = read_data('F:/MyProjects/Data/TXT/txt1/lst_stats_by_land_cover_with_average.txt')

# Plot the data and save to specified path
plot_data(data, 'F:/论文/论文基础/超大字体图片/lst_stats_plot.png')
