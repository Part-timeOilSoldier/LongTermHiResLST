import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib as mpl
import pandas as pd

# To ensure only Times New Roman (English) is used, we set the font family to 'Times New Roman'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'

# Land use type mapping dictionary (only for types used in this study)
landuse_labels = {
    1: "Farmland",
    2: "Forest",
    5: "Waterbody",
    8: "Built-up"
}


def analyze_tif(file_path):
    with rasterio.open(file_path) as src:
        print(f"\nAnalyzing file: {file_path}")
        print("Basic information:")
        print("  Width: ", src.width)
        print("  Height: ", src.height)
        print("  Band count: ", src.count)
        print("  CRS: ", src.crs)
        print("  Transform: ", src.transform)
        print("-" * 40)
        for band in range(1, src.count + 1):
            data = src.read(band)
            min_val = np.nanmin(data)
            max_val = np.nanmax(data)
            mean_val = np.nanmean(data)
            std_val = np.nanstd(data)
            print(f"Statistics for band {band}:")
            print("  Min: ", min_val)
            print("  Max: ", max_val)
            print("  Mean: ", mean_val)
            print("  Std: ", std_val)
            print("-" * 40)


def display_temperature_images(file_paths, output_folder=None, dpi=500, vmin=295, vmax=365):
    """
    Display temperature images with fixed color limits [vmin, vmax],
    and a common horizontal colorbar at the top (using 'RdYlGn_r' colormap).

    Each subplot displays the first band of the image.
    Titles are placed below each image.
    All text is in Times New Roman font.
    If output_folder is provided, the resulting figure is saved there.

    Parameters:
      file_paths: List of image file paths (strings), total 3 images.
      vmin: Minimum value for color mapping (default 295).
      vmax: Maximum value for color mapping (default 365).
      output_folder: (Optional) Folder path to save the output image.
      dpi: (Optional) Resolution of the saved image (default 500).
    """
    num_images = len(file_paths)
    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    ims = []  # Store imshow objects for later use in the common colorbar.
    custom_titles = ["LST_1000M", "Predect_LST_30M", "LST_30M"]

    for ax, file_path, title_text in zip(axes, file_paths, custom_titles):
        with rasterio.open(file_path) as src:
            data = src.read(1)
        im = ax.imshow(data, cmap='RdYlGn_r', vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(title_text, fontname="Times New Roman", fontsize=16, labelpad=10)
        ax.xaxis.set_label_position('bottom')
        ims.append(im)

    plt.subplots_adjust(top=0.95, bottom=0.0, left=0.05, right=0.95, wspace=0.1)
    cbar_ax = fig.add_axes([0.05, 0.88, 0.9, 0.03])
    cbar = fig.colorbar(ims[0], cax=cbar_ax, orientation="horizontal")
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(labelsize=12)
    for label in cbar.ax.get_xticklabels():
        label.set_fontname("Times New Roman")

    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, "temperature_images.png")
        fig.savefig(output_path, dpi=dpi)

    plt.show()


def plot_r2_scatter(file1, file2, sample_size=10000, output_folder=None, dpi=500):
    """
    Plot a scatter plot for two temperature images (first band),
    compute linear regression and display the R² value with the fitted line.
    If image sizes differ, use the overlapping region; if too many points, subsample.
    If output_folder is provided, the resulting figure is saved there.
    """
    with rasterio.open(file1) as src1, rasterio.open(file2) as src2:
        data1 = src1.read(1)
        data2 = src2.read(1)
    if data1.shape != data2.shape:
        print("Warning: image sizes differ; using overlapping region")
        rows = min(data1.shape[0], data2.shape[0])
        cols = min(data1.shape[1], data2.shape[1])
        data1 = data1[:rows, :cols]
        data2 = data2[:rows, :cols]
    x = data1.flatten()
    y = data2.flatten()
    if len(x) > sample_size:
        indices = np.random.choice(len(x), sample_size, replace=False)
        x = x[indices]
        y = y[indices]
    slope, intercept = np.polyfit(x, y, 1)
    r_value = np.corrcoef(x, y)[0, 1]
    r_squared = r_value ** 2
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=1, alpha=0.5, label='Data Points')
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, color='red', label=f'Fit Line (R²={r_squared:.3f})')
    plt.xlabel("Temperature Image 1")
    plt.ylabel("Temperature Image 2")
    plt.title("Scatter Plot and Fit Line")
    plt.legend()
    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, "r2_scatter.png")
        plt.savefig(output_path, dpi=dpi)
    plt.show()


def count_landuse_types(file_path):
    """
    Count the pixel types for land use data.
    Reads the first band of the tif file and uses np.unique.
    """
    with rasterio.open(file_path) as src:
        data = src.read(1)
    unique_values, counts = np.unique(data, return_counts=True)
    print(f"\nLand use type counts ({file_path}):")
    for value, count in zip(unique_values, counts):
        label = landuse_labels.get(value, f"Type {value}")
        print(f"  {label}: {count} pixels")
    print(f"Total {len(unique_values)} types.")


def compute_r2_by_landuse(pred_file, act_file, landuse_file, ignore_types=[4, 7]):
    """
    For each land use type (except those in ignore_types), compute the pixel count,
    mean predicted and actual temperature, and R² between them.
    Uses the overlapping region of the three images.
    """
    with rasterio.open(pred_file) as src_pred, \
            rasterio.open(act_file) as src_act, \
            rasterio.open(landuse_file) as src_land:
        pred_data = src_pred.read(1)
        act_data = src_act.read(1)
        landuse_data = src_land.read(1)
    common_rows = min(pred_data.shape[0], act_data.shape[0], landuse_data.shape[0])
    common_cols = min(pred_data.shape[1], act_data.shape[1], landuse_data.shape[1])
    pred_data = pred_data[:common_rows, :common_cols]
    act_data = act_data[:common_rows, :common_cols]
    landuse_data = landuse_data[:common_rows, :common_cols]
    unique_types = np.unique(landuse_data)
    print("\n[Land Use Type Statistics for Predicted vs Actual Temperature]")
    for t in unique_types:
        if t in ignore_types:
            continue
        mask = (landuse_data == t)
        count = np.sum(mask)
        if count == 0:
            continue
        pred_values = pred_data[mask]
        act_values = act_data[mask]
        mean_pred = np.mean(pred_values)
        mean_act = np.mean(act_values)
        if len(pred_values) > 1:
            r_value = np.corrcoef(pred_values.flatten(), act_values.flatten())[0, 1]
            r_squared = r_value ** 2
        else:
            r_squared = np.nan
        label = landuse_labels.get(t, f"Type {t}")
        print(f"{label}:")
        print(f"  Pixel Count: {count}")
        print(f"  Predicted Mean Temperature: {mean_pred:.2f} K")
        print(f"  Actual Mean Temperature: {mean_act:.2f} K")
        print(f"  R²: {r_squared:.3f}")
        print("-" * 40)


def plot_scatter_by_landuse(pred_file, act_file, landuse_file, ignore_types=[4, 7],
                            sample_size=10000, output_folder=None, dpi=500):
    """
    For each land use type (except those in ignore_types), plot a scatter plot comparing
    predicted and actual temperature, compute linear regression and R², and display them in subplots.
    Each scatter plot is labeled with a letter in parentheses (e.g., (a), (b), ...), with letters in 24pt font.
    All legends are placed uniformly at the lower left corner.
    All text is in Times New Roman, English.
    If output_folder is provided, the resulting figure is saved there.
    """
    with rasterio.open(pred_file) as src_pred, \
            rasterio.open(act_file) as src_act, \
            rasterio.open(landuse_file) as src_land:
        pred_data = src_pred.read(1)
        act_data = src_act.read(1)
        landuse_data = src_land.read(1)
    common_rows = min(pred_data.shape[0], act_data.shape[0], landuse_data.shape[0])
    common_cols = min(pred_data.shape[1], act_data.shape[1], landuse_data.shape[1])
    pred_data = pred_data[:common_rows, :common_cols]
    act_data = act_data[:common_rows, :common_cols]
    landuse_data = landuse_data[:common_rows, :common_cols]

    unique_types = np.unique(landuse_data)
    valid_types = [t for t in unique_types if t not in ignore_types]
    num_types = len(valid_types)
    ncols = 2
    nrows = math.ceil(num_types / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if nrows * ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, t in enumerate(valid_types):
        mask = (landuse_data == t)
        count = np.sum(mask)
        if count < 2:
            print(f"{landuse_labels.get(t, f'Type {t}')}: Not enough pixels, skipping scatter plot")
            continue
        pred_values = pred_data[mask].flatten()
        act_values = act_data[mask].flatten()
        if len(pred_values) > sample_size:
            indices = np.random.choice(len(pred_values), sample_size, replace=False)
            pred_values = pred_values[indices]
            act_values = act_values[indices]
        slope, intercept = np.polyfit(pred_values, act_values, 1)
        r_value = np.corrcoef(pred_values, act_values)[0, 1]
        r_squared = r_value ** 2

        ax = axes[i]
        ax.scatter(pred_values, act_values, s=1, alpha=0.5, label='Data Points')
        x_line = np.linspace(pred_values.min(), pred_values.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color='red', label=f'Fit (R²={r_squared:.3f})')
        ax.set_xlabel("Predicted Temperature", fontname="Times New Roman", fontsize=16, fontweight="bold")
        ax.set_ylabel("Actual Temperature", fontname="Times New Roman", fontsize=16, fontweight="bold")
        ax.set_title(f"{landuse_labels.get(t, f'Type {t}')}",
                     fontname="Times New Roman", fontsize=16, fontweight="bold")
        ax.text(0.1, 0.9, f"({chr(97 + i)})", transform=ax.transAxes,
                fontname="Times New Roman", fontsize=34, fontweight="bold",
                ha="left", va="top")
        ax.legend(loc="lower right", prop={'family': 'Times New Roman', 'size': 16, 'weight': 'bold'})

    # 删除多余子图
    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, "scatter_by_landuse.png")
        fig.savefig(output_path, dpi=dpi)
    plt.show()


def plot_pixel_distribution(file_paths, bins=50, value_range=(295, 340), output_folder=None, dpi=500):
    """
    绘制像素分布折线图，图例名称与另一张图相同：
      - Row_LST_1000M
      - Predect_LST_30M
      - Actual_LST_30M
    """

    # 这里手动指定与另一张图相同的名称，顺序要与 file_paths 对应
    custom_titles = ["Row_LST_1000M", "Predect_LST_30M", "Actual_LST_30M"]

    plt.figure(figsize=(8, 6))

    for file_path, custom_label in zip(file_paths, custom_titles):
        with rasterio.open(file_path) as src:
            data = src.read(1).flatten()

        hist, bin_edges = np.histogram(data, bins=bins, range=value_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 将图例名称改为与另一张图一致
        plt.plot(bin_centers, hist, marker='o', linestyle='-', label=custom_label)

    plt.xlabel("Temperature (K)")
    plt.ylabel("Frequency")
    plt.title("Pixel Distribution of Temperature Images")
    plt.legend()
    plt.tight_layout()

    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, "pixel_distribution.png")
        plt.savefig(output_path, dpi=dpi)

    plt.show()


def plot_distribution_by_landuse(row_file, pred_file, act_file, landuse_file, ignore_types=[4, 7],
                                 bins=50, value_range=(295, 365), output_folder=None, dpi=500):
    """
    对每个（除去 ignore_types 外的）地表利用类型：
      - 分别从 Row Data、Predicted 和 Actual 数据中提取温度值，
      - 分别计算各自的统计指标和直方图（过滤掉频次低于总像元 0.1% 的直方图箱）；
    最后：
      - 分别为“Row Data”、“Predicted”、“Actual”绘制一张折线图，
      - 将所有统计信息汇总为一张表格图展示。
    如果 output_folder 不为 None，则将每个图像保存到指定文件夹中。
    """
    with rasterio.open(row_file) as src_row, \
            rasterio.open(pred_file) as src_pred, \
            rasterio.open(act_file) as src_act, \
            rasterio.open(landuse_file) as src_land:
        row_data = src_row.read(1)
        pred_data = src_pred.read(1)
        act_data = src_act.read(1)
        landuse_data = src_land.read(1)

    common_rows = min(row_data.shape[0], pred_data.shape[0], act_data.shape[0], landuse_data.shape[0])
    common_cols = min(row_data.shape[1], pred_data.shape[1], act_data.shape[1], landuse_data.shape[1])
    row_data = row_data[:common_rows, :common_cols]
    pred_data = pred_data[:common_rows, :common_cols]
    act_data = act_data[:common_rows, :common_cols]
    landuse_data = landuse_data[:common_rows, :common_cols]

    unique_types = np.unique(landuse_data)
    valid_types = [t for t in unique_types if t not in ignore_types]

    hist_data = {
        'Row Data': {},
        'Predicted': {},
        'Actual': {}
    }
    table_rows = []

    def compute_stats(temp_values, bins, value_range):
        count = temp_values.size
        mean_val = np.mean(temp_values)
        max_val = np.max(temp_values)
        min_val = np.min(temp_values)
        range_val = max_val - min_val
        std_val = np.std(temp_values)
        cv_before = (std_val / mean_val) * 100 if mean_val != 0 else np.nan

        if value_range is not None:
            hist, bin_edges = np.histogram(temp_values, bins=bins, range=value_range)
        else:
            hist, bin_edges = np.histogram(temp_values, bins=bins)
        threshold = 0.001 * count  # 0.1% 阈值
        valid_bins = hist >= threshold
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        filtered_bin_centers = bin_centers[valid_bins]
        filtered_hist = hist[valid_bins]

        bin_indices = np.digitize(temp_values, bin_edges) - 1
        valid_value_mask = np.array([valid_bins[idx] if (0 <= idx < len(valid_bins)) else False for idx in bin_indices])
        filtered_temp_values = temp_values[valid_value_mask]
        if filtered_temp_values.size > 0:
            filtered_std = np.std(filtered_temp_values)
            filtered_mean = np.mean(filtered_temp_values)
            cv_after = (filtered_std / filtered_mean) * 100 if filtered_mean != 0 else np.nan
        else:
            filtered_std, cv_after = np.nan, np.nan

        stats = {
            'Pixel Count': count,
            'Mean': mean_val,
            'Max': max_val,
            'Min': min_val,
            'Range': range_val,
            'CV Before': cv_before,
            'Filtered Std': filtered_std,
            'CV After': cv_after
        }
        return stats, (filtered_bin_centers, filtered_hist)

    for t in valid_types:
        mask = (landuse_data == t)
        if np.sum(mask) == 0:
            continue
        label_text = landuse_labels.get(t, f"Type {t}")

        row_vals = row_data[mask].flatten()
        pred_vals = pred_data[mask].flatten()
        act_vals = act_data[mask].flatten()

        stats_row, hist_row = compute_stats(row_vals, bins, value_range)
        stats_pred, hist_pred = compute_stats(pred_vals, bins, value_range)
        stats_act, hist_act = compute_stats(act_vals, bins, value_range)

        row_dict = {'Land Use': label_text, 'Data Type': 'Row Data'}
        row_dict.update(stats_row)
        table_rows.append(row_dict)

        pred_dict = {'Land Use': label_text, 'Data Type': 'Predicted'}
        pred_dict.update(stats_pred)
        table_rows.append(pred_dict)

        act_dict = {'Land Use': label_text, 'Data Type': 'Actual'}
        act_dict.update(stats_act)
        table_rows.append(act_dict)

        hist_data['Row Data'][label_text] = hist_row
        hist_data['Predicted'][label_text] = hist_pred
        hist_data['Actual'][label_text] = hist_act

    # 分别绘制直线图并保存
    for data_type, type_hist in hist_data.items():
        plt.figure(figsize=(8, 6))
        for label, (bin_centers, hist_values) in type_hist.items():
            plt.plot(bin_centers, hist_values, marker='o', linestyle='-', label=label)
        plt.xlabel("Temperature (K)")
        plt.ylabel("Frequency")
        plt.title(f"{data_type} Temperature Distribution by Land Use Type")
        plt.legend()
        plt.tight_layout()
        if output_folder is not None:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            filename = f"distribution_by_landuse_{data_type.replace(' ', '_')}.png"
            plt.savefig(os.path.join(output_folder, filename), dpi=dpi)
        plt.show()

    # 绘制统计表
    df = pd.DataFrame(table_rows)
    df_display = df.copy()
    for col in ['Mean', 'Max', 'Min', 'Range', 'CV Before', 'Filtered Std', 'CV After']:
        df_display[col] = df_display[col].round(2)
    fig, ax = plt.subplots(figsize=(12, len(df_display) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df_display.values, colLabels=df_display.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.title("Temperature Statistics by Land Use and Data Type")
    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, "temperature_statistics_table.png")
        fig.savefig(output_path, dpi=dpi)
    plt.show()


def load_data(data):
    """
    如果 data 是文件路径（字符串），则读取影像第一波段的数据；否则认为 data 已经是数组，直接返回。
    """
    if isinstance(data, str):
        with rasterio.open(data) as src:
            return src.read(1)
    else:
        return data


def plot_rmse_by_landcover(pred_data, obs_data, landcover_data, landcover_labels, ignore_types=[], output_folder=None,
                           dpi=500):
    """
    计算并打印六种固定类型（Type1, Type2, Type4, Type5, Type7, Type8）的 RMSE，
    并以雷达图形式展示结果，同时在图片左下角显示一个框，框内列出各类型说明，
    并在图例中显示红色蜘蛛线代表 RMSE，并在雷达图中对每个点标注具体数值。
    如果 output_folder 不为 None，则将图像保存到该文件夹中。
    """
    pred_img = load_data(pred_data)
    obs_img = load_data(obs_data)
    landcover_img = load_data(landcover_data)

    types_to_plot = [1, 2, 4, 5, 7, 8]
    rmse_values = []

    for t in types_to_plot:
        if t in ignore_types:
            rmse = 0.0
            print(f"Type{t} (Ignored): RMSE = {rmse:.2f}")
        else:
            mask = (landcover_img == t)
            if np.sum(mask) == 0:
                rmse = 0.0
            else:
                p_vals = pred_img[mask]
                o_vals = obs_img[mask]
                rmse = np.sqrt(np.mean((p_vals - o_vals) ** 2))
            print(f"Type{t}: RMSE = {rmse:.2f}")
        rmse_values.append(rmse)

    labels = [f"Type{t}" for t in types_to_plot]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    rmse_plot = rmse_values + [rmse_values[0]]
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    line, = ax.plot(angles, rmse_plot, color='red', linewidth=3, label="RMSE")
    ax.fill(angles, rmse_plot, color='red', alpha=0.25)
    fig.text(0.5, 0.98, "RMSE by Land Cover Type", ha="center", va="top",
             fontname="Times New Roman", fontsize=20)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontname="Times New Roman", fontsize=16)
    ax.tick_params(axis='x', pad=20)

    custom_offsets = {
        0: (20, 0),
        1: (10, 10),
        2: (-10, 10),
        3: (-20, 0),
        4: (-15, 18),
        5: (5, -13)
    }

    for i in range(num_vars):
        angle = angles[i]
        value = rmse_values[i]
        offset = custom_offsets.get(i, (10 * np.cos(angle), 10 * np.sin(angle)))
        ax.annotate(f"{value:.2f}",
                    xy=(angle, value),
                    xytext=offset,
                    textcoords="offset points",
                    ha="center", va="center",
                    fontsize=16, fontname="Times New Roman", color="blue")

    # 添加自定义图例（红色蜘蛛线）
    legend_ax = fig.add_axes([0.45, 0.01, 0.1, 0.1])
    legend_ax.axis("off")
    legend_ax.plot([0.1, 0.3], [0.5, 0.5], color='red', lw=3)
    legend_ax.text(0.35, 0.5, "RMSE", fontname="Times New Roman", fontsize=16, va="center", ha="left")

    legend_entries = []
    for t in types_to_plot:
        if t in ignore_types:
            legend_entries.append(f"Type{t}: Ignored")
        elif t in landcover_labels:
            legend_entries.append(f"Type{t}: {landcover_labels[t]}")
        else:
            legend_entries.append(f"Type{t}: Ignored")
    legend_text = "\n".join(legend_entries)
    plt.figtext(0.01, 0.01, legend_text, ha="left", va="bottom", fontsize=16,
                fontname="Times New Roman",
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='square'))

    plt.tight_layout()
    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, "rmse_by_landcover.png")
        fig.savefig(output_path, dpi=dpi)
    plt.show()


def crop_and_subplots_images(file_paths, output_folder=None, dpi=500,
                             crop_frac=0.5, crop_pos=(0.5, 0.5)):
    """
    先基于原图像的所有像素确定统一的色带范围，再执行裁剪并以子图方式显示裁剪结果。
    如果 output_folder 被指定，则将结果保存到该文件夹下，文件名为 'combined_cropped_subplots.png'。
    """
    full_images = []
    for file_path in file_paths:
        with rasterio.open(file_path) as src:
            data = src.read(1)
        full_images.append(data)

    all_pixels = np.concatenate([img.ravel() for img in full_images])
    global_vmin = np.min(all_pixels)
    global_vmax = np.max(all_pixels)

    cropped_images = []
    for data in full_images:
        height, width = data.shape
        crop_width = int(width * crop_frac)
        crop_height = int(height * crop_frac)
        center_x = int(crop_pos[0] * width)
        center_y = int(crop_pos[1] * height)
        left = max(0, center_x - crop_width // 2)
        top = max(0, center_y - crop_height // 2)
        if left + crop_width > width:
            left = width - crop_width
        if top + crop_height > height:
            top = height - crop_height
        cropped = data[top:top + crop_height, left:left + crop_width]
        cropped_images.append(cropped)

    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    custom_titles = ["Cropped 1", "Cropped 2", "Cropped 3"]
    ims = []
    for ax, img, title_text in zip(axes, cropped_images, custom_titles):
        im = ax.imshow(img, cmap='RdYlGn_r', vmin=global_vmin, vmax=global_vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(title_text, fontname="Times New Roman", fontsize=16, labelpad=10)
        ax.xaxis.set_label_position('bottom')
        ims.append(im)

    plt.subplots_adjust(top=0.95, bottom=0.0, left=0.05, right=0.95, wspace=0.1)
    cbar_ax = fig.add_axes([0.05, 0.88, 0.9, 0.03])
    cbar = fig.colorbar(ims[0], cax=cbar_ax, orientation="horizontal")
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(labelsize=12)
    for label in cbar.ax.get_xticklabels():
        label.set_fontname("Times New Roman")

    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, "combined_cropped_subplots.png")
        fig.savefig(output_path, dpi=dpi)
    plt.show()


def display_original_and_cropped_images(file_paths, output_folder=None, dpi=500,
                                        vmin=295, vmax=365, crop_frac=0.1, crop_pos=(0.42, 0.7)):
    """
    从三个图像中渲染原图并标记裁剪区域，显示原图和裁剪图，并用箭头标示二者关系。
    如果 output_folder 被指定，则将结果保存到该文件夹下，文件名为 'combined_original_and_cropped.png'。
    """
    from matplotlib.patches import Rectangle, ConnectionPatch

    original_images = []
    cropped_images = []
    crop_coords = []
    titles = ["Row_LST_1000M", "Predect_LST_30M", "Actual_LST_30M"]

    for file_path in file_paths:
        with rasterio.open(file_path) as src:
            data = src.read(1)
        original_images.append(data)
        height, width = data.shape
        crop_width = int(width * crop_frac)
        crop_height = int(height * crop_frac)
        center_x = int(crop_pos[0] * width)
        center_y = int(crop_pos[1] * height)
        left = max(0, center_x - crop_width // 2)
        top = max(0, center_y - crop_height // 2)
        if left + crop_width > width:
            left = width - crop_width
        if top + crop_height > height:
            top = height - crop_height
        crop_coords.append((left, top, crop_width, crop_height))
        cropped = data[top:top + crop_height, left:left + crop_width]
        cropped_images.append(cropped)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    ims = []
    for i, ax in enumerate(axes[0]):
        im = ax.imshow(original_images[i], cmap='RdYlGn_r', vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(titles[i], fontname="Times New Roman", fontsize=20, labelpad=10)
        ax.xaxis.set_label_position('bottom')
        left, top, crop_width, crop_height = crop_coords[i]
        rect = Rectangle((left, top), crop_width, crop_height,
                         linewidth=3, edgecolor='black', facecolor='none', alpha=0.7)
        ax.add_patch(rect)
        ims.append(im)

    for i, ax in enumerate(axes[1]):
        im = ax.imshow(cropped_images[i], cmap='RdYlGn_r', vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(titles[i], fontname="Times New Roman", fontsize=20, labelpad=10)
        ax.xaxis.set_label_position('bottom')
        ims.append(im)

    for i in range(3):
        top_ax = axes[0][i]
        bottom_ax = axes[1][i]
        left, top, crop_width, crop_height = crop_coords[i]
        bl = (left, top + crop_height)
        br = (left + crop_width, top + crop_height)
        target_left = (0, 1)
        target_right = (1, 1)
        con_left = ConnectionPatch(xyA=bl, coordsA=top_ax.transData,
                                   xyB=target_left, coordsB=bottom_ax.transAxes,
                                   arrowstyle="->", color="black", linewidth=3)
        fig.add_artist(con_left)
        con_right = ConnectionPatch(xyA=br, coordsA=top_ax.transData,
                                    xyB=target_right, coordsB=bottom_ax.transAxes,
                                    arrowstyle="->", color="black", linewidth=3)
        fig.add_artist(con_right)

    plt.subplots_adjust(top=0.93, bottom=0.05, left=0.05, right=0.95, hspace=0.15, wspace=0.1)
    cbar_ax = fig.add_axes([0.05, 0.93, 0.9, 0.03])
    cbar = fig.colorbar(ims[0], cax=cbar_ax, orientation="horizontal")
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(labelsize=23)
    for label in cbar.ax.get_xticklabels():
        label.set_fontname("Times New Roman")

    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, "combined_original_and_cropped.png")
        fig.savefig(output_path, dpi=dpi)
    plt.show()


def plot_distribution_by_landuse_1(row_file, pred_file, act_file, landuse_file, ignore_types=[4, 7],
                                   bins=50, value_range=(295, 365), output_folder=None, dpi=500):
    """
    对每个地表利用类型（除 ignore_types 外），提取 Row Data、Predicted 和 Actual 温度值，
    计算统计指标，并展示包含 Land Use, Data Type, Pixel Count, Std 和 CV 的表格。
    如果 output_folder 被指定，则将表格图保存到该文件夹下，文件名为 'distribution_by_landuse_table.png'。
    """
    with rasterio.open(row_file) as src_row, \
            rasterio.open(pred_file) as src_pred, \
            rasterio.open(act_file) as src_act, \
            rasterio.open(landuse_file) as src_land:
        row_data = src_row.read(1)
        pred_data = src_pred.read(1)
        act_data = src_act.read(1)
        landuse_data = src_land.read(1)

    common_rows = min(row_data.shape[0], pred_data.shape[0], act_data.shape[0], landuse_data.shape[0])
    common_cols = min(row_data.shape[1], pred_data.shape[1], act_data.shape[1], landuse_data.shape[1])
    row_data = row_data[:common_rows, :common_cols]
    pred_data = pred_data[:common_rows, :common_cols]
    act_data = act_data[:common_rows, :common_cols]
    landuse_data = landuse_data[:common_rows, :common_cols]

    unique_types = np.unique(landuse_data)
    valid_types = [t for t in unique_types if t not in ignore_types]

    table_rows = []

    def compute_stats(temp_values, bins, value_range):
        count = temp_values.size
        mean_val = np.mean(temp_values)
        max_val = np.max(temp_values)
        min_val = np.min(temp_values)
        range_val = max_val - min_val
        std_val = np.std(temp_values)
        cv_before = (std_val / mean_val) * 100 if mean_val != 0 else np.nan

        if value_range is not None:
            hist, bin_edges = np.histogram(temp_values, bins=bins, range=value_range)
        else:
            hist, bin_edges = np.histogram(temp_values, bins=bins)
        threshold = 0.001 * count
        valid_bins = hist >= threshold
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        stats = {
            'Pixel Count': count,
            'Mean': mean_val,
            'Max': max_val,
            'Min': min_val,
            'Range': range_val,
            'Std': std_val,
            'CV Before': cv_before
        }
        return stats, (bin_centers[valid_bins], hist[valid_bins])

    for t in valid_types:
        mask = (landuse_data == t)
        if np.sum(mask) == 0:
            continue
        label_text = landuse_labels.get(t, f"Type {t}")

        row_vals = row_data[mask].flatten()
        pred_vals = pred_data[mask].flatten()
        act_vals = act_data[mask].flatten()

        stats_row, _ = compute_stats(row_vals, bins, value_range)
        stats_pred, _ = compute_stats(pred_vals, bins, value_range)
        stats_act, _ = compute_stats(act_vals, bins, value_range)

        new_row = {
            'Land Use': label_text,
            'Data Type': 'Row Data',
            'Pixel Count': stats_row['Pixel Count'],
            'Std': stats_row['Std'],
            'CV': stats_row['CV Before']
        }
        table_rows.append(new_row)

        new_pred = {
            'Land Use': label_text,
            'Data Type': 'Predicted',
            'Pixel Count': stats_pred['Pixel Count'],
            'Std': stats_pred['Std'],
            'CV': stats_pred['CV Before']
        }
        table_rows.append(new_pred)

        new_act = {
            'Land Use': label_text,
            'Data Type': 'Actual',
            'Pixel Count': stats_act['Pixel Count'],
            'Std': stats_act['Std'],
            'CV': stats_act['CV Before']
        }
        table_rows.append(new_act)

    df = pd.DataFrame(table_rows, columns=['Land Use', 'Data Type', 'Pixel Count', 'Std', 'CV'])
    df_display = df.copy()
    for col in ['Pixel Count', 'Std', 'CV']:
        df_display[col] = df_display[col].round(2)

    fig, ax = plt.subplots(figsize=(12, len(df_display) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df_display.values, colLabels=df_display.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.title("Temperature Statistics by Land Use and Data Type", fontname="Times New Roman", fontsize=16,
              fontweight="bold")
    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, "distribution_by_landuse_table.png")
        fig.savefig(output_path, dpi=dpi)
    plt.show()

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def combine_two_images_vertical(top_image_path, bottom_image_path,
                                output_folder=None, dpi=300):
    """
    将两张图像(路径分别为 top_image_path 和 bottom_image_path)上下合并为一张图。
    如果指定 output_folder，则保存输出图像到该文件夹中，文件名为 "final_merged.png"。
    dpi 默认为 300。
    """
    import os

    # 读取图像
    img_top = mpimg.imread(top_image_path)
    img_bottom = mpimg.imread(bottom_image_path)

    # 创建上下布局
    fig, axes = plt.subplots(2, 1, figsize=(8, 12))  # 可根据需要调整 figsize
    # 显示顶部图像
    axes[0].imshow(img_top)
    axes[0].axis('off')

    # 显示底部图像
    axes[1].imshow(img_bottom)
    axes[1].axis('off')

    plt.tight_layout()

    # 若指定了输出文件夹，则保存
    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        outpath = os.path.join(output_folder, "final_merged.png")
        plt.savefig(outpath, dpi=dpi)

    plt.show()


if __name__ == "__main__":
    pred_file = r"E:\MyProjects\MachineLearning\Data\Final_Data\Predict_Data\Predict_LST_06.tif"
    act_file = r"E:\MyProjects\MachineLearning\Data\Usable_Data\LST_30m\LC08_L2SP_121038_20220616_20220411_02_T1_ST_B10_processed.tif"
    row_file = r"E:\MyProjects\MachineLearning\Data\Usable_Data\LST_30m_990m\LST_Day_1km_20220616.tif"
    landuse_file = r"E:\MyProjects\MachineLearning\Data\Usable_Data\Type_30m\anhuiwgs84_6.tif"
    out_file = r"D:\LaoGong\论文图片"
    temperature_files = [row_file, pred_file, act_file]
    # for file in temperature_files:
    #     analyze_tif(file)
    #
    # display_temperature_images(temperature_files, out_file, 500, vmin=295, vmax=345)
    # crop_and_subplots_images(temperature_files, out_file)
    # display_original_and_cropped_images(temperature_files, out_file)
    # plot_r2_scatter(pred_file, act_file, output_folder=out_file)
    # count_landuse_types(landuse_file)
    # compute_r2_by_landuse(pred_file, act_file, landuse_file, ignore_types=[4, 7])
    # plot_scatter_by_landuse(pred_file, act_file, landuse_file, ignore_types=[4, 7], output_folder=out_file)
    # plot_pixel_distribution(temperature_files, bins=50, value_range=(295, 340), output_folder=out_file)
    # plot_distribution_by_landuse(row_file, pred_file, act_file, landuse_file, ignore_types=[4, 7],
    #                              bins=50, value_range=(295, 365), output_folder=out_file)
    # plot_rmse_by_landcover(pred_file, act_file, landuse_file, landuse_labels, output_folder=out_file)
    # plot_distribution_by_landuse_1(row_file, pred_file, act_file, landuse_file, output_folder=out_file)
    # 最后，将上面生成的两张图上下合并为一张
    top_img_path = os.path.join(out_file, "pixel_distribution.png")
    bottom_img_path = os.path.join(out_file, "combined_original_and_cropped.png")

    combine_two_images_vertical(
        top_image_path=top_img_path,
        bottom_image_path=bottom_img_path,
        output_folder=out_file,  # 保存到同一个文件夹
        dpi=500  # 分辨率可自定义
    )

