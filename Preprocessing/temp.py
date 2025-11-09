import rasterio
from rasterio.enums import Resampling


def clamp_image(input_tif, output_tif):
    # 打开原始图像
    with rasterio.open(input_tif) as src:
        data = src.read(1)  # 读取第一个波段

        # 将数据大于1的部分设置为1，小于-1的部分设置为-1
        data[data > 1] = 1
        data[data < -1] = -1

        # 设置输出图像的元数据与输入图像相同
        profile = src.profile

        # 写入修改后的数据到新文件
        with rasterio.open(output_tif, 'w', **profile) as dst:
            dst.write(data, 1)


# 使用函数
input_tif = 'F:/论文/论文基础/图片/Usable/NDVI_20220616.tif'
output_tif = 'F:/论文/论文基础/图片/Usable/NDVI.tif'
clamp_image(input_tif, output_tif)
