from PIL import Image
import os

'''
对12个月的地表温度进行拼接
'''
def combine_images_in_folder(folder_path, output_path):
    # 定义月份排序
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]

    # 根据月份排序找到文件路径
    filepaths = [os.path.join(folder_path, f"{month}.tif") for month in months if
                 os.path.exists(os.path.join(folder_path, f"{month}.tif"))]

    # 确保找到了十二个图像
    if len(filepaths) != 12:
        raise ValueError("文件夹中应该有12个月的tif图像文件。")

    # 打开图像
    images = [Image.open(fp) for fp in filepaths]

    # 获取最大的宽度和高度以确定单个图像的目标尺寸
    max_width = max(img.size[0] for img in images)
    max_height = max(img.size[1] for img in images)

    # 计算总的图像尺寸（三行四列）
    total_width = max_width * 4
    total_height = max_height * 3

    # 创建一个新的图像以放置所有图像
    new_img = Image.new('RGB', (total_width, total_height))

    # 按行列顺序粘贴图像
    for index, img in enumerate(images):
        # 计算每个图像的粘贴位置
        x = index % 4 * max_width
        y = index // 4 * max_height
        new_img.paste(img, (x, y))

    # 保存为PNG文件
    new_img.save(output_path, "PNG")


# 示例使用
folder_path = 'F:/论文/论文基础/图片/LSTusable'  # 替换为你的文件夹路径
output_path = 'F:/论文/论文基础/图片/LST12.png'  # 输出文件名
combine_images_in_folder(folder_path, output_path)
