from osgeo import gdal, ogr, osr
import numpy as np
import os
from shapely.geometry import Polygon
from shapely.ops import unary_union


def create_shapefile_from_tif_edge(tif_path, shp_path):
    # 打开TIF文件
    dataset = gdal.Open(tif_path)
    if not dataset:
        raise Exception("无法打开文件: " + tif_path)

    # 读取栅格数据
    band = dataset.GetRasterBand(1)
    arr = band.ReadAsArray()

    # 获取数据的有效区域
    mask = arr > 0  # 假设非零像素是我们关心的区域

    # 创建边界的多边形
    polygons = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                # 为每个像素创建一个小方块
                polygons.append(Polygon([
                    (j, i),
                    (j + 1, i),
                    (j + 1, i + 1),
                    (j, i + 1)
                ]))

    # 合并所有小方块
    boundary = unary_union(polygons).boundary

    # 创建SHP文件
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if driver is None:
        raise Exception("ESRI Shapefile driver not available.")

    # 如果文件已存在，则删除
    if os.path.exists(shp_path):
        driver.DeleteDataSource(shp_path)

    # 创建空间参考系统
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(dataset.GetProjectionRef())

    # 创建数据源和图层
    data_source = driver.CreateDataSource(shp_path)
    layer = data_source.CreateLayer(shp_path, spatial_ref, ogr.wkbLineString)

    # 创建几何特征
    feature_def = layer.GetLayerDefn()
    feature = ogr.Feature(feature_def)
    line = ogr.CreateGeometryFromWkb(boundary.wkb)
    feature.SetGeometry(line)
    layer.CreateFeature(feature)

    # 清理资源
    feature = None
    data_source = None
    dataset = None
    print("SHP文件已保存：", shp_path)


# 使用示例
tif_path = 'F:/论文/论文基础/图片/Usable/LST30.tif'
shp_path = 'F:/遥感影像/行政区划shp/安徽省shp/各个市shp/研究区/yanjiuqu.shp'
create_shapefile_from_tif_edge(tif_path, shp_path)