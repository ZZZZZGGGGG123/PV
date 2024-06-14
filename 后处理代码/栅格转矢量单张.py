from osgeo import gdal, ogr, osr
import numpy as np
import os

def raster_to_vector(raster_input, vector_output):
    # 打开栅格数据集
    raster_ds = gdal.Open(raster_input)
    band = raster_ds.GetRasterBand(1)

    # 创建掩模波段，仅包含值为255的区域
    mask_band = gdal.GetDriverByName('MEM').Create('', raster_ds.RasterXSize, raster_ds.RasterYSize, 1, gdal.GDT_Byte)
    mask_band_array = band.ReadAsArray()
    # 将目标区域设置为1，背景区域设置为0
    mask_band_array[mask_band_array == 255] = 1
    mask_band_array[mask_band_array != 1] = 0
    mask_band.GetRasterBand(1).WriteArray(mask_band_array)

    # 创建输出矢量数据集（Shapefile）
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(vector_output):
        driver.DeleteDataSource(vector_output)
    vector_ds = driver.CreateDataSource(vector_output)  #指定保存位置
    srs = osr.SpatialReference()
    srs.ImportFromWkt(raster_ds.GetProjectionRef())
    layer = vector_ds.CreateLayer("layer_name", srs=srs)

    # 创建ID字段
    id_field = ogr.FieldDefn("label", ogr.OFTInteger)
    layer.CreateField(id_field)

    # 使用掩模波段进行栅格转矢量
    gdal.Polygonize(band, mask_band.GetRasterBand(1), layer, 0)

    # 清理
    vector_ds.Destroy()
    raster_ds = None
    mask_band = None

# 示例用法
raster_input = r'D:\TEST3\1\0_0_0.tif'
vector_output = r'D:\TEST3\3'
raster_to_vector(raster_input, vector_output)
