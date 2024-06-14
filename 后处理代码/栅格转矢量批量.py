from osgeo import gdal, ogr, osr
import os

def raster_to_vector(raster_input, layer):
    raster_ds = gdal.Open(raster_input, gdal.GA_ReadOnly)
    if not raster_ds:
        print(f"无法打开文件 {raster_input}")
        return

    band = raster_ds.GetRasterBand(1)
    mask_band = gdal.GetDriverByName('MEM').Create('', raster_ds.RasterXSize, raster_ds.RasterYSize, 1, gdal.GDT_Byte)
    mask_band_array = band.ReadAsArray()
    mask_band_array[mask_band_array == 255] = 1
    mask_band_array[mask_band_array != 1] = 0
    mask_band.GetRasterBand(1).WriteArray(mask_band_array)
    
    gdal.Polygonize(band, mask_band.GetRasterBand(1), layer, 0)  #第四个参数代表把栅格值写入

def batch_raster_to_vector(input_folder, output_shapefile):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(output_shapefile):
        driver.DeleteDataSource(output_shapefile)
    vector_ds = driver.CreateDataSource(output_shapefile)
    srs = osr.SpatialReference()
    # 注意：这里需要指定合适的EPSG代码
    srs.ImportFromEPSG(3857)  # 假设使用WGS84坐标系
    layer = vector_ds.CreateLayer('raster_vector', srs=srs,)
    id_field = ogr.FieldDefn("ID", ogr.OFTInteger)
    layer.CreateField(id_field)

    for filename in os.listdir(input_folder):
        if filename.endswith(".tif"):
            raster_input = os.path.join(input_folder, filename)
            print(f"Processing {raster_input}")
            raster_to_vector(raster_input, layer)
    
    vector_ds = None  # 保存并关闭Shapefile

# 示例用法
input_folder = r'D:\TEST\dst_img2'
output_shapefile = r'D:\TEST3\3'
batch_raster_to_vector(input_folder, output_shapefile)
