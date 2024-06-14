from osgeo import gdal
import numpy as np
import os

def crop_tif_image(input_file, output_folder, crop_size=512):
    dataset = gdal.Open(input_file)
    if dataset is None:
        print(f"文件{input_file}无法打开")
        return
    
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount
    
    #获取地理坐标和投影
    geotrans = dataset.GetGeoTransform()
    proj = dataset.GetProjection()

    #处理边缘部分，如果边缘部分不足裁剪尺寸，rows和cols都要加1
    rows = height // crop_size + (1 if height % crop_size > 0 else 0)
    cols = width // crop_size + (1 if height % crop_size > 0 else 0)
    
    for i in range(rows):
        #print(f"第{i}行")
        for j in range(cols):
            x_off = j * crop_size
            y_off = i * crop_size
            #print("x_off",x_off)
            #print("y_off",y_off)
            if x_off + crop_size > width and y_off + crop_size < height:
  
                 width = (width % crop_size)

                 data = dataset.ReadAsArray(x_off, y_off, width, crop_size)
             

            elif y_off + crop_size > height and x_off + crop_size < width:

                 height = (height % crop_size)
              
                 
                 data = dataset.ReadAsArray(x_off, y_off, crop_size, height)
                 
            elif x_off + crop_size > width and y_off + crop_size > height:
                 height =  (height % crop_size)
                 width =  (width % crop_size)
            
                 data = dataset.ReadAsArray(x_off, y_off, width, height)
                 
            else:
                data = dataset.ReadAsArray(x_off, y_off, crop_size, crop_size)
                
            new_geotrans = list(geotrans)
            new_geotrans[0] = geotrans[0] + x_off * geotrans[1]
            new_geotrans[3] = geotrans[3] + y_off * geotrans[5]
            print(f"新坐标{i}{j}的投影信息为",new_geotrans[0],new_geotrans[3])
            output_file = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_{i}_{j}.tif")
            write_tif_image(data, output_file, new_geotrans, proj , crop_size, bands)

#批量裁剪，文件夹的方式
def batch_crop_tif_images(input_folder, output_folder, crop_size=512):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入文件夹中的所有.tif文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):
            input_file = os.path.join(input_folder, filename)
            print(f"正在处理文件：{input_file}")
            crop_tif_image(input_file, output_folder, crop_size)

def write_tif_image(im_data, output_file, geotrans, proj , crop_size, bands):
    # 根据im_data的数据类型选择GDAL数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
        #print('int8')
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
        print('int16')
    else:
        datatype = gdal.GDT_Float32
        print('Float32')
    
    # 确定波段数量
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    
    driver = gdal.GetDriverByName("GTiff")
    
    dataset = driver.Create(output_file, int(im_width), int(im_height), int(im_bands), datatype)
    # 更新地理变换参数
    dataset.SetGeoTransform(geotrans)
    dataset.SetProjection(proj)

    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset
        

# 示例用法
input_file = r"D:\ceshixiazai3"
output_folder = r"D:\TEST\img_crop_out"
batch_crop_tif_images(input_file, output_folder)
