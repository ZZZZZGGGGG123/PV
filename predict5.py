from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np

from unet import Unet_ONNX, Unet
from osgeo import gdal
import os
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np

from unet import Unet_ONNX, Unet
# ... [此处为您的裁剪代码，无需修改] ...
import os
from osgeo import gdal
import numpy as np
 
#  读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    return dataset
    
#  保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
        dataset.SetProjection(im_proj) #写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset
    
#  保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
        dataset.SetProjection(im_proj) #写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

# ... [其他导入] ...
def BatchTifCrop(InputFolder, SavePath, CropSize, RepetitionRate):
    """
    批量裁剪Tiff图像
    InputFolder: 输入的包含多张遥感图像的文件夹
    SavePath: 裁剪后保存的目录
    CropSize: 裁剪尺寸
    RepetitionRate: 重复率
    """
    # 列出目录中所有的tif文件
    tif_files = [f for f in os.listdir(InputFolder) if f.endswith('.tif')]

    # 为每个tif文件执行裁剪
    for tif_file in tif_files:
        TifCrop(os.path.join(InputFolder, tif_file), SavePath, CropSize, RepetitionRate)


def TifCrop(TifPath, SavePath, CropSize, RepetitionRate):
    dataset_img = readTif(TifPath)
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
    proj = dataset_img.GetProjection()
    geotrans = dataset_img.GetGeoTransform()
    img = dataset_img.ReadAsArray(0, 0, width, height)
    
    # 获取原始图像文件名(不带扩展名)
    base_name = os.path.splitext(os.path.basename(TifPath))[0]
    
    # 按照新的命名约定来裁剪图像
    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
            if(len(img.shape) == 2):
                cropped = img[int(i * CropSize * (1 - RepetitionRate)) : int(i * CropSize * (1 - RepetitionRate)) + CropSize, 
                              int(j * CropSize * (1 - RepetitionRate)) : int(j * CropSize * (1 - RepetitionRate)) + CropSize]
            else:
                cropped = img[:,
                              int(i * CropSize * (1 - RepetitionRate)) : int(i * CropSize * (1 - RepetitionRate)) + CropSize, 
                              int(j * CropSize * (1 - RepetitionRate)) : int(j * CropSize * (1 - RepetitionRate)) + CropSize]
            output_filename = f"{base_name}_{i+1}_{j+1}.tif"
            writeTiff(cropped, geotrans, proj, os.path.join(SavePath, output_filename))
    
    # 向前裁剪最后一列
    for i in range(int((height-CropSize*RepetitionRate)/(CropSize*(1-RepetitionRate)))):
        if(len(img.shape) == 2):
            cropped = img[int(i * CropSize * (1 - RepetitionRate)) : int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          (width - CropSize) : width]
        else:
            cropped = img[:,
                          int(i * CropSize * (1 - RepetitionRate)) : int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          (width - CropSize) : width]
        output_filename = f"{base_name}_{i+1}_{width // CropSize + 1}.tif"
        writeTiff(cropped, geotrans, proj, os.path.join(SavePath, output_filename))
    
    # 向前裁剪最后一行
    for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        if(len(img.shape) == 2):
            cropped = img[(height - CropSize) : height,
                          int(j * CropSize * (1 - RepetitionRate)) : int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        else:
            cropped = img[:,
                          (height - CropSize) : height,
                          int(j * CropSize * (1 - RepetitionRate)) : int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        output_filename = f"{base_name}_{height // CropSize + 1}_{j+1}.tif"
        writeTiff(cropped, geotrans, proj, os.path.join(SavePath, output_filename))
    
    # 裁剪右下角
    if(len(img.shape) == 2):
        cropped = img[(height - CropSize) : height,
                      (width - CropSize) : width]
    else:
        cropped = img[:,
                      (height - CropSize) : height,
                      (width - CropSize) : width]
    output_filename = f"{base_name}_{height // CropSize + 1}_{width // CropSize + 1}.tif"
    writeTiff(cropped, geotrans, proj, os.path.join(SavePath, output_filename))
    
# ... [其余的代码，例如BatchTifCrop和主程序] ...

def stitch_images(predicted_folder, output_path):
    # 获取预测文件夹中的所有文件名
    filenames = os.listdir(predicted_folder)
    
    # 使用一个字典来存储每个大图像的名字和其对应的小图像
    images_dict = {}
    
    for fname in filenames:
        if fname.lower().endswith('.tif'):
            base_name, i, j = fname.rsplit('.', 1)[0].rsplit('_', 2)
            i, j = int(i), int(j)
            
            if base_name not in images_dict:
                images_dict[base_name] = {}
            
            # 使用GDAL读取图像数据
            ds = gdal.Open(os.path.join(predicted_folder, fname))
            bands = ds.RasterCount
            arrs = [ds.GetRasterBand(b+1).ReadAsArray() for b in range(bands)]
            images_dict[base_name][(i, j)] = arrs
    
    # 遍历每个大图像并将其对应的小图像拼接起来
    for base_name, slices in images_dict.items():
        # 获取大图像的尺寸
        max_i = max([i for i, j in slices.keys()])
        max_j = max([j for i, j in slices.keys()])
        
        first_image_path = os.path.join(predicted_folder, f"{base_name}_1_1.tif")
        first_image = gdal.Open(first_image_path)
        tile_width = first_image.RasterXSize
        tile_height = first_image.RasterYSize
        bands = first_image.RasterCount
        
        # 创建一个大数组来存储拼接后的图像
        stitched_images = [np.zeros((max_i * tile_height, max_j * tile_width), dtype=np.uint8) for b in range(bands)]
        
        for (i, j), arrs in slices.items():
            for b in range(bands):
                stitched_images[b][(i-1)*tile_height:i*tile_height, (j-1)*tile_width:j*tile_width] = arrs[b]
        
        # 使用GDAL将拼接后的数组保存为TIF文件
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(os.path.join(output_path, f"{base_name}_stitched.tif"), stitched_images[0].shape[1], stitched_images[0].shape[0], bands, gdal.GDT_Byte)
        out_ds.SetGeoTransform(first_image.GetGeoTransform())  # 使用第一个小图像的GeoTransform
        out_ds.SetProjection(first_image.GetProjection())      # 使用第一个小图像的Projection
        
        for b in range(bands):
            out_band = out_ds.GetRasterBand(b+1)
            out_band.WriteArray(stitched_images[b])
        
        out_ds.FlushCache()
        out_ds = None 

# 在主程序中调用上述函数进行拼接

     

# 执行批量裁剪
input_folder = r"E:\code\unet-pytorch-main\test2\img"
crop_output_folder = r"E:\code\unet-pytorch-main\test2\img_crop_out"
BatchTifCrop(input_folder, crop_output_folder, 512, 0)

# 设置裁剪结果作为预测的输入
dir_origin_path = crop_output_folder



if __name__ == "__main__":
    # 指定模式为批量预测
    mode = "dir_predict"
    
    # 指定待预测图片所在文件夹路径
    dir_origin_path = r"E:\code\unet-pytorch-main\test2\img_crop_out"
    
    # 指定保存预测结果的文件夹路径
    dir_save_path = r"E:\code\unet-pytorch-main\test2\img_predict_out"
    
    # 初始化模型
    unet = Unet()
    name_classes = ["background", "PV"]
    # 获取待预测图片文件列表
    img_names = os.listdir(dir_origin_path)
    total_area = 0  # 添加这一行来累积面积
    # 遍历文件夹中的图片并进行预测
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path = os.path.join(dir_origin_path, img_name)
            image = Image.open(image_path)
            
            # 进行预测
            r_image, area = unet.detect_image(image, count=True,name_classes=name_classes)  # 获取返回的面积
            total_area += area
            # 确保保存结果的文件夹存在
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            
            # 保存预测结果，文件名保持不变
            r_image.save(os.path.join(dir_save_path, img_name))
    print(f"\n\n\nTotal area for all images: {total_area}")

    stitched_output_folder = r"E:\code\unet-pytorch-main\test2\img_predict_stitch"
    stitch_images(dir_save_path, stitched_output_folder)


