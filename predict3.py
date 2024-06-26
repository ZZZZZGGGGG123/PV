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
    
 
'''
滑动窗口裁剪函数
TifPath 影像路径
SavePath 裁剪后保存目录
CropSize 裁剪尺寸
RepetitionRate 重复率
'''
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
    img = dataset_img.ReadAsArray(0, 0, width, height)#获取数据
    
    #  获取当前文件夹的文件个数len,并以len+1命名即将裁剪得到的图像
    new_name = len(os.listdir(SavePath)) + 1
    #  裁剪图片,重复率为RepetitionRate
    
    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
            #  如果图像是单波段
            if(len(img.shape) == 2):
                cropped = img[int(i * CropSize * (1 - RepetitionRate)) : int(i * CropSize * (1 - RepetitionRate)) + CropSize, 
                              int(j * CropSize * (1 - RepetitionRate)) : int(j * CropSize * (1 - RepetitionRate)) + CropSize]
            #  如果图像是多波段
            else:
                cropped = img[:,
                              int(i * CropSize * (1 - RepetitionRate)) : int(i * CropSize * (1 - RepetitionRate)) + CropSize, 
                              int(j * CropSize * (1 - RepetitionRate)) : int(j * CropSize * (1 - RepetitionRate)) + CropSize]
            #  写图像
            writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif"%new_name)
            #  文件名 + 1
            new_name = new_name + 1
    #  向前裁剪最后一列
    for i in range(int((height-CropSize*RepetitionRate)/(CropSize*(1-RepetitionRate)))):
        if(len(img.shape) == 2):
            cropped = img[int(i * CropSize * (1 - RepetitionRate)) : int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          (width - CropSize) : width]
        else:
            cropped = img[:,
                          int(i * CropSize * (1 - RepetitionRate)) : int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          (width - CropSize) : width]
        #  写图像
        writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif"%new_name)
        new_name = new_name + 1
    #  向前裁剪最后一行
    for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        if(len(img.shape) == 2):
            cropped = img[(height - CropSize) : height,
                          int(j * CropSize * (1 - RepetitionRate)) : int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        else:
            cropped = img[:,
                          (height - CropSize) : height,
                          int(j * CropSize * (1 - RepetitionRate)) : int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif"%new_name)
        #  文件名 + 1
        new_name = new_name + 1
    #  裁剪右下角
    if(len(img.shape) == 2):
        cropped = img[(height - CropSize) : height,
                      (width - CropSize) : width]
    else:
        cropped = img[:,
                      (height - CropSize) : height,
                      (width - CropSize) : width]
    writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif"%new_name)
    new_name = new_name + 1
     

# 执行批量裁剪
input_folder = r"E:\code\unet-pytorch-main\test\img"
crop_output_folder = r"E:\code\unet-pytorch-main\test\img_out"
BatchTifCrop(input_folder, crop_output_folder, 512, 0)

# 设置裁剪结果作为预测的输入
dir_origin_path = crop_output_folder



if __name__ == "__main__":
    # 指定模式为批量预测
    mode = "dir_predict"
    
    # 指定待预测图片所在文件夹路径
    dir_origin_path = r"E:\code\unet-pytorch-main\test\img_out"
    
    # 指定保存预测结果的文件夹路径
    dir_save_path = r"E:\code\unet-pytorch-main\test\img_prdict_out2"
    
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


