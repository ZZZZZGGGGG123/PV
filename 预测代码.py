from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
from unet import Unet_ONNX, Unet
from osgeo import gdal
import os
from tqdm import tqdm
from PIL import Image

from unet import Unet_ONNX, Unet
# ... [此处为您的裁剪代码，无需修改] ...
from osgeo import gdal
import numpy as np

if __name__ == "__main__":
    # 指定模式为批量预测
    mode = "dir_predict"
    
    # 指定待预测图片所在文件夹路径
    dir_origin_path = r"D:\TEST2\img_crop_out"
    
    # 指定保存预测结果的文件夹路径
    dir_save_path = r"D:\TEST2\img_predict_out"
    
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
            r_image, area = unet.detect_image(image,img_name, count=True,name_classes=name_classes)  # 获取返回的面积
            total_area += area
            # 确保保存结果的文件夹存在
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)

            ds = gdal.Open(image_path)
            geotrans = ds.GetGeoTransform()
            proj = ds.GetProjection()

            # 2. 将PIL.Image转换为NumPy数组
            r_image_array = np.array(r_image)
            print()
            # 3. 创建一个新的GDAL数据集（文件），用于保存带有地理信息的预测结果
            bands_count = ds.RasterCount  # 获取原始图像的波段数
            print("bands_count", bands_count)
            driver = gdal.GetDriverByName('GTiff')

            out_ds = driver.Create(os.path.join(dir_save_path, img_name), r_image_array.shape[1], r_image_array.shape[0], bands_count, eType=gdal.GDT_Byte)
            out_ds.SetGeoTransform(geotrans)
            out_ds.SetProjection(proj)
            print('##',out_ds.GetRasterBand(1))
             # 写入预测图像数据
            for b in range(1, bands_count + 1):
                # 获取原始数据集的波段数据
                band_data = ds.GetRasterBand(b).ReadAsArray()
                # 将预测结果写入对应波段
                out_band = out_ds.GetRasterBand(b)
                out_band.WriteArray(r_image_array[:,:,b-1])  # 假设预测结果存储在第一个波段
            
            # 清理和关闭数据集
            out_ds.FlushCache()
            out_ds = None


            # 保存预测结果，文件名保持不变
            # r_image.save(os.path.join(dir_save_path, img_name))
    print(f"\n\n\nTotal area for all images: {total_area}")

    