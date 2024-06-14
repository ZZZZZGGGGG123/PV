from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
from osgeo import gdal
import os
from tqdm import tqdm
from PIL import Image



def stitch_images(predicted_folder, output_path, original_folder):
    images_dict = {}
    
    # 读取预测的小图像并将其添加到字典中
    filenames = os.listdir(predicted_folder) #获取所有子文件
    for fname in filenames:
        if fname.lower().endswith('.tif'):
            base_name, i, j = fname.rsplit('.', 1)[0].rsplit('_', 2) #根据'.'和'_'进行分割
            i, j = int(i), int(j)
            if base_name not in images_dict:
                images_dict[base_name] = {}
            ds = gdal.Open(os.path.join(predicted_folder, fname))
            bands = ds.RasterCount
            #存储波段信息
            arrs = [ds.GetRasterBand(b+1).ReadAsArray() for b in range(bands)]
            #images_dict 是一个字典，它的键是基本文件名 base_name，对应的值是另一个字典。在这个内部字典中，键是位置 (i, j)，对应的值是一个列表，包含了所有波段的数据数组。
            images_dict[base_name][(i, j)] = arrs


    # 拼接图像并赋予地理坐标和投影信息
    for base_name, slices in images_dict.items():
        # 从原始文件夹中为每个拼接后的图像获取地理信息
        original_file_path = os.path.join(original_folder, f"{base_name}.tif")
        original_ds = gdal.Open(original_file_path)
        if original_ds is None:
            print(f"Failed to open original file at {original_file_path}")
            continue  # 如果无法打开原始图像，跳过这个图像

        # 获取原始图像的地理坐标和投影信息
        original_geotrans = original_ds.GetGeoTransform()
        original_proj = original_ds.GetProjection()

    # 拼接图像
    for base_name, slices in images_dict.items():
        max_i = max([i for i, j in slices.keys()])+1
        max_j = max([j for i, j in slices.keys()])+1
        #构建第一张图像的路径
        first_image_path = os.path.join(predicted_folder, f"{base_name}_1_1.tif")
        first_image = gdal.Open(first_image_path)
        tile_width = first_image.RasterXSize
        tile_height = first_image.RasterYSize
        bands = first_image.RasterCount
        stitched_images = [np.zeros((max_i * tile_height, max_j * tile_width), dtype=np.uint8) for b in range(bands)]
        for arr in stitched_images:
            print("arr.shape",arr.shape)
        for (i, j), arrs in slices.items():
            print(f"Shape of image at position ({i}, {j}): {arrs[0].shape}")
            for b in range(bands):
                print("arrs[b]",arrs[b])
                print("i",i)
                print("j",j)
                print("tile_height",tile_height)
                print("tile_width",tile_width)
                print("被赋形状：", stitched_images[b][(i)*tile_height:(i+1)*tile_height, j*tile_width:(j+1)*tile_width].shape)
                
                stitched_images[b][i*tile_height:(i+1)*tile_height, j*tile_width:(j+1)*tile_width] = arrs[b]

        # 保存拼接后的图像
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(os.path.join(output_path, f"{base_name}_stitched.tif"), stitched_images[0].shape[1], stitched_images[0].shape[0], bands, gdal.GDT_Byte)
        
        # 使用 original_geotrans 和 original_proj 为当前拼接后的图像设置地理信息
        if original_geotrans is not None and original_proj is not None:
            out_ds.SetGeoTransform(original_geotrans)
            out_ds.SetProjection(original_proj)
        else:
            print("Warning: Could not set GeoTransform and Projection for the output image.")
        
        # 写入拼接图像数据
        for b in range(bands):
            out_band = out_ds.GetRasterBand(b+1)
            out_band.WriteArray(stitched_images[b])
        
        out_ds.FlushCache()
        out_ds = None


if __name__ == "__main__":
    original_folder = r"D:\TEST\img"
    dir_save_path = r"D:\TEST\img_crop_out"
    stitched_output_folder = r"D:\TEST\img_predict_stitch"
    stitch_images(dir_save_path, stitched_output_folder, original_folder)