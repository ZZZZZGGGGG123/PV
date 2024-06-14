import os
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np

from unet import Unet_ONNX, Unet

if __name__ == "__main__":
    # 指定模式为批量预测
    mode = "dir_predict"
    
    # 指定待预测图片所在文件夹路径
    dir_origin_path = r"E:\code\unet-pytorch-main\test\img"
    
    # 指定保存预测结果的文件夹路径
    dir_save_path = r"E:\code\unet-pytorch-main\test\img_out"
    
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
    print(f"Total area for all images: {total_area}")