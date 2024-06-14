from osgeo import gdal
import os

inputdir = r'D:\TEST\dst_img'  # 请替换为您的文件夹路径
outputdir = r'D:\TEST\yasuo'  # 输出文件夹路径
filenames = os.listdir(inputdir)

for image_name in filenames:
    image_path = os.path.join(inputdir, image_name)
    ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    if ds is None:
        print(f"Failed to open file {image_name}")
        continue

    # 使用原始文件名构建输出文件的完整路径，但添加"_compressed"标识
    output_path = os.path.join(outputdir, image_name.replace('.tif', '_compressed.tif'))

    # 使用gdal的驱动创建新的TIFF文件
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.CreateCopy(output_path, ds, options=["COMPRESS=LZW"])

    # 清理
    del ds, dst_ds
