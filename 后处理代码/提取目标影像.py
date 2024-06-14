import os
import shutil

def copy_images(src_folder, dst_folder, txt_file):
    """
    从源文件夹复制特定的影像文件到目标文件夹。
    :param src_folder: 源文件夹路径。
    :param dst_folder: 目标文件夹路径。
    :param txt_file: 包含影像名称的txt文件路径。
    """
    # 确保目标文件夹存在
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # 读取txt文件中的影像名称
    with open(txt_file, 'r') as file:
        image_names = file.readlines()
    
    # 移除名称两端的空白字符（例如换行符）
    image_names = [name.strip() for name in image_names]

    count = 0
    all_images = 0
    # 遍历所有名称
    for name in image_names:
        # 构建源文件全路径
        src_path = os.path.join(src_folder, name)
        
        # 如果源文件存在，则复制
        if os.path.exists(src_path):
            count +=1
            shutil.copy(src_path, dst_folder)
        else:
            print(f"文件 {name} 未找到。")

        
    

# 使用示例
src_folder = r'D:\TEST\img_predict_out2'
dst_folder = r'D:\TEST\dst_img2'
txt_file = r'D:\TEST\name.txt'

copy_images(src_folder, dst_folder, txt_file)
