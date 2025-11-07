import os
import numpy as np
from PIL import Image

# 定义类别ID到RGB颜色的映射（CamVid 11类）
ID2COLOR = {
    0: (128, 128, 128),  # Sky
    1: (128, 0, 0),  # Building
    2: (192, 192, 128),  # Pole
    3: (128, 64, 128),  # Road
    4: (0, 0, 192),  # Pavement
    5: (128, 128, 0),  # Tree
    6: (192, 128, 128),  # SignSymbol
    7: (64, 64, 128),  # Fence
    8: (64, 0, 128),  # Car
    9: (64, 64, 0),  # Pedestrian
    10: (0, 128, 192),  # Bicyclist
    255: (0, 0, 0)  # 忽略的索引用黑色表示
}


# 将类别索引转换回 RGB 标签
def index2rgb(index_img):
    h, w = index_img.shape
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)

    # 为每个类别ID设置对应的RGB颜色
    for class_id, color in ID2COLOR.items():
        mask = (index_img == class_id)
        rgb_img[mask] = color

    return rgb_img


# 批量转换目录中的所有图像（索引转RGB）
def convert_index_to_rgb(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(in_dir):
        if not fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            continue
        p = os.path.join(in_dir, fn)
        img = Image.open(p)
        arr = np.array(img)
        rgb = index2rgb(arr)

        # 保存为PNG格式以保持质量
        output_fn = os.path.splitext(fn)[0] + '.png'
        Image.fromarray(rgb, mode='RGB').save(os.path.join(out_dir, output_fn))
        print(f'转换: {fn} -> {output_fn}')


if __name__ == '__main__':
    # 设置输入和输出路径
    input_dir = r"E:\CARB\work_dirs\camvid_carb_dual\vis\test"
    output_dir = r"E:\CARB\work_dirs\camvid_carb_dual\vis_rgb\test"

    # 执行转换
    convert_index_to_rgb(input_dir, output_dir)
    print(f'完成! 所有图像已从索引转换回RGB，保存在: {output_dir}')