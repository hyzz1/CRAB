import os
import numpy as np
from PIL import Image

# 定义 RGB 颜色到类别 ID 的映射（CamVid 11 类）
COLOR2ID = {
    (128, 128, 128): 0,  # Sky
    (128,   0,   0): 1,  # Building
    (192, 192, 128): 2,  # Pole
    (128,  64, 128): 3,  # Road
    ( 0,  0, 192): 4,  # Sidewalk
    (128, 128,   0): 5,  # Tree
    (192, 128, 128): 6,  # SignSymbol
    ( 64,  64, 128): 7,  # Fence
    ( 64,   0, 128): 8,  # Car
    ( 64,  64,   0): 9,  # Pedestrian
    (  0, 128, 192):10,  # Bicyclist
}
IGNORE_INDEX = 255

# 将 RGB 标签转换为类别索引
def rgb2index(rgb_img):
    h, w = rgb_img.shape[:2]
    out = np.full((h, w), IGNORE_INDEX, dtype=np.uint8)
    rgb = rgb_img.reshape(-1, 3)
    out_flat = out.reshape(-1)

    # 向量化映射
    color_arr = np.array(list(COLOR2ID.keys()), dtype=np.uint8)
    id_arr    = np.array(list(COLOR2ID.values()), dtype=np.uint8)
    # 为了速度，做个哈希映射
    key = rgb[:,0].astype(np.uint32) << 16 | rgb[:,1].astype(np.uint32) << 8 | rgb[:,2].astype(np.uint32)
    color_key = color_arr[:,0].astype(np.uint32) << 16 | color_arr[:,1].astype(np.uint32) << 8 | color_arr[:,2].astype(np.uint32)

    for k, cid in zip(color_key, id_arr):
        out_flat[key == k] = cid
    return out

# 批量转换目录中的所有图像
def convert_dir(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(in_dir):
        if not fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            continue
        p = os.path.join(in_dir, fn)
        img = Image.open(p).convert('RGB')  # 强制三通道
        arr = np.array(img)
        index = rgb2index(arr)
        Image.fromarray(index, mode='L').save(os.path.join(out_dir, os.path.splitext(fn)[0] + '.png'))

if __name__ == '__main__':
    # 替换为你的实际路径
    root = r"E:\DLearn\Dataset\CamVid"
    for split in ['test']:
        convert_dir(os.path.join(root, 'mask', split),
                    os.path.join(root, 'mask_idx', split))
    print('Done. 索引标签已输出到 mask_idx/{train,val}')
