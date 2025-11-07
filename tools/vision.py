import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm  # 用于显示进度条


def overlay_segmentation(image, mask, color_map, alpha=0.5):
    """
    在原图上覆盖分割掩码
    """
    if len(mask.shape) == 3:
        mask = mask.squeeze()

    colored_mask = np.zeros_like(image)
    for class_id, color in color_map.items():
        colored_mask[mask == class_id] = color

    overlaid = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    return overlaid


def visualize_single_pair(original, label, prediction, color_map, alpha=0.5, save_dir=None, prefix=""):
    """
    可视化单组图像对，并保存结果
    """
    # 创建叠加图像
    label_overlay = overlay_segmentation(original.copy(), label, color_map, alpha)
    pred_overlay = overlay_segmentation(original.copy(), prediction, color_map, alpha)

    # 创建对比图像
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 第一行：原图、标注图、预测图
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(label, cmap='jet')
    axes[0, 1].set_title('Ground Truth Mask')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(prediction, cmap='jet')
    axes[0, 2].set_title('Prediction Mask')
    axes[0, 2].axis('off')

    # 第二行：原图、标注叠加、预测叠加
    axes[1, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Original Image')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(cv2.cvtColor(label_overlay, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'GT Overlay (alpha={alpha})')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(cv2.cvtColor(pred_overlay, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f'Pred Overlay (alpha={alpha})')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.close(fig)  # 不显示，只保存（批量处理时避免弹出过多窗口）

    # 保存结果
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存对比图
        comparison_path = save_dir / f'{prefix}_comparison.png'
        fig.savefig(comparison_path, dpi=200, bbox_inches='tight')

        # 保存叠加图
        label_overlay_path = save_dir / f'{prefix}_label_overlay.png'
        pred_overlay_path = save_dir / f'{prefix}_pred_overlay.png'

        cv2.imwrite(str(label_overlay_path), label_overlay)
        cv2.imwrite(str(pred_overlay_path), pred_overlay)

    return label_overlay, pred_overlay, comparison_path


def batch_visualize_segmentation(
        img_dir,  # 原图文件夹路径
        label_dir,  # 标注掩码文件夹路径
        pred_dir,  # 预测掩码文件夹路径
        output_root,  # 输出根目录
        color_map,  # 颜色映射
        alpha=0.5,
        img_suffix=['.jpg', '.png', '.jpeg'],  # 支持的原图后缀
        mask_suffix=['.png'],  # 支持的掩码后缀（通常是png，无损压缩）
        match_mode='name'  # 匹配模式：'name'（按文件名匹配）或 'index'（按文件顺序匹配）
):
    """
    批量可视化分割结果

    参数:
        img_dir: 原图文件夹
        label_dir: 标注掩码文件夹
        pred_dir: 预测掩码文件夹
        output_root: 输出根目录（所有结果会保存在这里）
        color_map: 颜色映射字典
        alpha: 透明度
        img_suffix: 原图支持的文件后缀
        mask_suffix: 掩码支持的文件后缀
        match_mode: 匹配模式 - 'name'（按文件名前缀匹配，推荐）或 'index'（按文件列表顺序匹配）
    """
    # 转换为Path对象
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)
    pred_dir = Path(pred_dir)
    output_root = Path(output_root)

    # 检查输入文件夹是否存在
    for dir_path in [img_dir, label_dir, pred_dir]:
        if not dir_path.exists():
            raise FileNotFoundError(f"文件夹不存在: {dir_path}")

    # 获取所有文件列表（过滤后缀）
    def get_file_list(dir_path, suffix_list):
        file_list = []
        for suffix in suffix_list:
            file_list.extend(list(dir_path.glob(f'*{suffix}')))
        return sorted(file_list)  # 排序保证顺序一致

    img_files = get_file_list(img_dir, img_suffix)
    label_files = get_file_list(label_dir, mask_suffix)
    pred_files = get_file_list(pred_dir, mask_suffix)

    # 检查文件数量
    print(f"找到 {len(img_files)} 张原图")
    print(f"找到 {len(label_files)} 个标注掩码")
    print(f"找到 {len(pred_files)} 个预测掩码")

    if match_mode == 'name':
        # 按文件名前缀匹配（推荐）
        # 示例：img.jpg -> label.png, pred.png（前缀相同）
        def get_prefix(file_path):
            return file_path.stem  # 获取文件名（不含后缀）

        # 创建文件名前缀到文件路径的映射
        img_prefix_map = {get_prefix(f): f for f in img_files}
        label_prefix_map = {get_prefix(f): f for f in label_files}
        pred_prefix_map = {get_prefix(f): f for f in pred_files}

        # 找到三个文件夹中共同的前缀（即可以匹配的文件对）
        common_prefixes = set(img_prefix_map.keys()) & set(label_prefix_map.keys()) & set(pred_prefix_map.keys())
        common_prefixes = sorted(list(common_prefixes))

        if not common_prefixes:
            raise ValueError("没有找到可以匹配的文件对！请检查文件名是否一致（不含后缀）")

        print(f"找到 {len(common_prefixes)} 组可匹配的文件对")

        # 批量处理
        for prefix in tqdm(common_prefixes, desc="批量处理进度"):
            img_path = img_prefix_map[prefix]
            label_path = label_prefix_map[prefix]
            pred_path = pred_prefix_map[prefix]

            # 读取图像
            original = cv2.imread(str(img_path))
            label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)

            # 检查读取是否成功
            if original is None:
                print(f"警告：无法读取原图 {img_path}，跳过该文件对")
                continue
            if label is None:
                print(f"警告：无法读取标注图 {label_path}，跳过该文件对")
                continue
            if pred is None:
                print(f"警告：无法读取预测图 {pred_path}，跳过该文件对")
                continue

            # 调整掩码尺寸
            if original.shape[:2] != label.shape:
                label = cv2.resize(label, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
            if original.shape[:2] != pred.shape:
                pred = cv2.resize(pred, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)

            # 可视化并保存
            visualize_single_pair(
                original=original,
                label=label,
                prediction=pred,
                color_map=color_map,
                alpha=alpha,
                save_dir=output_root,
                prefix=prefix
            )

    elif match_mode == 'index':
        # 按文件列表顺序匹配（需确保三个文件夹中文件顺序一致）
        min_count = min(len(img_files), len(label_files), len(pred_files))
        print(f"按顺序匹配，共处理 {min_count} 组文件")

        for i in tqdm(range(min_count), desc="批量处理进度"):
            img_path = img_files[i]
            label_path = label_files[i]
            pred_path = pred_files[i]

            # 读取图像
            original = cv2.imread(str(img_path))
            label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)

            # 检查读取是否成功
            if original is None:
                print(f"警告：无法读取原图 {img_path}，跳过该文件对")
                continue
            if label is None:
                print(f"警告：无法读取标注图 {label_path}，跳过该文件对")
                continue
            if pred is None:
                print(f"警告：无法读取预测图 {pred_path}，跳过该文件对")
                continue

            # 调整掩码尺寸
            if original.shape[:2] != label.shape:
                label = cv2.resize(label, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
            if original.shape[:2] != pred.shape:
                pred = cv2.resize(pred, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)

            # 使用原图文件名作为前缀
            prefix = img_path.stem
            # 可视化并保存
            visualize_single_pair(
                original=original,
                label=label,
                prediction=pred,
                color_map=color_map,
                alpha=alpha,
                save_dir=output_root,
                prefix=prefix
            )

    else:
        raise ValueError("match_mode 只能是 'name' 或 'index'")

    print(f"\n批量处理完成！所有结果已保存到：{output_root.absolute()}")


# 示例用法
if __name__ == "__main__":
    # ---------------------- 配置参数 ----------------------
    # 1. 颜色映射（根据你的类别修改）
    color_map = {
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
        10: (0, 128, 192),  # 类别4 - 黄色（可根据需要添加更多）
    }

    # 2. 文件夹路径配置（请根据实际情况修改）
    img_dir = "E:\DLearn\Dataset\CamVid\img\\test"  # 原图文件夹
    label_dir = "E:\DLearn\Dataset\CamVid\mask\\test"  # 标注掩码文件夹
    pred_dir = "E:\CARB\work_dirs\camvid_carb_dual\\vis_rgb\\test"  # 预测掩码文件夹
    output_root = "E:\CARB\\vis\\test"  # 批量输出根目录

    # 3. 其他参数
    alpha = 0.6  # 透明度
    match_mode = "name"  # 匹配模式：'name'（推荐）或 'index'
    img_suffix = ['.jpg', '.png']  # 支持的原图后缀
    mask_suffix = ['.png']  # 支持的掩码后缀
    # ------------------------------------------------------

    # 运行批量可视化
    batch_visualize_segmentation(
        img_dir=img_dir,
        label_dir=label_dir,
        pred_dir=pred_dir,
        output_root=output_root,
        color_map=color_map,
        alpha=alpha,
        img_suffix=img_suffix,
        mask_suffix=mask_suffix,
        match_mode=match_mode
    )
