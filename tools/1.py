import os


def batch_remove_mid_suffix(folder_path, target="_L", target_ext=None):
    """
    批量删除文件名中「后缀之前」的指定字符（默认删除 _L）
    :param folder_path: 目标文件夹路径
    :param target: 要删除的字符（默认 _L）
    :param target_ext: 仅处理指定后缀的文件（如 ".png"，默认 None 处理所有后缀）
    """
    # 验证文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 {folder_path} 不存在！")
        return

    # 遍历文件夹中的所有文件（不递归子文件夹，如需递归见扩展）
    for filename in os.listdir(folder_path):
        file_full_path = os.path.join(folder_path, filename)

        # 只处理文件，跳过文件夹
        if os.path.isfile(file_full_path):
            # 拆分文件名和后缀（如 "photo_L.png" → ("photo_L", ".png")）
            file_name_no_ext, file_ext = os.path.splitext(filename)

            # 筛选条件：1. 文件名部分以 target（_L）结尾；2. 可选：仅处理指定后缀（如 .png）
            if file_name_no_ext.endswith(target):
                # 如果指定了后缀（如 .png），则只处理该后缀的文件
                if target_ext is None or file_ext.lower() == target_ext.lower():
                    # 生成新文件名：去掉文件名末尾的 target，再拼接后缀
                    new_file_name_no_ext = file_name_no_ext[:-len(target)]
                    new_filename = new_file_name_no_ext + file_ext
                    new_file_full_path = os.path.join(folder_path, new_filename)

                    # 执行重命名并捕获异常
                    try:
                        os.rename(file_full_path, new_file_full_path)
                        print(f"成功：{filename} → {new_filename}")
                    except Exception as e:
                        print(f"失败：{filename} → 原因：{str(e)}")

    print("\n批量处理完成！")


# -------------------------- 请修改这里的配置 --------------------------
# 1. 目标文件夹路径（参考注释示例）
target_folder = "请替换为你的图片文件夹路径"  # 如：Windows→r"C:\Users\XXX\Pictures"；mac→"/Users/XXX/Pictures"

# 2. 要删除的字符（默认 _L，无需修改）
delete_str = "_L"

# 3. 仅处理指定后缀（如 .png，如需处理所有文件则设为 None）
target_extension = ".png"  # 只处理 .png 文件；改为 None 处理所有后缀（如 .jpg/.gif 等）

# -------------------------- 执行脚本 --------------------------
batch_remove_mid_suffix(
    folder_path=target_folder,
    target=delete_str,
    target_ext=target_extension
)