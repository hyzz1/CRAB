1. 所需具体环境见requirements.txt（我的虚拟环境所安装的库）
    mmcv-full  1.n.0版本与mmcv 2.n版本不同，存在一些函数命名和使用方法不同，若使用2.n则会出现一些列问题
2. 配置环境时遇到的一些问题，简单问题不记录
    （1）AttributeError:'NoneType' object has no attribute 'shape' 已解决，代码已修改。在配置文件中修改见注释
        提取图片时失效，图片未能正确读取
    
    （2）一些难以安装的库，已上传预编译版
    
    （3）IndexError: too many indices for tensor of dimension 2。 此处问题在与 pred_label（预测标签张量）和 
        mask（有效像素掩码）的维度 不匹配：pred_label 是 2 维张量（如 [H, W]），但 mask 是更高维度（如 [N, H, W]）
        或维度不兼容的张量，导致索引时出现 “维度过多” 的错误。
        此处问题出现在预测阶段，不要修改mmseg中的代码！！！。数据集维度与掩码不同，预测过程中需要单通道的标注而数据集标注为rgb格式，
        我在tools中给出单通道与rgb格式相互转换的代码(trans_to_HW,trans_to_RGB)。在训练与预测使用单通道，可视化将预测结果转化为rgb格式即可。'
    
    （4）严格使用给出的环境即可轻松复现，本次复现使用Camvid数据集，
    
    （5）在执行训练代码前，需要优先加载当前环境的mmseg库，使用以下命令Set PYTHONPATH=E:你的项目路径\CARB;%PYTHONPATH%
    （6）提前安装好c++编译器，可以节省时间
3. 操作方法我在README中修改