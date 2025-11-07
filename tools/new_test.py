import argparse
import os.path as osp
import mmcv
import torch
import numpy as np
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.core.evaluation import eval_metrics


class DictAction(argparse.Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options can
    be passed as comma separated values, i.e. 'KEY=V1,V2,V3', or with explicit
    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build
    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ['true', 'false']:
            return True if val.lower() == 'true' else False
        return val

    @staticmethod
    def _parse_iterable(val):
        """Parse iterable values in the string.
        All elements inside '()' or '[]' are treated as iterable values.
        Args:
            val (str): Value string.
        Returns:
            list | tuple: The expanded list or tuple from the string.
        Examples:
            >>> DictAction._parse_iterable('1,2,3')
            [1, 2, 3]
            >>> DictAction._parse_iterable('[a,b,c]')
            ['a', 'b', 'c']
            >>> DictAction._parse_iterable('[(1,2,3),[a,b],c]')
            [(1, 2, 3), ['a', 'b'], 'c']
        """

        def find_next_comma(string):
            """Find the position of next comma in the string.
            If no ',' is found in the string, return the string length.
            Inside nested brackets, commas are ignored.
            """
            assert (string.count('(') == string.count(')')) and (
                    string.count('[') == string.count(']')), \
                f'Imbalanced brackets exist in {string}'
            end = len(string)
            for i, c in enumerate(string):
                if c == '(' or c == '[':
                    stack = [c]
                    for j in range(i + 1, len(string)):
                        if string[j] == '(' or string[j] == '[':
                            stack.append(string[j])
                        elif string[j] == ')' or string[j] == ']':
                            stack.pop()
                        if len(stack) == 0:
                            break
                    if j + 1 < end and string[j + 1] == ',':
                        end = j + 1
                        break
                elif c == ',':
                    end = i
                    break
            return end

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            if comma_idx == len(val):
                next_val = val
                val = ''
            else:
                next_val = val[:comma_idx]
                val = val[comma_idx + 1:]
            # Remove redundant brackets
            if (next_val.startswith('(') and next_val.endswith(')')) or (
                    next_val.startswith('[') and next_val.endswith(']')):
                next_val = next_val[1:-1]
            if len(next_val) > 0:
                values.append(next_val)
        return values

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split('=', maxsplit=1)
            if '[' in val or '(' in val:
                val = self._parse_iterable(val)
            else:
                val = [self._parse_int_float_bool(v) for v in val.split(',')]
            if len(val) == 1:
                val = val[0]
            options[key] = val
        setattr(namespace, self.dest, options)


def simple_test_with_eval():
    parser = argparse.ArgumentParser(description='Test with evaluation but without visualization')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--eval', type=str, nargs='+',
                        help='evaluation metrics, e.g., "mIoU" or "mDice"')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--eval-options', nargs='+', action=DictAction,
                        help='custom options for evaluation')
    args = parser.parse_args()

    # 加载配置
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # 构建数据集和加载器
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # 构建模型
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # 设置类别和调色板
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
        print('使用数据集中的 CLASSES')

    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        model.PALETTE = dataset.PALETTE

    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    results = []
    gts = []

    print(f'开始处理 {len(data_loader)} 张图片...')

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        # 处理模型输出
        if isinstance(result, list):
            result = result[0]  # 取第一个结果

        # 确保结果是numpy数组
        if isinstance(result, torch.Tensor):
            result = result.cpu().numpy()

        # 检查数据中是否有GT标签
        if 'gt_semantic_seg' in data:
            gt_seg = data['gt_semantic_seg'][0]
            if isinstance(gt_seg, torch.Tensor):
                gt_seg = gt_seg.cpu().numpy()
            gts.append(gt_seg)
        else:
            # 如果没有GT标签，创建一个空的占位符
            print(f"警告: 第 {i} 个样本没有GT标签，跳过评估")
            # 创建一个与预测结果相同形状的占位符
            if len(result.shape) == 3:  # [C, H, W]
                gt_seg = np.zeros((1, result.shape[1], result.shape[2]), dtype=np.int64)
            else:  # [H, W]
                gt_seg = np.zeros(result.shape, dtype=np.int64)
            gts.append(gt_seg)

        # 保存结果
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f'已处理 {i + 1}/{len(data_loader)} 张图片')

    print(f'处理完成! 共处理 {len(results)} 张图片')

    # 保存结果
    if args.out:
        # 确保输出目录存在
        out_dir = osp.dirname(args.out)
        if out_dir:
            mmcv.mkdir_or_exist(out_dir)
        print(f'保存结果到 {args.out}')
        mmcv.dump(results, args.out)
        print('保存成功!')

    # 评估
    if args.eval:
        print(f'开始评估指标: {args.eval}')

        # 准备评估参数
        eval_kwargs = {}
        if args.eval_options:
            eval_kwargs.update(args.eval_options)

        # 获取类别数和忽略索引
        num_classes = len(dataset.CLASSES)
        ignore_index = getattr(dataset, 'ignore_index', 255)

        # 检查是否有有效的GT标签
        valid_gts = []
        valid_results = []
        for gt, result in zip(gts, results):
            # 如果GT不全为零，则认为它是有效的
            if np.any(gt != 0):
                valid_gts.append(gt)
                valid_results.append(result)

        if not valid_gts:
            print("警告: 没有找到有效的GT标签，无法进行评估")
            return

        print(f"使用 {len(valid_gts)} 个有GT标签的样本进行评估")

        # 计算指标
        metrics = eval_metrics(
            valid_results,
            valid_gts,
            num_classes,
            ignore_index,
            metrics=args.eval,
            nan_to_num=None,
            label_map=eval_kwargs.get('label_map', {}),
            reduce_zero_label=eval_kwargs.get('reduce_zero_label', False)
        )

        # 打印结果
        print('\n=== 评估结果 ===')
        for metric_name, metric_value in metrics.items():
            if metric_name == 'aAcc':
                print(f'整体准确率: {metric_value:.4f}')
            elif metric_name == 'mIoU':
                print(f'平均IoU: {metric_value:.4f}')
                # 打印各类别IoU
                if 'IoU' in metrics:
                    print('各类别IoU:')
                    for i, iou in enumerate(metrics['IoU']):
                        class_name = dataset.CLASSES[i] if i < len(dataset.CLASSES) else f'Class_{i}'
                        print(f'  {class_name}: {iou:.4f}')
            elif metric_name == 'mAcc':
                print(f'平均准确率: {metric_value:.4f}')
                # 打印各类别准确率
                if 'Acc' in metrics:
                    print('各类别准确率:')
                    for i, acc in enumerate(metrics['Acc']):
                        class_name = dataset.CLASSES[i] if i < len(dataset.CLASSES) else f'Class_{i}'
                        print(f'  {class_name}: {acc:.4f}')
            elif metric_name == 'mDice':
                print(f'平均Dice系数: {metric_value:.4f}')
            else:
                print(f'{metric_name}: {metric_value:.4f}')


if __name__ == '__main__':
    simple_test_with_eval()