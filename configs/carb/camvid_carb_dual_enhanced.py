"""
Enhanced CamVid configuration with improved pseudo-mask quality.

This configuration demonstrates how to use the pseudo-mask quality enhancement
features to potentially improve mIoU:

1. mask_temperature: Controls the sharpness of softmax distribution
   - Lower values (e.g., 0.5) make predictions more confident/sharp
   - Higher values (e.g., 2.0) make predictions smoother
   - Default: 1.0

2. adaptive_threshold: Enables adaptive confidence thresholding
   - When True, uses percentile-based threshold instead of fixed value
   - More robust across different images and datasets
   - Default: False

3. threshold_percentile: Percentile for adaptive threshold (0.0-1.0)
   - Lower values keep more pixels, higher values filter more
   - Default: 0.3 (filters bottom 30% lowest confidence pixels)

4. confidence_threshold: Fixed confidence threshold (used when adaptive_threshold=False)
   - Pixels with confidence below this threshold are ignored (set to 255)
   - Default: 0.0 (no filtering)

5. multi_scale_mask: Enables multi-scale pseudo-mask generation
   - Generates masks at multiple scales and fuses them
   - Can improve robustness for objects at different scales
   - Default: False

6. mask_scales: Scales used for multi-scale mask generation
   - Default: (0.5, 1.0, 1.5)

Usage:
    python tools/train.py configs/carb/camvid_carb_dual_enhanced.py
"""

_base_ = [
    '../_base_/models/carb.py', '../_base_/datasets/camvid_w.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_12k_lr_0.005.py'
]
suppress_labels = list(range(0, 11))
model = dict(
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(depth=50),
    decode_head=dict(
        num_classes=11,
        text_categories=11,
        text_embeddings_path='pretrain/camvid_ViT16_clip_text.pth',
        clip_unlabeled_cats=suppress_labels,
        coeff=1,
        warmup_iter=4000,
        patch_size=(512, 256),
        resize_rate=2,
        resize_offset=0.5,
        adaptive=True,
        get_train_mask=False,
        # Enhanced pseudo-mask quality settings:
        # Use lower temperature for sharper predictions
        mask_temperature=0.7,
        # Use adaptive thresholding based on confidence distribution
        adaptive_threshold=True,
        threshold_percentile=0.3,  # Filter bottom 30% lowest confidence pixels
        # Enable multi-scale mask fusion for better small object handling
        multi_scale_mask=True,
        mask_scales=(0.75, 1.0, 1.25),
        loss_decode=dict(
            type='CrossEntropyLoss', use_masked=True, loss_weight=1.0),
    ),
    feed_img_to_decode_head=True,
)


find_unused_parameters=True
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (960, 720)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(960, 720), ratio_range=(1.0, 4.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'],
         meta_keys = ['filename', 'ori_filename', 'ori_shape',
                      'img_shape', 'pad_shape', 'scale_factor', 'flip',
                      'flip_direction', 'img_norm_cfg', 'img_label']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(960, 720),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'],
                 meta_keys=['filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg', 'img_label']),
        ])
]
data = dict(
    samples_per_gpu=1,
    train=dict(
        img_dir='img/train',
        ann_dir='mask/train',
        img_labels='metadata/camvid/labels.npy',
        pipeline=train_pipeline
    ),
    val=dict(
        img_dir='img/val',
        ann_dir='mask_idx/val',
        img_labels='metadata/camvid/labels.npy',
        pipeline=test_pipeline),
    test=dict(
        img_dir='img/test',
        ann_dir='mask_idx/test',
        img_labels='metadata/camvid/labels.npy',
        pipeline=test_pipeline)
)
