_base_ = './faster_rcnn_r50_fpn_1x_coco_person.py'

# Modify dataset related settings
dataset_type = 'CocoPersonDataset'
data_root = 'data/trampoline/'
classes = ('person',)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        img_prefix=data_root + 'detections/',
        classes=classes,
        pipeline=train_pipeline,
        ann_file=data_root +'annotations/trampoline-detections-coco.json'),
    val=dict(
        type=dataset_type,
        img_prefix=data_root + 'detections_val/',
        classes=classes,
        pipeline=test_pipeline,
        ann_file=data_root + 'annotations/trampoline-detection-val-coco.json'),
    test=dict(
        type=dataset_type,
        img_prefix=data_root + 'detections_val/',
        classes=classes,
        pipeline=test_pipeline,
        ann_file=data_root + 'annotations/trampoline-detection-val-coco.json'))

# finetune model
load_from = 'work_dirs/faster_rcnn_r50_fpn_1x_coco_person/epoch_11.pth'