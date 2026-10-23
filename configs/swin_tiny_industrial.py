"""One-epoch pipeline baseline, adapted from the supplied Starter model.

10 classes, 10 training clips, 10 validation clips. Random initialization.
No test set and no accuracy target. Source-session independence is unknown.
"""
ROOT = '{{fileDirname}}/..'
default_scope = 'mmaction'
model = dict(
    type='Recognizer3D',
    backbone=dict(type='SwinTransformer3D', arch='tiny', pretrained=None,
                  pretrained2d=False, patch_size=(2, 4, 4),
                  window_size=(8, 7, 7), with_cp=True, drop_path_rate=0.1),
    cls_head=dict(type='I3DHead', in_channels=768, num_classes=10,
                  spatial_type='avg', dropout_ratio=0.5, average_clips='prob'),
    data_preprocessor=dict(type='ActionDataPreprocessor',
                           mean=[123.675, 116.28, 103.53],
                           std=[58.395, 57.12, 57.375], format_shape='NCTHW'))
train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 128)),
    dict(type='RandomCrop', size=112),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')]
val_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=2, num_clips=1,
         test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 128)),
    dict(type='CenterCrop', crop_size=112),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')]
train_dataloader = dict(
    batch_size=1, num_workers=0, persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='RawframeDataset',
                 ann_file=ROOT + '/data/splits/train.txt',
                 data_prefix=dict(img=ROOT + '/data/ucf101/rawframes'),
                 filename_tmpl='img_{:05}.jpg', start_index=1,
                 modality='RGB', pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1, num_workers=0, persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='RawframeDataset',
                 ann_file=ROOT + '/data/splits/val.txt',
                 data_prefix=dict(img=ROOT + '/data/ucf101/rawframes'),
                 filename_tmpl='img_{:05}.jpg', start_index=1,
                 modality='RGB', pipeline=val_pipeline, test_mode=True))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
val_evaluator = dict(type='AccMetric', metric_list=('top_k_accuracy', 'mean_class_accuracy'),
                     metric_options=dict(top_k_accuracy=dict(topk=(1, 5))))
optim_wrapper = dict(type='OptimWrapper',
                     optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
                     clip_grad=dict(max_norm=40, norm_type=2))
param_scheduler = []
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_optimizer=True,
                    save_best='acc/top1', rule='greater', max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook'))
env_cfg = dict(cudnn_benchmark=False,
               mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
               dist_cfg=dict(backend='gloo'))
randomness = dict(seed=20261017, deterministic=True)
log_processor = dict(type='LogProcessor', window_size=1, by_epoch=True)
visualizer = dict(type='ActionVisualizer', vis_backends=[dict(type='LocalVisBackend')])
log_level = 'INFO'
work_dir = ROOT + '/work_dirs/swin_tiny_1epoch'
load_from = None
resume = False
