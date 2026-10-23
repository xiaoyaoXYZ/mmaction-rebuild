"""Bounded fixed-input memorization diagnostic; no independent validation."""
_base_ = './swin_tiny_industrial.py'
model = dict(backbone=dict(drop_path_rate=0.0), cls_head=dict(dropout_ratio=0.0))
randomness = dict(seed=20261023, deterministic=True)
diagnostic = dict(
    clips=['anya_LCD/v_anya_LCD_1', 'bao/v_bao_1'],
    max_updates=50,
    evaluate_every=5,
    success_observation='Both fixed training inputs correct at each of the last three scheduled evaluations; final mean CE below initial mean CE',
    scope='Optimization diagnostic on recorded training-label identities only; semantic labels and original source chronology remain unverified',
    changes_from_baseline=['Deterministic center sampling/crop reused from validation pipeline on TRAIN clips',
                           'Fixed CPU inputs cached, no input augmentation',
                           'drop_path_rate=0; dropout_ratio=0; seed=20261023',
                           'Alternate two training inputs, at most 50 SGD updates; no independent validation'])
