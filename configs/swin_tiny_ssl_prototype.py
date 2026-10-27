"""Engineering prototype, NOT the complete E-VIST paper architecture.

Paper DOI 10.1109/ACCESS.2024.3455749. Missing decoder/target details are
explicit assumptions below. The supervised baseline remains unchanged.
"""
_base_ = './swin_tiny_two_sample_diagnostic.py'
randomness = dict(seed=20261026, deterministic=True)
ssl = dict(
    scope='Engineering masked reconstruction on two TRAIN inputs only; no independent validation or paper reproduction claim',
    max_updates=20,
    evaluate_every=5,
    mask_ratio=0.5,
    mask_mode='spatial_tube',
    decoder_channels=48,
    target='rgb_0_1_patch',
    loss='masked_patch_mse',
    optimizer=dict(lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    grad_clip=5.0,
    assumptions=[
        'Official MMAction2 Swin-Tiny, 8x112x112, not paper Base/16x224x224',
        'Time-shared spatial patch mask follows Eq11/Fig7-8 interpretation; Eq16 temporal-block conflict unresolved',
        'All RGB channels of each patch masked together; learned dense mask vector after patch embedding',
        'Four normalized stage maps aligned by nearest-neighbor interpolation to the initial token grid',
        'Concatenate channels then Linear(1440,48), GELU, Linear(48,96); no decoder transformer blocks',
        'C/2 interpreted as half initial Tiny embedding width, not confirmed by original paper',
        'Detached RGB/255 target, no per-patch target normalization, masked-only element MSE; D/h/T unspecified in paper',
        'AdamW 1e-4, weight_decay .05, constant LR, batch 1, 20 updates; not paper .003/warmup/cosine/150 epochs',
        'Fixed training clips/crops, fresh training masks; fixed diagnostic masks on the same inputs',
        'No K400, no validation inputs, no class labels in reconstruction loss, no source-independence claim'])
