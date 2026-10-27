"""Engineering masked-reconstruction variant, not a claimed E-VIST replica.

Call run_baseline.register_framework() before importing this module on mmcv-lite.
Installed MMAction2 modules are not patched. See the stage3 method specification.
"""
import torch
from torch import nn
from torch.nn import functional as F
from mmaction.models.backbones.swin import SwinTransformer3D


class DenseMaskedSwin(SwinTransformer3D):
    """Same official stage operations with dense masking after patch embedding."""
    def __init__(self):
        super().__init__(arch='tiny', pretrained=None, pretrained2d=False,
                         patch_size=(2, 4, 4), window_size=(8, 7, 7),
                         with_cp=True, drop_path_rate=0.,
                         out_indices=(0, 1, 2, 3), out_after_downsample=False)
        self.mask_token = nn.Parameter(torch.zeros(1, self.embed_dims, 1, 1, 1))
        self.init_weights()
        nn.init.trunc_normal_(self.mask_token, std=.02)

    def forward(self, x, masked):
        x = self.patch_embed(x)
        if masked.dtype != torch.bool or tuple(masked.shape) != (x.shape[0], *x.shape[2:]):
            raise ValueError('Expected bool mask [B,D,H,W] matching patch grid')
        if masked.device != x.device:
            raise ValueError('Mask and input must share a device')
        x = self.pos_drop(torch.where(masked[:, None], self.mask_token, x))
        outputs = []
        # Matches installed SwinTransformer3D.forward; only the mask above is added.
        for i, layer in enumerate(self.layers):
            x = layer(x.contiguous(), do_downsample=False)
            outputs.append(getattr(self, 'norm'+str(i))(x).permute(0, 4, 1, 2, 3).contiguous())
            if layer.downsample is not None:
                x = layer.downsample(x)
            if i < self.num_layers - 1:
                x = x.permute(0, 4, 1, 2, 3)
        return tuple(outputs)


class MaskedReconstructionPrototype(nn.Module):
    def __init__(self, decoder_channels=48):
        super().__init__()
        if decoder_channels != 48:
            raise ValueError('This bounded prototype fixes and documents width 48')
        self.encoder = DenseMaskedSwin()
        self.decoder = nn.Sequential(nn.Linear(96+192+384+768, decoder_channels),
                                     nn.GELU(), nn.Linear(decoder_channels, 96))

    def forward(self, normalized_rgb, masked):
        features = self.encoder(normalized_rgb, masked)
        grid = masked.shape[1:]
        aligned = [f if f.shape[2:] == grid else F.interpolate(f, size=grid, mode='nearest')
                   for f in features]
        joined = torch.cat(aligned, dim=1).permute(0, 2, 3, 4, 1)
        return self.decoder(joined), [list(f.shape) for f in features]
