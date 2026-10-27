"""Tensor utilities for the stage 3 self-supervision audit.

These are engineering utilities, not the complete paper's architecture.
Spatial tubes and temporal blocks are separate interpretations of unresolved
source descriptions. Raw MSE is an explicit diagnostic choice, not a claim
that the paper's target representation or distance has been established.
"""

import math
from fractions import Fraction
from numbers import Integral, Real

import torch


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("{} must be a positive integer".format(name))
    return int(value)


def _triple(value, name):
    if not isinstance(value, (tuple, list, torch.Size)) or len(value) != 3:
        raise ValueError("{} must contain three positive integers".format(name))
    return tuple(_positive_int(v, name) for v in value)


def _tensor5(value, name):
    if not isinstance(value, torch.Tensor):
        raise TypeError("{} must be a torch.Tensor".format(name))
    if value.ndim != 5 or any(dim <= 0 for dim in value.shape):
        raise ValueError("{} must have five nonempty dimensions".format(name))


def patchify(x, patch_size=(2, 4, 4)):
    """RGB [B,3,T,H,W] -> [B,D,Hg,Wg,pt*ph*pw*3].

    Within each patch the order is (dt, dh, dw, channel), RGB adjacent.
    No padding, normalization, conversion, or detachment is performed.
    """
    _tensor5(x, "x")
    pt, ph, pw = _triple(patch_size, "patch_size")
    batch, channels, frames, height, width = x.shape
    if channels != 3:
        raise ValueError("x must contain exactly three RGB channels")
    if frames % pt or height % ph or width % pw:
        raise ValueError("T, H and W must be divisible by patch_size")
    depth, grid_h, grid_w = frames // pt, height // ph, width // pw
    return (
        x.reshape(batch, channels, depth, pt, grid_h, ph, grid_w, pw)
        .permute(0, 2, 4, 6, 3, 5, 7, 1)
        .reshape(batch, depth, grid_h, grid_w, pt * ph * pw * channels)
    )


def unpatchify(patches, patch_size=(2, 4, 4)):
    """Exactly invert patchify using its (dt, dh, dw, RGB) ordering."""
    _tensor5(patches, "patches")
    pt, ph, pw = _triple(patch_size, "patch_size")
    batch, depth, grid_h, grid_w, features = patches.shape
    if features != pt * ph * pw * 3:
        raise ValueError("patch feature dimension must equal pt*ph*pw*3")
    return (
        patches.reshape(batch, depth, grid_h, grid_w, pt, ph, pw, 3)
        .permute(0, 7, 1, 4, 2, 5, 3, 6)
        .reshape(batch, 3, depth * pt, grid_h * ph, grid_w * pw)
    )


def _exact_count(units, ratio):
    # Decimal-string conversion avoids silently rounding a requested count.
    # 0.5 and 0.58 are exact decimal fractions here; an approximate 1/3 float
    # does not authorize rounding a nonintegral requested count.
    count = units * Fraction(str(float(ratio)))
    if count.denominator != 1:
        raise ValueError("mask ratio must yield an integer count at this granularity")
    return count.numerator


def make_mask(batch, grid, ratio, mode, generator=None):
    """CPU bool [B,D,Hg,Wg]; True means masked, count exact per sample.

    spatial_tube shares a randomly chosen spatial pattern across D slices.
    temporal_block masks consecutive whole temporal slices.
    random_3d chooses individual grid positions as an engineering control.
    The decimal ratio must give an integer count at the mode's granularity.
    Ratios 0 and 1 are allowed here; the loss rejects an empty mask. Pass a
    seeded CPU torch.Generator for reproducibility; None uses the CPU RNG.
    """
    batch = _positive_int(batch, "batch")
    depth, height, width = _triple(grid, "grid")
    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, Real)
        or not math.isfinite(float(ratio))
        or not 0 <= ratio <= 1
    ):
        raise ValueError("ratio must be a finite real number in [0, 1]")
    if mode not in ("spatial_tube", "temporal_block", "random_3d"):
        raise ValueError("unsupported mask mode: {}".format(mode))
    if generator is not None:
        if not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a CPU torch.Generator or None")
        if generator.device.type != "cpu":
            raise ValueError("make_mask requires a CPU generator")
    total_count = _exact_count(depth * height * width, ratio)
    mask = torch.zeros((batch, depth, height, width), dtype=torch.bool, device="cpu")
    if mode == "spatial_tube":
        count = _exact_count(height * width, ratio)
        for item in range(batch):
            spatial = torch.zeros(height * width, dtype=torch.bool, device="cpu")
            indices = torch.randperm(height * width, generator=generator, device="cpu")[:count]
            spatial[indices] = True
            mask[item] = spatial.reshape(height, width).unsqueeze(0)
    elif mode == "temporal_block":
        count = _exact_count(depth, ratio)
        for item in range(batch):
            if count:
                start = int(torch.randint(depth - count + 1, (1,), generator=generator, device="cpu"))
                mask[item, start:start + count] = True
    else:
        for item in range(batch):
            indices = torch.randperm(depth * height * width, generator=generator, device="cpu")[:total_count]
            mask[item].view(-1)[indices] = True
    return mask


def masked_patch_mse(pred, target, mask):
    """Average per-patch MSE, then average only over True mask positions.

    pred/target: finite real floating [B,D,Hg,Wg,F]; mask: bool [B,D,Hg,Wg].
    All inputs must share a device. target is detached, so only pred receives
    gradients. This choice does not assert a paper-defined target or decoder.
    """
    _tensor5(pred, "pred")
    _tensor5(target, "target")
    if pred.shape != target.shape:
        raise ValueError("pred and target shapes must match exactly")
    if not torch.is_floating_point(pred) or not torch.is_floating_point(target):
        raise TypeError("pred and target must be real floating tensors")
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise TypeError("mask must be a boolean tensor")
    if tuple(mask.shape) != tuple(pred.shape[:-1]):
        raise ValueError("mask shape must equal pred.shape[:-1]")
    if pred.device != target.device or pred.device != mask.device:
        raise ValueError("pred, target and mask must be on the same device")
    if not bool(mask.any()):
        raise ValueError("masked_patch_mse requires at least one masked patch")
    detached_target = target.detach()
    if not bool(torch.isfinite(pred).all()) or not bool(torch.isfinite(detached_target).all()):
        raise ValueError("pred and target must contain only finite values")
    # Select first to avoid computing discarded visible-patch squared errors.
    per_patch = (pred[mask] - detached_target[mask]).square().mean(dim=-1)
    loss = per_patch.mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError("masked patch MSE is nonfinite")
    return loss
