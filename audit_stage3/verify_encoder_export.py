"""CPU-only verification of the serialized SSL encoder export.

Loads both local checkpoints and checks exact tensor equality for every
official Swin-Tiny backbone key. It does not run training or predictions.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_baseline import now, register_framework, sha256, write_json


ALLOWED_EXCLUSIONS = ['mask_token', 'norm0.bias', 'norm0.weight',
                      'norm1.bias', 'norm1.weight', 'norm2.bias', 'norm2.weight']


def verify(run_dir):
    import torch
    from mmengine.config import Config

    framework = register_framework()
    from mmaction.models.backbones.swin import SwinTransformer3D
    cfg = Config.fromfile(str(ROOT / 'configs/swin_tiny_ssl_prototype.py'))
    backbone_cfg = dict(cfg.model.backbone)
    assert backbone_cfg.pop('type') == 'SwinTransformer3D'
    assert backbone_cfg['arch'] == 'tiny'
    assert backbone_cfg['pretrained'] is None and not backbone_cfg['pretrained2d']
    model = SwinTransformer3D(**backbone_cfg).cpu()
    expected = model.state_dict()

    export_path = run_dir / 'encoder_for_classification.pth'
    full_path = run_dir / 'ssl_step_20.pth'
    exported = torch.load(export_path, map_location='cpu')
    full = torch.load(full_path, map_location='cpu')
    raw = exported['state_dict']
    assert all(n.startswith('backbone.') for n in raw), 'Unexpected export namespace'
    state = {n[len('backbone.'):]: v for n, v in raw.items()}
    assert len(state) == len(raw), 'Duplicate stripped export keys'
    assert set(state) == set(expected), 'Export keys do not exactly match official backbone'
    source = {n[len('encoder.'):]: v for n, v in full['state_dict'].items()
              if n.startswith('encoder.')}
    assert sorted(set(source) - set(expected)) == ALLOWED_EXCLUSIONS
    assert not set(expected) - set(source), 'Missing source encoder keys'
    assert sorted(exported['meta']['excluded_source_keys']) == ALLOWED_EXCLUSIONS
    mismatches = []
    for name, tensor in state.items():
        if tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype:
            mismatches.append(name + ': shape/dtype')
        elif tensor.dtype.is_floating_point and not bool(torch.isfinite(tensor).all()):
            mismatches.append(name + ': nonfinite')
        elif not torch.equal(tensor, source[name]):
            mismatches.append(name + ': differs from serialized full checkpoint')
    assert not mismatches, mismatches
    loaded = model.load_state_dict(state, strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    assert all(torch.equal(v, model.state_dict()[n]) for n, v in state.items())
    return dict(status='PASS', created=now(), device='cpu', torch=str(torch.__version__),
                scope='Serialized checkpoint key/tensor verification only; no training or predictions',
                framework_mode=framework['mode'], expected_keys=len(expected),
                loaded_keys=len(state), checked_keys=sorted(state),
                exact_serialized_source_tensor_matches=len(state),
                allowed_excluded_source_keys=ALLOWED_EXCLUSIONS,
                missing_keys=[], unexpected_keys=[], mismatches=[],
                strict_load='PASS',
                encoder_export_sha256=sha256(export_path),
                full_ssl_checkpoint_sha256=sha256(full_path),
                verifier_sha256=sha256(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path,
                        default=ROOT / 'work_dirs/ssl_proto_20261026_01')
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.relative_to((ROOT / 'work_dirs').resolve())
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    output = run_dir / 'encoder_export_check.json'
    if output.exists():
        raise FileExistsError('Preserve existing export verification: ' + str(output))
    try:
        result = verify(run_dir)
    except Exception as exc:
        failure = dict(status='FAIL', created=now(), error=type(exc).__name__ + ': ' + str(exc),
                       scope='Serialized encoder export verification; no training')
        write_json(output, failure)
        traceback.print_exc()
        raise
    write_json(output, result)
    print(json.dumps({k: result[k] for k in ('status', 'expected_keys', 'loaded_keys',
                                            'exact_serialized_source_tensor_matches',
                                            'encoder_export_sha256')}, indent=2))


if __name__ == '__main__':
    main()
