"""Run at most 20 masked-reconstruction updates on two fixed TRAIN clips.

This is an explicitly simplified engineering variant, not exact E-VIST/MAE.
All raw evidence and weights stay in a new local work_dirs folder.
"""
import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from run_baseline import ROOT, Tee, now, sha256, write_json, register_framework, make_model, memory_stats
from run_two_sample_diagnostic import baseline_check, inputs

CONFIG = ROOT / 'configs/swin_tiny_ssl_prototype.py'


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def gradients(model, torch):
    groups = dict(encoder=[], decoder=[], mask_token=[])
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise RuntimeError('Missing/nonfinite gradient: '+name)
        group = 'mask_token' if name == 'encoder.mask_token' else name.split('.')[0]
        groups[group].append(torch.linalg.vector_norm(p.grad.detach()))
    result = {}
    for name, values in groups.items():
        norm = float(torch.linalg.vector_norm(torch.stack(values)).item())
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError('Invalid group gradient: '+name)
        result[name] = dict(tensors=len(values), finite=True, l2_norm=norm)
    return result


def evaluate(model, samples, masks, torch, step):
    from ssl_primitives import masked_patch_mse
    model.eval()
    losses, predictions = [], []
    with torch.no_grad():
        for sample, mask in zip(samples, masks):
            prediction, _ = model(sample['input'], mask)
            losses.append(float(masked_patch_mse(prediction, sample['target'], mask).item()))
            predictions.append(prediction.detach().cpu())
    return dict(step=step, mean_mse=sum(losses)/len(losses), input_mse=losses,
                scope='Same TRAIN inputs, fixed masks, no independent validation'), predictions


def preflight(model, sample, mask, torch):
    from ssl_primitives import patchify, unpatchify, masked_patch_mse
    model.eval()
    x = sample['input'].detach().clone().requires_grad_(True)
    prediction, shapes = model(x, mask)
    assert list(prediction.shape) == [1, 4, 28, 28, 96]
    assert shapes == [[1, 96, 4, 28, 28], [1, 192, 4, 14, 14],
                      [1, 384, 4, 7, 7], [1, 768, 4, 4, 4]]
    target = sample['target'].detach().clone().requires_grad_(True)
    loss = masked_patch_mse(prediction, target, mask)
    loss.backward()
    grad = patchify(x.grad)
    blocked_max = float(grad[mask].abs().max().item())
    visible_max = float(grad[~mask].abs().max().item())
    assert blocked_max == 0 and visible_max > 0 and target.grad is None
    gradient_report = gradients(model, torch)
    # Hidden pixel values must not reach the encoder via overlapping embeddings.
    changed = patchify(sample['input']).clone()
    changed[mask] = changed[mask] + 17.
    with torch.no_grad():
        altered, _ = model(unpatchify(changed), mask)
    delta = float((prediction.detach()-altered).abs().max().item())
    assert delta == 0, 'Masked target values leaked into encoder'
    model.zero_grad(set_to_none=True)
    return dict(status='PASS', output_shape=list(prediction.shape), stage_shapes=shapes,
                masked_input_gradient_max=blocked_max, visible_input_gradient_max=visible_max,
                target_detached=target.grad is None, hidden_pixel_perturb_prediction_max=delta,
                gradients=gradient_report)


def transfer_check(model, cfg, samples, output, torch):
    classifier = make_model(cfg, torch.device('cuda:0')).eval()
    head_before = {n: tensor_hash(v) for n, v in classifier.cls_head.state_dict().items()}
    source = model.encoder.state_dict()
    expected = classifier.backbone.state_dict()
    excluded = sorted(set(source)-set(expected))
    assert excluded == ['mask_token', 'norm0.bias', 'norm0.weight', 'norm1.bias',
                        'norm1.weight', 'norm2.bias', 'norm2.weight']
    assert not set(expected)-set(source)
    state = {n: source[n] for n in expected}
    assert all(state[n].shape == expected[n].shape for n in expected)
    classifier.backbone.load_state_dict(state, strict=True)
    assert all(torch.equal(state[n], classifier.backbone.state_dict()[n]) for n in expected)
    assert head_before == {n: tensor_hash(v) for n, v in classifier.cls_head.state_dict().items()}
    model.eval()
    x = samples[0]['input']
    with torch.no_grad():
        no_mask = torch.zeros((1, 4, 28, 28), dtype=torch.bool, device=x.device)
        reference = model.encoder(x, no_mask)[-1]
        feature = classifier.backbone(x)
        delta = float((reference-feature).abs().max().item())
        scores = classifier.cls_head(feature).softmax(-1)
    assert delta <= 1e-6 and list(scores.shape) == [1, 10] and torch.isfinite(scores).all()
    export = output/'encoder_for_classification.pth'
    torch.save(dict(state_dict={'backbone.'+n: v.detach().cpu() for n, v in state.items()},
                    meta=dict(scope=cfg.ssl.scope, excluded_source_keys=excluded,
                              note='Backbone only; classification head must be initialized separately')), export)
    classifier_path = output/'classifier_transfer_check.pth'
    torch.save(dict(state_dict={n: v.detach().cpu() for n, v in classifier.state_dict().items()},
                    meta=dict(scope='Migration plumbing only, no supervised fine-tuning or validation')), classifier_path)
    reference_scores = scores.cpu()
    keys = sorted(expected)
    del classifier, source, expected, state, feature, reference, scores
    gc.collect(); torch.cuda.empty_cache()
    classifier = make_model(cfg, torch.device('cuda:0')).eval()
    classifier.load_state_dict(torch.load(classifier_path, map_location='cpu')['state_dict'], strict=True)
    with torch.no_grad():
        scores = classifier.cls_head(classifier.backbone(x)).softmax(-1).cpu()
    reload_delta = float((scores-reference_scores).abs().max().item())
    assert reload_delta <= 1e-6
    return dict(status='PASS', loaded_backbone_keys=keys, excluded_source_keys=excluded,
                missing_backbone_keys=[], shape_mismatches=[], randomly_initialized_head_unchanged=True,
                no_mask_feature_max_abs_difference=delta,
                classifier_strict_reload_score_max_abs_difference=reload_delta,
                classifier_output_shape=list(scores.shape),
                encoder_checkpoint_sha256=sha256(export), classifier_checkpoint_sha256=sha256(classifier_path),
                note='No class score/accuracy evaluation; no supervised fine-tuning')


def execute(cfg, output, torch, report):
    from ssl_primitives import patchify, make_mask, masked_patch_mse
    from ssl_swin_prototype import MaskedReconstructionPrototype
    batches, records = inputs(cfg, output)
    prior = json.loads((ROOT/'work_dirs/two_sample_diag_20261023_01/prepared.json').read_text(encoding='utf8'))
    assert records == prior['records'], 'Fixed inputs changed since reviewed stage2 diagnostic'
    report['inputs'] = records
    device = torch.device('cuda:0')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no automatic environment or budget changes')
    torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
    mean = torch.tensor(cfg.model.data_preprocessor.mean, device=device).reshape(1, 3, 1, 1, 1)
    std = torch.tensor(cfg.model.data_preprocessor.std, device=device).reshape(1, 3, 1, 1, 1)
    samples = []
    for batch in batches:
        raw = batch['inputs'][0].to(device=device, dtype=torch.float32)
        assert list(raw.shape) == [1, 3, 8, 112, 112] and raw.min() >= 0 and raw.max() <= 255
        samples.append(dict(input=(raw-mean)/std, target=patchify(raw/255.).detach()))
    fixed_rng = torch.Generator().manual_seed(cfg.randomness.seed+1)
    train_rng = torch.Generator().manual_seed(cfg.randomness.seed+2)
    def mask(rng):
        return make_mask(1, (4, 28, 28), cfg.ssl.mask_ratio, cfg.ssl.mask_mode, rng).to(device)
    fixed_masks = [mask(fixed_rng) for _ in samples]
    report['fixed_mask_sha256'] = [tensor_hash(m) for m in fixed_masks]
    report['masked_tokens_per_input'] = [int(m.sum().item()) for m in fixed_masks]
    model = MaskedReconstructionPrototype(cfg.ssl.decoder_channels).to(device)
    report['parameters'] = sum(p.numel() for p in model.parameters())
    report['preflight'] = preflight(model, samples[0], fixed_masks[0], torch)
    optimizer = torch.optim.AdamW(model.parameters(), **dict(cfg.ssl.optimizer))
    report['evaluations'] = [evaluate(model, samples, fixed_masks, torch, 0)[0]]
    tracked_names = ['encoder.patch_embed.proj.weight', 'encoder.mask_token', 'decoder.0.weight', 'decoder.2.weight']
    tracked = {n: dict(model.named_parameters())[n] for n in tracked_names}
    report['updates'] = []
    start = time.perf_counter()
    assert 0 < cfg.ssl.max_updates <= 20
    with (output/'updates.jsonl').open('x', encoding='utf8', buffering=1) as log:
        for step in range(1, cfg.ssl.max_updates+1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            idx = (step-1) % len(samples)
            masked = mask(train_rng)
            before = {n: p.detach().clone() for n, p in tracked.items()}
            prediction, _ = model(samples[idx]['input'], masked)
            loss = masked_patch_mse(prediction, samples[idx]['target'], masked)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss before update')
            loss.backward()
            observed = gradients(model, torch)
            total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.ssl.grad_clip, error_if_nonfinite=True).item())
            optimizer.step()
            changes = {n: float((p.detach()-before[n]).abs().max().item()) for n, p in tracked.items()}
            if not all(math.isfinite(v) and v > 0 for v in changes.values()):
                raise RuntimeError('Tracked parameter did not update finitely')
            row = dict(step=step, input_index=idx, loss=float(loss.item()), masked_tokens=int(masked.sum().item()),
                       mask_sha256=tensor_hash(masked), gradients=observed, grad_norm_before_clip=total_norm,
                       parameter_max_abs_changes=changes)
            report['updates'].append(row); log.write(json.dumps(row)+'\n')
            if step % cfg.ssl.evaluate_every == 0:
                evaluation, _ = evaluate(model, samples, fixed_masks, torch, step)
                report['evaluations'].append(evaluation)
                print(json.dumps(dict(step=step, train_mse=row['loss'], fixed_train_mse=evaluation['mean_mse'])), flush=True)
    torch.cuda.synchronize()
    report['optimization_seconds'] = time.perf_counter()-start
    report['gpu_memory_training'] = memory_stats(torch, device)
    checkpoint = output/'ssl_step_20.pth'
    torch.save(dict(state_dict={n: p.detach().cpu() for n, p in model.state_dict().items()},
                    optimizer=optimizer.state_dict(), meta=dict(scope=cfg.ssl.scope, seed=cfg.randomness.seed,
                    iterations=len(report['updates']), assumptions=list(cfg.ssl.assumptions))), checkpoint)
    report['checkpoint'] = dict(path=checkpoint.name, bytes=checkpoint.stat().st_size, sha256=sha256(checkpoint))
    _, reference = evaluate(model, samples, fixed_masks, torch, cfg.ssl.max_updates)
    del model, optimizer, tracked, before, prediction, loss
    gc.collect(); torch.cuda.empty_cache()
    reloaded = MaskedReconstructionPrototype(cfg.ssl.decoder_channels).to(device).eval()
    reloaded.load_state_dict(torch.load(checkpoint, map_location='cpu')['state_dict'], strict=True)
    _, restored = evaluate(reloaded, samples, fixed_masks, torch, cfg.ssl.max_updates)
    delta = max(float((a-b).abs().max().item()) for a, b in zip(reference, restored))
    assert delta <= 1e-6
    report['strict_reload_prediction_max_abs_difference'] = delta
    report['transfer'] = transfer_check(reloaded, cfg, samples, output, torch)
    report['gpu_memory_including_reload_transfer'] = memory_stats(torch, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.relative_to((ROOT/'work_dirs').resolve())
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    report = dict(status='RUNNING', started=now(), command=sys.argv)
    sources = [CONFIG, ROOT/'scripts/run_ssl_prototype.py', ROOT/'scripts/ssl_swin_prototype.py',
               ROOT/'scripts/ssl_primitives.py', ROOT/'scripts/run_two_sample_diagnostic.py',
               ROOT/'scripts/run_baseline.py', ROOT/'configs/swin_tiny_two_sample_diagnostic.py',
               ROOT/'configs/swin_tiny_industrial.py']
    with (output/'console.log').open('x', encoding='utf8', buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                import torch
                from mmengine.config import Config
                from mmengine.runner import set_random_seed
                report['framework'] = register_framework()
                cfg = Config.fromfile(str(CONFIG)); cfg.dump(str(output/'resolved_config.py'))
                report.update(scope=cfg.ssl.scope, assumptions=list(cfg.ssl.assumptions),
                              source_sha256={p.relative_to(ROOT).as_posix(): sha256(p) for p in sources},
                              versions={n: importlib.metadata.version(n) for n in ('torch', 'torchvision', 'mmaction2', 'mmengine', 'mmcv-lite', 'numpy')},
                              python=sys.version, device=torch.cuda.get_device_name(0))
                saved_code = output/'executed_source'; saved_code.mkdir()
                for p in sources:
                    shutil.copyfile(p, saved_code/p.name)
                set_random_seed(cfg.randomness.seed, deterministic=True)
                torch.use_deterministic_algorithms(True)
                print('Verifying baseline_v0 before bounded SSL diagnostic...', flush=True)
                report['baseline_before'] = baseline_check()
                stage2_model = ROOT/'work_dirs/two_sample_diag_20261023_01/diagnostic_step_50.pth'
                report['stage2_checkpoint_before'] = sha256(stage2_model)
                execute(cfg, output, torch, report)
                report['baseline_after'] = baseline_check()
                report['stage2_checkpoint_after'] = sha256(stage2_model)
                assert report['stage2_checkpoint_before'] == report['stage2_checkpoint_after']
                assert report['source_sha256'] == {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources}
                report.update(status='PASS', finished=now())
                write_json(output/'result.json', report)
                print(json.dumps(dict(status='PASS', result=str(output/'result.json'))), flush=True)
            except Exception as exc:
                report.update(status='FAIL', error=type(exc).__name__+': '+str(exc), finished=now())
                write_json(output/'failure.json', report)
                traceback.print_exc()
                raise


if __name__ == '__main__':
    main()
