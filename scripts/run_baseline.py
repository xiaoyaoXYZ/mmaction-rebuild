"""Reproducible real-data smoke, one-epoch training, and checkpoint validation.

Run with the isolated .venv Python. This script never installs dependencies or
edits framework/source data. Each invocation requires a new output directory.
"""
import argparse
import contextlib
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[1]
LIMITATION = ('Engineering validation only. Source sessions/persons are unknown; '
              'clips may share an acquisition source. No independent test set '
              'and no target accuracy. No MAE pretraining is performed here.')


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf8')


class Tee:
    def __init__(self, console, logfile):
        self.console, self.logfile = console, logfile

    def write(self, text):
        self.console.write(text)
        self.logfile.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.logfile.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return 'utf-8'


def package_namespace(name, path):
    """Skip broad optional-package __init__ imports, not model calculations."""
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    module.__file__ = str(path / '__init__.py')
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None,
                                                   is_package=True)
    module.__spec__.submodule_search_locations = module.__path__
    sys.modules[name] = module
    parent, _, child = name.rpartition('.')
    if parent in sys.modules:
        setattr(sys.modules[parent], child, module)


def register_framework():
    """Register official modules, with a narrow mmcv-lite classification path.

    MMAction2's broad model imports include detection RoI operators. Swin video
    classification itself does not use those compiled operators. On mmcv-lite,
    import the installed official classification files without importing every
    unrelated detector. No operators, layers, losses, or algorithms are faked.
    """
    import mmaction
    import mmcv
    from mmengine.registry import init_default_scope

    official_root = Path(mmaction.__file__).resolve().parent
    if importlib.util.find_spec('mmcv._ext') is not None:
        from mmaction.utils import register_all_modules
        register_all_modules(init_default_scope=True)
        mode = 'official_register_all_modules'
        imported = []
    else:
        for suffix in ('models', 'models.backbones', 'models.heads',
                       'models.losses', 'models.recognizers',
                       'models.data_preprocessors'):
            package_namespace('mmaction.' + suffix,
                              official_root.joinpath(*suffix.split('.')))
        imported = [
            'mmaction.models.losses.cross_entropy_loss',
            'mmaction.models.backbones.swin',
            'mmaction.models.heads.i3d_head',
            'mmaction.models.data_preprocessors.data_preprocessor',
            'mmaction.models.recognizers.recognizer3d',
            'mmaction.datasets', 'mmaction.evaluation',
            'mmaction.visualization']
        for name in imported:
            importlib.import_module(name)
        init_default_scope('mmaction')
        mode = 'scoped_official_classification_imports_for_mmcv_lite'
    files = {}
    for name in imported:
        module_path = Path(sys.modules[name].__file__)
        files[name] = {'path': str(module_path), 'sha256': sha256(module_path)}
    return dict(mode=mode, mmaction_version=mmaction.__version__,
                mmcv_version=mmcv.__version__, official_module_files=files,
                note='Compiled detection operators are unavailable with mmcv-lite; '
                     'this run only uses official Swin classification components.')


def load_config(path):
    from mmengine.config import Config
    cfg = Config.fromfile(str(path))
    if cfg.model.backbone.type != 'SwinTransformer3D' or cfg.model.backbone.arch != 'tiny':
        raise ValueError('This audit runner expects the full official Swin tiny.')
    if cfg.model.backbone.get('pretrained') is not None or cfg.get('load_from'):
        raise ValueError('Baseline training must start from random initialization.')
    if cfg.get('resume') or cfg.train_cfg.max_epochs != 1:
        raise ValueError('Use a new one-epoch baseline, without resume.')
    if cfg.optim_wrapper.type != 'OptimWrapper':
        raise ValueError('This memory baseline uses FP32 OptimWrapper.')
    return cfg


def describe_data(cfg):
    result = {'num_classes': int(cfg.model.cls_head.num_classes),
              'limitation': LIMITATION}
    paths = {}
    for split in ('train', 'val'):
        dataset = cfg[split + '_dataloader'].dataset
        ann_path = Path(dataset.ann_file)
        rows = []
        for line in ann_path.read_text(encoding='utf8').splitlines():
            if not line.strip():
                continue
            relative, frames, label = line.rsplit(maxsplit=2)
            row = dict(clip=relative, frames=int(frames), label=int(label))
            if row['frames'] <= 0 or not 0 <= row['label'] < result['num_classes']:
                raise ValueError('Invalid annotation: ' + line)
            if not (Path(dataset.data_prefix.img) / relative).is_dir():
                raise FileNotFoundError(relative)
            rows.append(row)
        labels = sorted(set(row['label'] for row in rows))
        if labels != list(range(result['num_classes'])):
            raise ValueError(split + ' does not cover every class')
        paths[split] = [row['clip'] for row in rows]
        if len(paths[split]) != len(set(paths[split])):
            raise ValueError('Duplicate clip in ' + split)
        result[split] = dict(annotations=str(ann_path), sha256=sha256(ann_path),
                             clips=len(rows), annotated_frames=sum(x['frames'] for x in rows),
                             labels=labels, rows=rows)
    if set(paths['train']) & set(paths['val']):
        raise ValueError('Train and validation clip directories overlap')
    return result


def memory_stats(torch, device):
    if device.type != 'cuda':
        return {'device': str(device), 'cuda_used': False}
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return dict(device=str(device), cuda_used=True,
                allocated_bytes=torch.cuda.memory_allocated(device),
                reserved_bytes=torch.cuda.memory_reserved(device),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                free_bytes=free, total_bytes=total)


def probabilities(torch, samples, num_classes):
    scores = torch.stack([x.pred_score.detach().float().cpu() for x in samples])
    if scores.shape != (len(samples), num_classes):
        raise RuntimeError('Wrong prediction shape: ' + str(tuple(scores.shape)))
    if not torch.isfinite(scores).all() or not (scores >= 0).all():
        raise RuntimeError('Invalid class probabilities')
    if not torch.allclose(scores.sum(dim=1), torch.ones(len(samples)), atol=1e-5):
        raise RuntimeError('Predicted probabilities do not sum to one')
    return scores


def make_model(cfg, device):
    from mmaction.registry import MODELS
    model = MODELS.build(copy.deepcopy(cfg.model))
    model.init_weights()
    return model.to(device)


def smoke(cfg, output, dataset_report, torch, device):
    from mmaction.registry import DATASETS
    from mmengine.dataset import pseudo_collate
    from mmengine.optim import OptimWrapper

    dataset = DATASETS.build(copy.deepcopy(cfg.train_dataloader.dataset))
    if len(dataset) < 2:
        raise RuntimeError('Smoke requires two distinct real training clips')
    model = make_model(cfg, device)
    model.train()
    optimizer_cfg = dict(cfg.optim_wrapper.optimizer)
    optimizer_type = optimizer_cfg.pop('type')
    if optimizer_type != 'SGD':
        raise ValueError('Expected SGD for this memory baseline')
    optimizer = torch.optim.SGD(model.parameters(), **optimizer_cfg)
    gradient_steps = []

    class ObservedOptimWrapper(OptimWrapper):
        def update_params(self, loss, *args, **kwargs):
            if not torch.isfinite(loss).all():
                raise RuntimeError('Non-finite loss before backward')
            return super().update_params(loss, *args, **kwargs)

        def step(self, **kwargs):
            observed = {}
            for group, module in (('backbone', model.backbone), ('head', model.cls_head)):
                norms = []
                missing = []
                nonzero = 0
                for name, parameter in module.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    if parameter.grad is None:
                        missing.append(name)
                        continue
                    grad = parameter.grad.detach()
                    if not torch.isfinite(grad).all():
                        raise RuntimeError('Non-finite gradient: ' + group + '.' + name)
                    norm = float(torch.linalg.vector_norm(grad).item())
                    norms.append(norm)
                    nonzero += int(norm > 0)
                norm = math.sqrt(sum(x * x for x in norms))
                if not math.isfinite(norm) or norm <= 0:
                    raise RuntimeError('No usable ' + group + ' gradient')
                observed[group] = dict(l2_norm_before_clipping=norm,
                                       tensors_with_gradient=len(norms),
                                       nonzero_gradient_tensors=nonzero,
                                       missing_gradient_parameters=missing)
            gradient_steps.append(observed)
            return super().step(**kwargs)

    wrapper = ObservedOptimWrapper(optimizer,
                                   clip_grad=dict(cfg.optim_wrapper.clip_grad))
    selected_parameters = {}
    for prefix in ('backbone.', 'cls_head.'):
        name, parameter = next((n, p) for n, p in model.named_parameters()
                               if n.startswith(prefix) and p.requires_grad and p.ndim >= 2)
        selected_parameters[name] = parameter
    records = []
    for index in range(2):
        batch = pseudo_collate([dataset[index]])
        raw_shape = list(batch['inputs'][0].shape)
        before = {name: p.detach().cpu().clone() for name, p in selected_parameters.items()}
        started = time.perf_counter()
        log_vars = model.train_step(copy.deepcopy(batch), wrapper)
        losses = {name: float(value.detach().cpu().item())
                  for name, value in log_vars.items()}
        if not all(math.isfinite(value) for value in losses.values()):
            raise RuntimeError('Non-finite train_step output')
        changes = {name: float((p.detach().cpu() - before[name]).abs().max().item())
                   for name, p in selected_parameters.items()}
        if not all(value > 0 for value in changes.values()):
            raise RuntimeError('Backbone/head selected parameters did not update')
        model.eval()
        with torch.no_grad():
            scores = probabilities(torch, model.val_step(copy.deepcopy(batch)),
                                   dataset_report['num_classes'])
        model.train()
        record = dict(index=index, clip=dataset_report['train']['rows'][index]['clip'],
                      input_shape_before_batch=raw_shape,
                      output_probability_shape=list(scores.shape),
                      losses=losses, gradients=gradient_steps[-1],
                      selected_parameter_max_abs_updates=changes,
                      probabilities=scores.tolist(),
                      seconds=time.perf_counter() - started,
                      gpu_memory=memory_stats(torch, device))
        records.append(record)
        print(json.dumps({'smoke_step': index + 1, 'loss': losses['loss'],
                          'parameter_update': changes, 'gpu_memory': record['gpu_memory']}),
              flush=True)
    return dict(status='PASS', mode='smoke', steps=records,
                parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                result_scope='Two real sequential batches of size 1, not a trained baseline checkpoint',
                gpu_memory=memory_stats(torch, device))


def load_checkpoint_strict(model, path, torch):
    # Only checkpoints generated locally by this experiment are intended here.
    checkpoint = torch.load(str(path), map_location='cpu')
    if 'state_dict' not in checkpoint:
        raise ValueError('Expected an MMEngine checkpoint containing state_dict')
    for name, value in checkpoint['state_dict'].items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise RuntimeError('Checkpoint contains non-finite tensor: ' + name)
    result = model.load_state_dict(checkpoint['state_dict'], strict=True)
    report = dict(path=str(Path(path).resolve()), sha256=sha256(path),
                  bytes=Path(path).stat().st_size, strict=True,
                  missing_keys=list(result.missing_keys),
                  unexpected_keys=list(result.unexpected_keys),
                  optimizer_present='optimizer' in checkpoint,
                  epoch=checkpoint.get('meta', {}).get('epoch'),
                  iteration=checkpoint.get('meta', {}).get('iter'))
    del checkpoint
    return report


def train(cfg, output, dataset_report, torch, device):
    from mmengine.hooks import Hook
    from mmengine.runner import Runner

    if device.type != 'cuda':
        raise ValueError('Official baseline training requires --device cuda:0 here')
    cfg.work_dir = str(output)
    val_metrics = {}
    iterations = []

    class AuditHook(Hook):
        def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
            scalars = {key: float(value.detach().cpu().item())
                       if torch.is_tensor(value) else float(value)
                       for key, value in outputs.items()}
            if not all(math.isfinite(value) for value in scalars.values()):
                raise RuntimeError('Non-finite training metric')
            iterations.append(dict(iteration=runner.iter + 1, metrics=scalars))

        def after_val_epoch(self, runner, metrics=None):
            val_metrics.update({k: float(v) for k, v in (metrics or {}).items()})

    runner = Runner.from_cfg(cfg)
    runner.register_hook(AuditHook(), priority='LOW')
    runner.train()
    expected = math.ceil(dataset_report['train']['clips'] / cfg.train_dataloader.batch_size)
    if runner.epoch != 1 or runner.iter != expected or len(iterations) != expected:
        raise RuntimeError('The requested full one-epoch training did not finish')
    if not val_metrics or not all(math.isfinite(x) for x in val_metrics.values()):
        raise RuntimeError('Finite validation metrics were not produced')
    checkpoint_path = output / 'epoch_1.pth'
    if not checkpoint_path.is_file():
        raise FileNotFoundError('Epoch checkpoint was not saved')
    checkpoint = load_checkpoint_strict(runner.model, checkpoint_path, torch)
    if not checkpoint['optimizer_present']:
        raise RuntimeError('Optimizer state was not saved')
    return dict(status='PASS', mode='train', epochs=runner.epoch, iterations=iterations,
                validation_metrics=val_metrics, checkpoint=checkpoint,
                checkpoint_files=[{'name': p.name, 'sha256': sha256(p), 'bytes': p.stat().st_size}
                                  for p in sorted(output.glob('*.pth'))],
                gpu_memory=memory_stats(torch, device))


def evaluate(cfg, output, dataset_report, torch, device, checkpoint_path):
    from mmaction.registry import DATASETS
    from mmengine.dataset import pseudo_collate
    from mmengine.evaluator import Evaluator

    dataset = DATASETS.build(copy.deepcopy(cfg.val_dataloader.dataset))
    model = make_model(cfg, device)
    checkpoint = load_checkpoint_strict(model, checkpoint_path, torch)
    model.eval()
    evaluator = Evaluator(copy.deepcopy(cfg.val_evaluator))
    classes_path = ROOT / 'data/splits/classes.json'
    class_names = json.loads(classes_path.read_text(encoding='utf8'))
    count = dataset_report['num_classes']
    confusion = [[0 for _ in range(count)] for _ in range(count)]
    records = []
    with torch.no_grad():
        for index in range(len(dataset)):
            # Decode twice to verify deterministic validation sampling and crop.
            batch = pseudo_collate([dataset[index]])
            repeated_batch = pseudo_collate([dataset[index]])
            if not torch.equal(batch['inputs'][0], repeated_batch['inputs'][0]):
                raise RuntimeError('Validation input sampling is not deterministic')
            predictions = model.val_step(copy.deepcopy(batch))
            scores = probabilities(torch, predictions, count)
            repeat = probabilities(torch, model.val_step(repeated_batch), count)
            delta = float((scores - repeat).abs().max().item())
            if not torch.allclose(scores, repeat, atol=1e-6, rtol=0):
                raise RuntimeError('Repeated validation predictions disagree')
            evaluator.process(data_samples=predictions, data_batch=batch)
            truth = int(predictions[0].gt_label.item())
            predicted = int(scores[0].argmax().item())
            row = dataset_report['val']['rows'][index]
            if truth != row['label']:
                raise RuntimeError('Decoded label differs from annotation')
            confusion[truth][predicted] += 1
            records.append(dict(clip=row['clip'], total_frames=row['frames'],
                                true_label=truth, true_class=class_names[str(truth)],
                                predicted_label=predicted,
                                predicted_class=class_names[str(predicted)],
                                probabilities=scores[0].tolist(),
                                repeat_max_abs_difference=delta))
            print(json.dumps({'validation_clip': index + 1, 'true': truth,
                              'predicted': predicted, 'repeat_difference': delta}), flush=True)
    metrics = {k: float(v) for k, v in evaluator.evaluate(len(dataset)).items()}
    if sum(sum(row) for row in confusion) != dataset_report['val']['clips']:
        raise RuntimeError('Validation count is incomplete')
    return dict(status='PASS', mode='evaluate', checkpoint=checkpoint,
                validation_metrics=metrics, classes=class_names,
                confusion_matrix=confusion,
                confusion_orientation='rows=true labels, columns=predicted labels',
                clips=records, repeated_prediction_tolerance=1e-6,
                gpu_memory=memory_stats(torch, device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['smoke', 'train', 'evaluate'])
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/swin_tiny_industrial.py')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.mode == 'evaluate' and (args.checkpoint is None or not args.checkpoint.is_file()):
        parser.error('evaluate requires an existing --checkpoint')
    if args.mode != 'evaluate' and args.checkpoint is not None:
        parser.error('--checkpoint is only accepted for evaluate')
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    output = (args.output_dir or ROOT / 'work_dirs' / (args.mode + '_' + stamp)).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(status='RUNNING', mode=args.mode, started=now(),
                  output_directory=str(output), command=sys.argv,
                  python=sys.executable, limitation=LIMITATION)
    with (output / 'console.log').open('x', encoding='utf8', buffering=1) as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout, logfile)), contextlib.redirect_stderr(Tee(sys.stderr, logfile)):
            try:
                os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
                import torch
                from mmengine.runner import set_random_seed

                device = torch.device(args.device)
                if device.type == 'cuda':
                    if not torch.cuda.is_available():
                        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
                    torch.cuda.set_device(device)
                    torch.cuda.reset_peak_memory_stats(device)
                report['framework_registration'] = register_framework()
                cfg = load_config(args.config.resolve())
                set_random_seed(cfg.randomness.seed, deterministic=True)
                torch.use_deterministic_algorithms(True)
                report['versions'] = {name: importlib.metadata.version(name)
                                      for name in ('torch', 'torchvision', 'mmengine',
                                                   'mmcv-lite', 'mmaction2', 'numpy')}
                report['config_source'] = str(args.config.resolve())
                report['config_sha256'] = sha256(args.config)
                cfg.dump(str(output / 'resolved_config.py'))
                dataset_report = describe_data(cfg)
                report['dataset'] = dataset_report
                write_json(output / 'dataset_snapshot.json', dataset_report)
                if args.mode == 'smoke':
                    result = smoke(cfg, output, dataset_report, torch, device)
                elif args.mode == 'train':
                    result = train(cfg, output, dataset_report, torch, device)
                else:
                    result = evaluate(cfg, output, dataset_report, torch, device,
                                      args.checkpoint.resolve())
                report.update(result)
                report.update(finished=now(), seconds=time.perf_counter() - started)
                write_json(output / 'result.json', report)
                print(json.dumps({'status': report['status'], 'mode': args.mode,
                                  'result': str(output / 'result.json')}), flush=True)
            except Exception as exc:
                report.update(status='FAIL', finished=now(),
                              seconds=time.perf_counter() - started,
                              error=type(exc).__name__ + ': ' + str(exc))
                write_json(output / 'result.json', report)
                traceback.print_exc()
                raise


if __name__ == '__main__':
    main()
