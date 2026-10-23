"""Prepare/review two fixed training inputs, then run at most 50 updates.

All outputs (including frames, labels, weights) remain local in a new work_dirs
folder. Existing baseline files and installed environments are not modified.
"""
import argparse
import contextlib
import copy
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from run_baseline import (ROOT, Tee, now, sha256, write_json, register_framework,
                          make_model, memory_stats, probabilities, load_checkpoint_strict)

CONFIG = ROOT / 'configs/swin_tiny_two_sample_diagnostic.py'
SCOPE = 'Fixed TRAIN input memorization diagnostic; no independent validation, generalization, MAE or paper reproduction claim. Existing integer labels are used without certifying action semantics or source chronology.'


def baseline_check():
    manifest = json.loads((ROOT/'baseline_v0/manifest.json').read_text(encoding='utf8'))
    failures = [x['path'] for x in manifest if not (ROOT/x['path']).is_file()
                or (ROOT/x['path']).stat().st_size != x['bytes']
                or sha256(ROOT/x['path']) != x['sha256']]
    expected = (ROOT/'baseline_v0/environment-freeze.txt').read_text(encoding='utf8')
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze', '--all'],
                            check=True, capture_output=True, text=True).stdout
    normalize = lambda text: sorted(x.strip() for x in text.splitlines() if x.strip())
    result = dict(files_checked=len(manifest), mismatches=failures,
                  package_versions_unchanged=normalize(expected)==normalize(freeze),
                  package_count=len(normalize(freeze)))
    if failures or not result['package_versions_unchanged']:
        raise RuntimeError('Frozen baseline integrity failed: '+str(result))
    return result


def inputs(cfg, output):
    import torch
    from mmaction.registry import DATASETS
    from mmengine.dataset import pseudo_collate
    rows = [line for line in (ROOT/'data/splits/train.txt').read_text().splitlines() if line.strip()]
    selected = [next(line for line in rows if line.rsplit(maxsplit=2)[0]==clip) for clip in cfg.diagnostic.clips]
    assert len(selected)==2 and len({int(x.rsplit(maxsplit=1)[1]) for x in selected})==2
    annotation = output/'fixed_train.txt'
    expected = '\n'.join(selected)+'\n'
    if annotation.exists():
        assert annotation.read_text(encoding='utf8') == expected
    else:
        annotation.write_text(expected, encoding='utf8')
    dataset_cfg = copy.deepcopy(cfg.train_dataloader.dataset)
    dataset_cfg.ann_file = str(annotation)
    dataset_cfg.test_mode = True
    dataset_cfg.pipeline = copy.deepcopy(cfg.val_pipeline)
    dataset_cfg.pipeline[-1]['meta_keys'] = ('frame_inds','frame_dir','total_frames','img_shape','crop_bbox','scale_factor')
    dataset = DATASETS.build(dataset_cfg)
    batches, records = [], []
    for i in range(2):
        sample, again = dataset[i], dataset[i]
        assert torch.equal(sample['inputs'], again['inputs']), 'Non-deterministic preprocessing'
        assert list(sample['inputs'].shape)==[1,3,8,112,112]
        batch = pseudo_collate([sample])
        meta = sample['data_samples'].metainfo
        frame_ids = [int(x) for x in meta['frame_inds'].reshape(-1)]
        folder = Path(meta['frame_dir'])
        record = dict(clip=cfg.diagnostic.clips[i], recorded_label=int(sample['data_samples'].gt_label.item()),
                      frame_indices=frame_ids, input_shape=list(sample['inputs'].shape),
                      packed_input_sha256=__import__('hashlib').sha256(sample['inputs'].numpy().tobytes()).hexdigest(),
                      deterministic_redecode=True,
                      frames=[dict(number=n,sha256=sha256(folder/f'img_{n:05}.jpg')) for n in frame_ids])
        batches.append(batch)
        records.append(record)
    return batches, records


def preview(output, records):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    canvas = Image.new('RGB',(1280,460),'white')
    draw = ImageDraw.Draw(canvas)
    for i, rec in enumerate(records):
        for j, idx in enumerate((0,2,4,7)):
            n=rec['frame_indices'][idx]
            with Image.open(ROOT/'data/ucf101/rawframes'/rec['clip']/f'img_{n:05}.jpg') as im:
                im.thumbnail((312,176))
                canvas.paste(im,(j*320,i*230+30))
            draw.text((j*320+2,i*230+208),f'label {rec["recorded_label"]}, frame {n}',fill='black',font=font)
        draw.text((4,i*230+4),rec['clip']+' | source chronology and semantics unverified',fill='black',font=font)
    canvas.save(output/'LOCAL_ONLY_fixed_inputs.png')


def evaluate(model,batches,records,torch,step):
    model.eval()
    rows=[]
    with torch.no_grad():
        for batch, record in zip(batches,records):
            score=probabilities(torch,model.val_step(copy.deepcopy(batch)),10)[0]
            truth=record['recorded_label']
            p=float(score[truth].item())
            if p<=0: raise RuntimeError('Ground-truth probability underflow')
            rows.append(dict(recorded_label=truth,predicted_label=int(score.argmax().item()),
                             ce=-math.log(p), probabilities=score.tolist()))
    return dict(step=step,mean_ce=sum(r['ce'] for r in rows)/2,
                correct=sum(r['recorded_label']==r['predicted_label'] for r in rows),
                total=2,inputs=rows,scope='Same fixed training inputs; not validation')


def charts(output,report):
    os.environ.setdefault('MPLCONFIGDIR',str(output/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5))
    ax.plot([x['step'] for x in report['updates']],[x['loss'] for x in report['updates']],alpha=.65,label='Alternating single-input training CE')
    ax.plot([x['step'] for x in report['evaluations']],[x['mean_ce'] for x in report['evaluations']],marker='o',label='Same two fixed training inputs: mean CE in eval mode')
    ax.set(xlabel='Optimizer updates',ylabel='Cross entropy',title='Two fixed training inputs: optimization diagnostic\nNOT independent validation / NOT paper reproduction')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(output/'diagnostic_loss.png',dpi=150)
    fig.savefig(output/'diagnostic_loss.svg')
    plt.close(fig)


def train(cfg,output,batches,records,torch):
    from mmengine.optim import OptimWrapper
    device=torch.device('cuda:0')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
    model=make_model(cfg,device)
    optim_cfg=dict(cfg.optim_wrapper.optimizer)
    assert optim_cfg.pop('type')=='SGD'
    optimizer=torch.optim.SGD(model.parameters(),**optim_cfg)
    observations=[]
    class ObservedWrapper(OptimWrapper):
        def update_params(self,loss,*args,**kwargs):
            if not torch.isfinite(loss).all():
                raise RuntimeError('Nonfinite loss before backward or optimizer update')
            return super().update_params(loss,*args,**kwargs)

        def step(self,**kwargs):
            observation={}
            for name,module in [('backbone',model.backbone),('head',model.cls_head)]:
                params=[p for p in module.parameters() if p.requires_grad]
                if any(p.grad is None for p in params): raise RuntimeError('Missing '+name+' gradient')
                finite=torch.stack([p.grad.isfinite().all() for p in params]).all().item()
                norms=torch.stack([torch.linalg.vector_norm(p.grad.detach()) for p in params])
                norm=float(torch.linalg.vector_norm(norms).item())
                if not finite or not math.isfinite(norm) or norm<=0: raise RuntimeError('Invalid '+name+' gradient')
                observation[name]=dict(tensors=len(params),finite=True,l2_norm_before_clipping=norm)
            observations.append(observation)
            return super().step(**kwargs)
    wrapper=ObservedWrapper(optimizer,clip_grad=dict(cfg.optim_wrapper.clip_grad))
    tracked={}
    for prefix in ('backbone.','cls_head.'):
        key,p=next((n,p) for n,p in model.named_parameters() if n.startswith(prefix) and p.ndim>=2 and p.requires_grad)
        tracked[key]=p
    result=dict(updates=[],evaluations=[evaluate(model,batches,records,torch,0)],scope=SCOPE,
                parameters=sum(p.numel() for p in model.parameters()),max_updates=cfg.diagnostic.max_updates)
    assert 0<cfg.diagnostic.max_updates<=50
    start=time.perf_counter()
    with (output/'updates.jsonl').open('x',encoding='utf8',buffering=1) as log:
        for step in range(1,cfg.diagnostic.max_updates+1):
            idx=(step-1)%2
            model.train()
            before={n:p.detach().clone() for n,p in tracked.items()}
            values=model.train_step(copy.deepcopy(batches[idx]),wrapper)
            loss=float(values['loss'].item())
            if not math.isfinite(loss): raise RuntimeError('Nonfinite loss')
            changes={n:float((p.detach()-before[n]).abs().max().item()) for n,p in tracked.items()}
            if not all(x>0 for x in changes.values()): raise RuntimeError('Tracked backbone/head parameter did not update')
            row=dict(step=step,input_index=idx,loss=loss,gradients=observations[-1],parameter_max_abs_changes=changes)
            result['updates'].append(row); log.write(json.dumps(row)+'\n')
            if step%cfg.diagnostic.evaluate_every==0 or step==cfg.diagnostic.max_updates:
                evaluation=evaluate(model,batches,records,torch,step)
                result['evaluations'].append(evaluation)
                write_json(output/'progress.json',dict(status='RUNNING',last_evaluation=evaluation,updates=step,scope=SCOPE))
                print(json.dumps(dict(step=step,loss=loss,fixed_train_mean_ce=evaluation['mean_ce'],fixed_train_correct=evaluation['correct'],seconds=time.perf_counter()-start)),flush=True)
    result['optimization_seconds']=time.perf_counter()-start
    result['gpu_memory']=memory_stats(torch,device)
    checkpoint=output/'diagnostic_step_50.pth'
    torch.save(dict(state_dict={n:p.detach().cpu() for n,p in model.state_dict().items()},optimizer=optimizer.state_dict(),meta=dict(iter=len(result['updates']),scope=SCOPE,seed=cfg.randomness.seed)),checkpoint)
    original=result['evaluations'][-1]
    # MMEngine log variables can retain grad_fn and old model parameters.
    del values,before,optimizer,wrapper,tracked,model
    gc.collect(); torch.cuda.empty_cache()
    reloaded=make_model(cfg,device)
    result['checkpoint']=load_checkpoint_strict(reloaded,checkpoint,torch)
    reload_eval=evaluate(reloaded,batches,records,torch,len(result['updates']))
    delta=max(abs(a-b) for r1,r2 in zip(original['inputs'],reload_eval['inputs']) for a,b in zip(r1['probabilities'],r2['probabilities']))
    if delta>1e-6: raise RuntimeError('Strict reload prediction mismatch')
    result['reload_max_abs_probability_difference']=delta
    result['reload_evaluation']=reload_eval
    recent=result['evaluations'][-3:]
    result['memorization_observed']=len(recent)==3 and all(x['correct']==2 for x in recent) and recent[-1]['mean_ce']<result['evaluations'][0]['mean_ce']
    result['memorization_criterion']=cfg.diagnostic.success_observation
    result['pipeline_status']='PASS'
    result['memorization_status']='observed_within_budget' if result['memorization_observed'] else 'not_observed_within_budget'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','train'])
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    output=args.output_dir.resolve()
    if args.mode=='prepare': output.mkdir(parents=True,exist_ok=False)
    else:
        assert (output/'prepared.json').is_file() and not (output/'result.json').exists()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    report=dict(status='RUNNING',scope=SCOPE,started=now(),command=sys.argv)
    with (output/(args.mode+'_console.log')).open('x',encoding='utf8',buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
            try:
                import torch
                from mmengine.config import Config
                from mmengine.runner import set_random_seed
                report['framework']=register_framework()
                cfg=Config.fromfile(str(CONFIG))
                set_random_seed(cfg.randomness.seed,deterministic=True)
                torch.use_deterministic_algorithms(True)
                batches,records=inputs(cfg,output)
                if args.mode=='prepare':
                    cfg.dump(str(output/'resolved_config.py'))
                    preview(output,records)
                    write_json(output/'prepared.json',dict(scope=SCOPE,records=records,config_sha256=sha256(CONFIG),runner_sha256=sha256(__file__),versions={name:importlib.metadata.version(name) for name in ('torch','torchvision','mmaction2','mmengine','mmcv-lite','numpy')}))
                    print('PREPARED: review LOCAL_ONLY_fixed_inputs.png before invoking train',flush=True)
                    return
                prepared=json.loads((output/'prepared.json').read_text(encoding='utf8'))
                assert prepared['records']==records and prepared['config_sha256']==sha256(CONFIG) and prepared['runner_sha256']==sha256(__file__)
                report.update(config_sha256=sha256(CONFIG),runner_sha256=sha256(__file__),inputs=records,versions=prepared['versions'],changes_from_baseline=list(cfg.diagnostic.changes_from_baseline))
                print('Checking frozen baseline before diagnostic...',flush=True)
                report['baseline_before']=baseline_check()
                report.update(train(cfg,output,batches,records,torch))
                print('Checking frozen baseline after diagnostic...',flush=True)
                report['baseline_after']=baseline_check()
                charts(output,report)
                report.update(status='PASS',finished=now())
                write_json(output/'result.json',report)
                print(json.dumps(dict(status=report['status'],memorization_status=report['memorization_status'],result=str(output/'result.json'))),flush=True)
            except Exception as exc:
                report.update(status='FAIL',finished=now(),error=type(exc).__name__+': '+str(exc))
                write_json(output/(args.mode+'_failure.json'),report)
                traceback.print_exc()
                raise


if __name__=='__main__':
    main()
