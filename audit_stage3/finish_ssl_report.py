"""Verify a completed SSL diagnostic; publish aggregate numbers, not inputs.

No model imports, GPU calls, video reads or training. Checkpoints are hashed,
not loaded. Optional independent encoder export validation is linked only if
its evidence file exists and passes explicit checks.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE2_SHA = 'dfcde43d7b24de052138b6a413161929fca03fc4cb7a5635bc9fb0f72d380b40'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, positive=False):
    return isinstance(value, (float, int)) and math.isfinite(value) and (value > 0 if positive else True)


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def close_to_zero(value):
    return finite(value) and 0 <= value <= 1e-6


def verify(run, result):
    require(result['status'] == 'PASS', 'The run must have status PASS')
    rows = result['updates']
    require(len(rows) == 20 and [r['step'] for r in rows] == list(range(1, 21)), 'Expected exactly 20 ordered updates')
    for row in rows:
        require(finite(row['loss']) and row['loss'] >= 0, 'Invalid update loss')
        require(row['masked_tokens'] == 1568, 'Expected 1568 masked tokens per update')
        require(finite(row['grad_norm_before_clip'], positive=True), 'Invalid total gradient norm')
        groups = row['gradients']
        require(set(groups) == {'encoder', 'decoder', 'mask_token'}, 'Gradient coverage differs')
        require(all(g['finite'] is True and g['tensors'] > 0 and finite(g['l2_norm'], positive=True)
                    for g in groups.values()), 'Missing, zero or nonfinite gradients')
        changes = row['parameter_max_abs_changes']
        require(set(changes) == {'encoder.patch_embed.proj.weight', 'encoder.mask_token',
                                 'decoder.0.weight', 'decoder.2.weight'}, 'Tracked parameter coverage differs')
        require(all(finite(v, positive=True) for v in changes.values()), 'Tracked parameter did not change finitely')
        require(valid_sha(row['mask_sha256']), 'Invalid mask hash')
    require(len({r['mask_sha256'] for r in rows}) == 20, 'Training mask hashes must differ at every update')
    require(result['masked_tokens_per_input'] == [1568, 1568], 'Fixed diagnostic mask count differs')
    require(len(result['fixed_mask_sha256']) == 2 and all(valid_sha(h) for h in result['fixed_mask_sha256']), 'Invalid fixed mask hashes')
    evaluations = result['evaluations']
    require([e['step'] for e in evaluations] == [0, 5, 10, 15, 20], 'Unexpected evaluation schedule')
    for e in evaluations:
        values = e['input_mse']
        require(len(values) == 2 and all(finite(v) and v >= 0 for v in values), 'Invalid diagnostic losses')
        require(finite(e['mean_mse']) and math.isclose(e['mean_mse'], sum(values) / 2, rel_tol=1e-12, abs_tol=1e-12), 'Mean mismatch')
        require(e['scope'] == 'Same TRAIN inputs, fixed masks, no independent validation', 'Diagnostic scope changed')
    p = result['preflight']
    require(p['status'] == 'PASS' and p['output_shape'] == [1, 4, 28, 28, 96], 'Preflight shape/status mismatch')
    require(p['stage_shapes'] == [[1, 96, 4, 28, 28], [1, 192, 4, 14, 14],
                                  [1, 384, 4, 7, 7], [1, 768, 4, 4, 4]], 'Encoder scale mismatch')
    require(p['masked_input_gradient_max'] == 0 and p['hidden_pixel_perturb_prediction_max'] == 0, 'Hidden pixel leakage')
    require(p['target_detached'] is True and finite(p['visible_input_gradient_max'], positive=True), 'Target/input gradient check failed')
    require(close_to_zero(result['strict_reload_prediction_max_abs_difference']), 'SSL strict reload differs')
    transfer = result['transfer']
    require(transfer['status'] == 'PASS' and transfer['loaded_backbone_keys'], 'Transfer failed')
    require(not transfer['missing_backbone_keys'] and not transfer['shape_mismatches'], 'Transfer incomplete')
    require(transfer['randomly_initialized_head_unchanged'] is True and transfer['classifier_output_shape'] == [1, 10], 'Classification head check failed')
    require(close_to_zero(transfer['no_mask_feature_max_abs_difference']), 'Transferred features differ')
    require(close_to_zero(transfer['classifier_strict_reload_score_max_abs_difference']), 'Classifier reload differs')
    for phase in ('before', 'after'):
        baseline = result['baseline_' + phase]
        require(baseline['files_checked'] == 3126 and not baseline['mismatches'], 'Baseline integrity failed')
        require(baseline['package_versions_unchanged'] is True and baseline['package_count'] == 52, 'Environment integrity failed')
        require(result['stage2_checkpoint_' + phase] == STAGE2_SHA, 'Stage 2 model differs')
    require(digest(ROOT / 'work_dirs/two_sample_diag_20261023_01/diagnostic_step_50.pth') == STAGE2_SHA, 'Current stage 2 checkpoint differs')
    checkpoint = run / result['checkpoint']['path']
    require(checkpoint.resolve().parent == run.resolve(), 'Checkpoint must remain inside run directory')
    require(checkpoint.stat().st_size == result['checkpoint']['bytes'], 'SSL checkpoint size differs')
    require(digest(checkpoint) == result['checkpoint']['sha256'], 'SSL checkpoint hash differs')
    require(digest(run / 'encoder_for_classification.pth') == transfer['encoder_checkpoint_sha256'], 'Encoder export hash differs')
    require(digest(run / 'classifier_transfer_check.pth') == transfer['classifier_checkpoint_sha256'], 'Classifier checkpoint hash differs')
    require(result['source_sha256'], 'Missing executed source hashes')
    for relative, expected in result['source_sha256'].items():
        require(valid_sha(expected) and digest(run / 'executed_source' / Path(relative).name) == expected, 'Archived executed source differs: ' + relative)
    require(finite(result['optimization_seconds'], positive=True), 'Invalid timing')
    for key in ('gpu_memory_training', 'gpu_memory_including_reload_transfer'):
        require(result[key]['cuda_used'] is True and result[key]['peak_allocated_bytes'] > 0, 'Missing measured GPU memory')


def export_evidence(run, result):
    path = run / 'encoder_export_check.json'
    evidence = json.loads(path.read_text(encoding='utf8'))
    require(evidence['status'] == 'PASS' and evidence['strict_load'] == 'PASS', 'Export reload failed')
    expected = result['transfer']['loaded_backbone_keys']
    require(evidence['checked_keys'] == sorted(expected), 'Export checked keys differ')
    require(evidence['expected_keys'] == evidence['loaded_keys'] == evidence['exact_serialized_source_tensor_matches'] == len(expected), 'Export key count differs')
    require(not evidence['missing_keys'] and not evidence['unexpected_keys'] and not evidence['mismatches'], 'Export mismatch')
    require(evidence['allowed_excluded_source_keys'] == result['transfer']['excluded_source_keys'], 'Export exclusions differ')
    require(evidence['encoder_export_sha256'] == result['transfer']['encoder_checkpoint_sha256'], 'Export evidence hash differs')
    require(evidence['full_ssl_checkpoint_sha256'] == result['checkpoint']['sha256'], 'Full source evidence hash differs')
    require(evidence['verifier_sha256'] == digest(ROOT / 'audit_stage3/verify_encoder_export.py'), 'Verifier changed')
    return dict(status='PASS', device='cpu', loaded_keys=len(expected), exact_tensor_matches=len(expected),
                strict_load='PASS', evidence_sha256=digest(path), verifier_sha256=evidence['verifier_sha256'],
                note='Independently read exported file and full SSL checkpoint; strict load into official Tiny backbone and exact tensor comparison')


def summary_from(run, result):
    values = [r['loss'] for r in result['updates']]
    evaluations = [dict(step=e['step'], mean_mse=e['mean_mse']) for e in result['evaluations']]
    t = result['transfer']
    started = datetime.fromisoformat(result['started'].replace('Z', '+00:00'))
    finished = datetime.fromisoformat(result['finished'].replace('Z', '+00:00'))
    return dict(status='PASS', report_date='2026-10-27', run_id=run.name,
                started=result['started'], finished=result['finished'],
                scope='Engineering prototype on the same two TRAIN inputs; no independent validation, accuracy or paper reproduction claim',
                paper_doi='10.1109/ACCESS.2024.3455749', updates=20, parameters=result['parameters'],
                mask_mode='spatial_tube', mask_ratio=0.5, masked_tokens_per_input=1568,
                engineering_assumptions=result['assumptions'],
                initial_fixed_train_mean_mse=evaluations[0]['mean_mse'], final_fixed_train_mean_mse=evaluations[-1]['mean_mse'],
                minimum_fixed_train_mean_mse=min(e['mean_mse'] for e in evaluations),
                minimum_update_mse=min(values), last_update_mse=values[-1],
                fixed_train_evaluations=evaluations, update_losses=values,
                fixed_train_curve_increases=sum(b['mean_mse'] > a['mean_mse'] for a, b in zip(evaluations, evaluations[1:])),
                all_step_losses_and_gradients_finite=True, all_tracked_parameters_changed_each_step=True,
                all_training_mask_hashes_unique=True, hidden_input_gradient_max=0.0,
                hidden_pixel_perturb_prediction_max=0.0, target_detached=True,
                ssl_strict_reload_prediction_max_abs_difference=result['strict_reload_prediction_max_abs_difference'],
                transfer=dict(status='PASS', loaded_backbone_key_count=len(t['loaded_backbone_keys']),
                              excluded_source_keys=t['excluded_source_keys'], missing_backbone_keys=0,
                              shape_mismatches=0, random_head_unchanged=True,
                              no_mask_feature_max_abs_difference=t['no_mask_feature_max_abs_difference'],
                              classifier_reload_score_max_abs_difference=t['classifier_strict_reload_score_max_abs_difference'],
                              note='In-memory backbone migration and full classifier checkpoint reload; no classification fine-tuning or accuracy evaluation'),
                independent_encoder_export_reload=export_evidence(run, result),
                baseline_integrity=dict(before_files_checked=3126, after_files_checked=3126, mismatches=0,
                                        package_count=52, package_versions_unchanged=True, stage2_checkpoint_unchanged=True),
                device=result['device'], versions=result['versions'],
                optimization_and_scheduled_evaluation_seconds=result['optimization_seconds'],
                run_record_elapsed_seconds=(finished - started).total_seconds(),
                training_peak_allocated_bytes=result['gpu_memory_training']['peak_allocated_bytes'],
                training_peak_reserved_bytes=result['gpu_memory_training']['peak_reserved_bytes'],
                reload_transfer_peak_allocated_bytes=result['gpu_memory_including_reload_transfer']['peak_allocated_bytes'],
                checkpoint_sha256=result['checkpoint']['sha256'], checkpoint_bytes=result['checkpoint']['bytes'],
                encoder_checkpoint_sha256=t['encoder_checkpoint_sha256'], classifier_checkpoint_sha256=t['classifier_checkpoint_sha256'],
                stage2_checkpoint_sha256=STAGE2_SHA, checkpoints_kept_local=True,
                source_sha256=result['source_sha256'], result_sha256=digest(run / 'result.json'),
                report_generator_sha256=digest(Path(__file__).resolve()))


def plot(s, run, output):
    os.environ.setdefault('MPLCONFIGDIR', str(run / 'plot_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'svg.hashsalt': 'ssl-prototype-stage3', 'svg.fonttype': 'none',
                                'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.plot(range(1, 21), s['update_losses'], marker='.', linewidth=1.4,
            label='Single TRAIN input / fresh mask / before update')
    ev = s['fixed_train_evaluations']
    ax.plot([e['step'] for e in ev], [e['mean_mse'] for e in ev], marker='o', linewidth=1.8,
            label='Same two TRAIN inputs / fixed masks / after update')
    ax.set(xlabel='Completed optimizer updates', ylabel='Masked RGB patch MSE',
           title='Engineering SSL diagnostic: training inputs only')
    ax.set_xticks(range(0, 21, 5))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(output / 'ssl_loss.svg', metadata={'Date': None})
    fig.savefig(run / 'ssl_loss.png', dpi=160)
    plt.close(fig)
    svg = (output / 'ssl_loss.svg').read_text(encoding='utf8')
    require('<image' not in svg.lower() and 'data:image' not in svg.lower(), 'SVG contains image data')


def report_text(s):
    return f'''# 自监督重建工程原型：20 次更新记录

训练日期：2026-10-26。报告日期：2026-10-27。运行：`{s['run_id']}`。

**前向、反向、遮挡隔离、权重保存与分类主干迁移检查通过。** 本轮采用明确列出的工程假设，只验证实现链路；不等于 E-VIST/MIS 完整复现，没有测量独立分类性能。

## 本次做了什么

保留原 MMAction2 分类基线及两样本诊断结果，新增独立配置和重建模型。从随机初始化开始，只用此前已检查的两个固定 TRAIN 输入，交替进行 20 次 AdamW 更新，批次为 1。时间采样和裁剪固定，每次更新重新生成遮挡；每 5 次更新在同样两个 TRAIN 输入上，使用固定遮挡及 eval 模式计算平均重建误差。没有使用独立验证集，重建损失不使用类别标签。

输入为 8 帧、112×112 RGB。2×4×4 patch 得到 4×28×28=3136 个 token，每次精确遮挡 1568 个。空间遮挡图案跨时间共享，三个 RGB 通道一起遮挡。目标为分块后的 RGB/255，目标张量断开梯度；仅在被遮挡 token 上计算均方误差。

## 与原文的关系及工程假设

依据用户提供的原文（DOI：10.1109/ACCESS.2024.3455749）开展核对；论文页码与未决问题见 `paper_method_spec.md`。以下均为本原型显式选择，不能补写为原论文既定结构。

| 项目 | 本轮工程实现及差异 |
|---|---|
| 主干与输入 | 官方 MMAction2 Swin-Tiny、8×112×112；论文实验采用 Video Swin-Base、16 帧及 224×224 设置 |
| 遮挡解释 | 采用公式 11、图 7/8 所支持的跨时间共享空间 mask；公式 16 的连续整帧遮挡描述仍存在冲突 |
| 编码方式 | patch embedding 后替换为可学习密集 mask 向量；不声称实现已确定的稀疏、仅可见 token 编码器 |
| 多尺度对齐 | 四个归一化 stage 特征用最近邻插值对齐至最初 token 网格；原文未给出这一确切规则 |
| 解码器 | 拼接 96+192+384+768=1440 通道，Linear(1440,48)→GELU→Linear(48,96)；没有解码 Transformer block。48 是把 C/2 解释为 Tiny 初始宽度一半，尚非原文已确认配置 |
| 目标与距离 | detached RGB/255、无逐 patch 目标标准化、masked-only MSE；原文公式 13 的 h、T、D 未充分定义，不能认定原文就是像素 MSE |
| 优化预算 | AdamW，lr=1e-4，betas=(0.9,0.999)，weight_decay=0.05，固定学习率、梯度裁剪 5、20 次更新；不同于论文 lr=0.003、warmup/cosine 及完整预训练预算 |
| 数据与迁移 | 无 K400，只用两个既有训练输入；仅验证权重迁移，没有分类微调或独立测试 |
| MIS | 不增造 MIS 模块或缩写全称；论文中的命名及实现仍待进一步核实 |

## 实测与确认方式

| 检查 | 实测 |
|---|---|
| 固定训练输入初始平均 MSE | {s['initial_fixed_train_mean_mse']:.8f} |
| 第 20 次更新后固定训练输入平均 MSE | {s['final_fixed_train_mean_mse']:.8f} |
| 5 个固定检查点中的最小平均 MSE | {s['minimum_fixed_train_mean_mse']:.8f} |
| 20 次更新的最小单步 MSE | {s['minimum_update_mse']:.8f} |
| 第 20 次更新前单步 MSE | {s['last_update_mse']:.8f} |
| 更新与梯度 | 完整 20 步；loss 与三组梯度均有限、组范数非零；每步跟踪的 embedding、mask token、两层 decoder 参数均发生非零有限变化 |
| 遮挡比例 | 所有训练和固定检查 mask 均为 1568/3136；20 次训练 mask 哈希各异 |
| 目标隔离 | 遮挡像素输入梯度最大值 0；大幅扰动隐藏像素后预测最大差值 0；可见输入梯度非零、目标无梯度 |
| 重建检查点 strict 重载 | 预测最大绝对差 {s['ssl_strict_reload_prediction_max_abs_difference']:.8g}，要求 ≤1e-6 |
| 内存编码器迁移 | {s['transfer']['loaded_backbone_key_count']} 个主干 state 键严格匹配，缺失键/形状差异均 0；随机分类头未覆盖 |
| 无 mask 特征一致性 | 迁移前后最大绝对差 {s['transfer']['no_mask_feature_max_abs_difference']:.8g} |
| 完整分类检查点 strict 重载 | 分数最大绝对差 {s['transfer']['classifier_reload_score_max_abs_difference']:.8g}；仅检查迁移与重载，不计算分类准确率 |
| GPU | {s['device']} |
| 训练阶段峰值张量分配显存 | {s['training_peak_allocated_bytes']/1024**2:.2f} MiB，含前置诊断，不含全部驱动/显示开销 |
| 包含重载及迁移后的峰值张量分配显存 | {s['reload_transfer_peak_allocated_bytes']/1024**2:.2f} MiB |
| 20 次优化及期间计划评估 | {s['optimization_and_scheduled_evaluation_seconds']:.2f} 秒 |
| 运行记录起止时间差 | {s['run_record_elapsed_seconds']:.2f} 秒，包含导入、完整性核对与保存等，不等于纯训练耗时 |
| 原结果保全 | 前后各核对 3126 个 baseline_v0 文件及 52 项包版本，无变化；旧两样本模型 SHA256 一致 |

![训练输入重建误差](ssl_loss.svg)

蓝线是交替单个训练输入在新随机遮挡下、参数更新前的 loss；橙线是同样两个训练输入在固定遮挡下、相应更新完成后的平均 MSE。**橙线不是独立验证 loss。** 曲线未平滑或补点，不要求单调下降；固定均值相邻上升次数为 {s['fixed_train_curve_increases']}。两条曲线输入/遮挡条件不同，不能将相邻值视为同一批次改进幅度。

## 保存与复核

- 本地目录 `work_dirs/{s['run_id']}/` 保存解析配置、执行源码副本、逐步日志、结果与模型；数据、预览、权重不进入发布目录。
- `ssl_step_20.pth`：{s['checkpoint_bytes']:,} 字节，SHA256 `{s['checkpoint_sha256']}`。
- `encoder_for_classification.pth`：SHA256 `{s['encoder_checkpoint_sha256']}`。
- `classifier_transfer_check.pth`：SHA256 `{s['classifier_checkpoint_sha256']}`；分类头仍为随机初始化。
- 原两样本模型 SHA256 `{s['stage2_checkpoint_sha256']}`，前后及当前文件复核一致。
- 完整 result.json 的 SHA256 `{s['result_sha256']}`；生成汇总前再次核对新旧权重及归档执行源码哈希，没有重新加载模型或启动 GPU。
- 另行在CPU上实际重读编码器导出文件与完整自监督检查点：183项参数/缓冲区逐项精确一致，严格载入官方Tiny骨干通过。仅显式排除mask token及norm0–2共7项；证据为本地 `encoder_export_check.json`，SHA256 `{s['independent_encoder_export_reload']['evidence_sha256']}`。报告生成器核对这一证据，未重跑训练。
- `ssl_summary.json` 仅保存汇总数值、版本、哈希，不含片段身份、逐帧信息或预测张量；`ssl_loss.svg` 为纯数值矢量图，本地 PNG 供显示检查。

## 结论范围与下一步

本轮支持“重建训练与迁移链路可运行”。不能证明学到了可迁移的动作语义，不能比较论文准确率，不能解决原视频来源未知及训练/验证同源风险。

下一轮先核实原文的目标表示、解码结构、遮挡冲突与作者代码，并补足原视频/会话分组，形成可信独立划分；再在同一划分和预算下比较随机初始化与预训练权重，记录分类 loss、独立混淆矩阵和准确率。数据不足时保留阻碍，不把训练重建误差解释为泛化证据。
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'work_dirs/ssl_proto_20261026_01')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'audit_stage3')
    args = parser.parse_args()
    run, output = args.run_dir.resolve(), args.output_dir.resolve()
    run.relative_to((ROOT / 'work_dirs').resolve())
    output.relative_to(ROOT.resolve())
    require((run / 'result.json').is_file(), 'No completed result.json; report generation does not start training')
    result = json.loads((run / 'result.json').read_text(encoding='utf8'))
    verify(run, result)
    summary = summary_from(run, result)
    output.mkdir(parents=True, exist_ok=True)
    plot(summary, run, output)
    (output / 'ssl_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    (output / 'ssl_prototype_report.md').write_text(report_text(summary), encoding='utf8')
    print(json.dumps(dict(status='PASS', run_id=run.name, updates=20,
                         initial_fixed_train_mean_mse=summary['initial_fixed_train_mean_mse'],
                         final_fixed_train_mean_mse=summary['final_fixed_train_mean_mse'],
                         output_dir=str(output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
