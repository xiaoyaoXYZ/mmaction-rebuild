"""Summarize the bounded diagnostic without publishing frames or per-clip data."""
import hashlib
import json
import math
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'work_dirs/two_sample_diag_20261023_01'
OUT=ROOT/'audit_stage2'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((RUN/'result.json').read_text(encoding='utf8'))
assert r['status']==r['pipeline_status']=='PASS'
assert len(r['updates'])==50 and len(r['evaluations'])==11
assert digest(RUN/'executed_runner.py')==r['runner_sha256']
assert digest(ROOT/'configs/swin_tiny_two_sample_diagnostic.py')==r['config_sha256']
assert all(math.isfinite(x['loss']) for x in r['updates'])
assert all(all(v>0 for v in x['parameter_max_abs_changes'].values()) for x in r['updates'])
assert all(all(g['finite'] and g['l2_norm_before_clipping']>0 for g in x['gradients'].values()) for x in r['updates'])
assert r['reload_max_abs_probability_difference']==0
assert not r['baseline_after']['mismatches'] and r['baseline_after']['package_versions_unchanged']
for key in ('before','after'):
    assert r['baseline_'+key]['files_checked']==3126
assert digest(RUN/'diagnostic_step_50.pth')==r['checkpoint']['sha256']
summary=dict(status='PASS',date='2026-10-23',scope=r['scope'],
             initial_mean_ce=r['evaluations'][0]['mean_ce'],final_mean_ce=r['evaluations'][-1]['mean_ce'],
             updates=50,parameters=r['parameters'],memorization_status=r['memorization_status'],
             criterion=r['memorization_criterion'],
             evaluations=[dict(step=e['step'],mean_ce=e['mean_ce'],correct=e['correct'],total=2) for e in r['evaluations']],
             training_loss=[x['loss'] for x in r['updates']],
             all_losses_and_gradients_finite=True,all_tracked_backbone_and_head_updates_nonzero=True,
             strict_reload_probability_difference=r['reload_max_abs_probability_difference'],
             checkpoint_sha256=r['checkpoint']['sha256'],checkpoint_kept_local=True,
             baseline_before=r['baseline_before'],baseline_after=r['baseline_after'],
             versions=r['versions'],optimization_seconds=r['optimization_seconds'],
             training_peak_allocated_bytes=r['gpu_memory']['peak_allocated_bytes'],
             executed_runner_sha256=r['runner_sha256'],current_runner_sha256=digest(ROOT/'scripts/run_two_sample_diagnostic.py'),
             config_sha256=r['config_sha256'],result_sha256=digest(RUN/'result.json'),
             post_run_code_changes=['Added nonfinite-loss check before backward/update','Release final log-variable graph before strict reload; no new training performed after these guards'])
(OUT/'diagnostic_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
shutil.copyfile(RUN/'diagnostic_loss.svg',OUT/'diagnostic_loss.svg')
report=f'''# 两样本过拟合诊断记录

日期：2026-10-23。**训练管线通过；在预定的 50 次更新内观察到对两个固定训练输入的稳定记忆。** 此结果不代表独立验证、泛化性能、MAE 预训练或论文复现。

## 本次实际做了什么

在完整 MMAction2 Swin Tiny + 十类分类头上，从随机初始化开始，只读取原工程训练集的两个不同标签片段。保留原有 SGD(lr=0.001, momentum=0.9, weight_decay=0.0001)、FP32、梯度裁剪 40、with_cp、8 帧112×112及单样本批次，交替更新两个输入各 25 次。种子为 20261023；关闭 dropout/drop-path；时间采样和中心裁剪固定，缓存原始 CPU 张量，每次前处理前深拷贝，防止重复归一化。

实际选中标签 2、6 的训练片段，固定帧编号分别为 68,70,…,82 与 76,78,…,90。准备时重复解码逐元素相等；本地预览抽查未发现明显视角跳变。动作语义和原视频时间顺序仍未经数据提供者确认；这里只检查对既有整数标签的优化能力。未使用验证集或额外视频。

运行前规定最多 50 次更新，每 5 次在同样两个训练输入上用 eval 模式计算概率与平均交叉熵；观察条件为最后三次计划检查均命中两例且最终平均 CE 低于初始值。没有以论文准确率为目标或根据结果追加训练预算。

## 实测与确认方式

| 检查 | 结果 |
|---|---|
| 固定输入初始平均交叉熵 | {summary['initial_mean_ce']:.8f} |
| 第 50 次更新后平均交叉熵 | {summary['final_mean_ce']:.8f} |
| 固定训练输入命中 | 第 0 次 0/2，第 5 次 1/2，第 10～50 次的每次计划检查均 2/2 |
| 优化步骤 | 50 次，没有自动增加预算 |
| loss / 梯度 | 每步均有限；主干与分类头梯度整体非零 |
| 参数更新 | 每步检查的主干/分类头参数均发生非零变化 |
| 训练阶段峰值张量显存 | {summary['training_peak_allocated_bytes']/1024**2:.2f} MiB，不含全部驱动/显示开销，记录时点在重载前 |
| 50 次优化及期间计划评估耗时 | {summary['optimization_seconds']:.2f} 秒；不含环境导入、完整性检查、保存和绘图 |
| 检查点 | 已保存权重及优化器；strict 重载无缺失/额外键 |
| 重载一致性 | 两例十类概率的最大绝对差 0 |
| 原基线保全 | 运行前后各核对 3126 个冻结文件，均无变更；52 项环境版本一致 |

![诊断 loss 曲线](diagnostic_loss.svg)

蓝线是交替单例更新时的实际 loss，橙线是同样两例在 eval 模式下的平均 CE。**橙线不是独立验证 loss。** 未平滑、未补点。完整数值见 `diagnostic_summary.json`。

## 配置、代码和本地证据

- 配置：`configs/swin_tiny_two_sample_diagnostic.py`，SHA256 `{r['config_sha256']}`。
- 运行目录：`work_dirs/two_sample_diag_20261023_01/`；含 prepared.json、resolved_config.py、逐步 updates.jsonl、prepare_console.log、train_console.log、result.json、曲线和模型。
- 模型：`diagnostic_step_50.pth`，237,731,060 字节，SHA256 `{r['checkpoint']['sha256']}`；仅保留本地，不上传。
- 实际执行源码已逐字节保存在本地运行目录 `executed_runner.py`，SHA256 `{r['runner_sha256']}`，与 prepared/result 记录匹配。
- 运行完成后当前入口补充了两个保护：反向/更新前检查 loss 有限，以及重载前显式释放最后的日志计算图。数学训练配置未改；没有为这两个保护重新训练或改写本次结果。当前入口 SHA256 `{summary['current_runner_sha256']}`；本次结果对应上面的归档执行版。

当前入口分为 prepare/train；prepare 后先查看仅本地保存的输入预览，再 train。每次使用新目录，不覆盖旧运行：

```powershell
& .\\.venv\\Scripts\\python.exe .\\scripts\\run_two_sample_diagnostic.py prepare --output-dir .\\work_dirs\\YOUR_NEW_DIAGNOSTIC
& .\\.venv\\Scripts\\python.exe .\\scripts\\run_two_sample_diagnostic.py train --output-dir .\\work_dirs\\YOUR_NEW_DIAGNOSTIC
```

## 下一步边界

本诊断说明固定输入、标签编号、梯度更新和权重重载这条链路可工作，不能证明动作语义正确、真实时序完整或训练/验证来源独立。独立分类验证仍需完成原视频/会话分组及标签复核；缺少独立来源的类别需要补素材。

MAE/Tube Masking/MIS 尚未开始。现有资料不足以确定目标论文结构，需补充题名、DOI/链接或可核实原文，再单独实现掩码任务、重建损失和编码器迁移；不把任意遮挡分类增强称为原论文自监督。

GitHub 已复查为 private。同步仅包括诊断代码、配置、汇总数值与本报告/曲线；原始数据、输入预览、逐帧登记、原始运行日志及权重继续保留本地。
'''
(OUT/'diagnostic_report.md').write_text(report,encoding='utf8')
print(json.dumps({k:summary[k] for k in ('status','initial_mean_ce','final_mean_ce','updates','memorization_status','strict_reload_probability_difference')},indent=2))
