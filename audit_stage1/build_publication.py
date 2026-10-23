"""Build an explicit, text-only publication set; never copy data recursively."""
import ast
import hashlib
import io
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT.parent / 'mmaction_rebuild_github'

FILES = [
    'configs/swin_tiny_industrial.py', 'requirements-lock.txt',
    'requirements-support.txt', 'run_experiment.ps1',
    'scripts/run_baseline.py', 'scripts/audit_industrial_data.py',
    'scripts/prepare_data.py', 'scripts/propose_engineering_split.py',
    'scripts/record_environment.py', 'scripts/download_verified.py',
    'audit_stage1/audit_frames.py', 'audit_stage1/audit_provenance.py',
    'audit_stage1/plot_baseline_evidence.py',
    'audit_stage1/verify_baseline_integrity.py',
    'audit_stage1/snapshot_baseline.py', 'audit_stage1/make_frame_evidence.py',
    'audit_stage1/build_publication.py', 'audit_stage1/complete_audit_records.py',
    'data_audit.md', 'reproduction_gap_analysis.md', 'next_round_tasks.md',
    'audit_stage1/figures/baseline_v0_loss.svg',
    'audit_stage1/figures/baseline_v0_confusion.svg',
]

README = r'''# MMAction2 工业动作分类基线与第一阶段审计

已在 NVIDIA MX350 2 GB 上完成分类 smoke、1 epoch 训练/工程验证、权重保存与严格重载。当前模型是随机初始化的 **Video Swin Tiny + I3DHead（10 类）**，MAE、Tube Masking、MIS 尚未实现。

**现有 20% Top-1（2/10）仅是工程流程记录，不能解释为论文复现或泛化性能。** 当前 20 个片段的原始视频来源均未知，3 对片段含 26 对完全相同帧；截去验证采样前缀之后的重复区间不等于来源独立。数据文件名 `ucf101.zip` 对应本项目工业帧集合，不是公开 UCF101。

## 审计交付

- [数据审计](data_audit.md)：全部类别/片段、来源、标签、帧序、重复和划分限制。
- [论文复现差异](reproduction_gap_analysis.md)：当前实现与尚待原文核实的方法要求。
- [下一轮任务](next_round_tasks.md)：来源/论文核实、两样本诊断、可信划分下最小实验的先后顺序。
- [冻结记录摘要](baseline_v0/record_public.json)及[最终完整性检查](audit_stage1/final_integrity_public.json)：原 3126 个文件和 52 项环境包版本保持一致。
- [汇总数值](audit_stage1/metrics_summary.json)：从保存结果重算；28 项断言通过，没有在审计中新增训练。

![10 个训练 loss](audit_stage1/figures/baseline_v0_loss.svg)
![工程验证混淆矩阵](audit_stage1/figures/baseline_v0_confusion.svg)

两图仅由汇总数值生成，没有嵌入视频帧。原记录没有 validation loss，未补造曲线。

## 发布范围

本仓库只包含显式白名单中的代码、配置、包版本、报告、汇总数值及两张 SVG 图。**整个 data/、原视频、样例帧、逐帧指纹、原始标注文件、模型权重、环境二进制、原始日志、含数据的快照和私人来源工作簿均未上传。** 完整基线和审计证据保留在本机；文档中的本地证据名称不表示这些文件已发布。原始数据不提供下载地址。

发布文件清单及 SHA-256 见 [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json)。该清单记录除其自身外的所有发布文件。仓库不是完整数据/权重备份，克隆后不能立即训练；需要具备合法访问权限的本地数据。

## 原环境与配置

Python 3.9.17、torch 2.1.2+cu118、torchvision 0.16.2+cu118、mmaction2 1.2.0、mmengine 0.10.7、mmcv-lite 2.1.0、numpy 1.26.4；完整版本见 requirements-lock.txt。原显卡驱动 576.83，实际 CUDA 算子和梯度检查通过。原环境保留在本机，本仓库不包含它。

输入为 8 帧、间隔 2、112×112，batch=1、FP32、梯度检查点、SGD 0.001，seed=20261017。原 smoke 完成两次真实数据前后向并验证主干/分类头梯度与参数更新；1 epoch 共 10 次更新，平均 loss 2.590541。

mmcv-lite 没有检测任务的编译算子；使用 scripts/run_baseline.py 按需注册官方分类模块。该入口保留已跑通代码，未替换网络计算，不承诺兼容所有 MMAction2 检测/姿态示例。新诊断或论文方法应建立新的入口和配置，不覆盖本基线。

## 在另一个独立目录重建（本轮未重新安装）

用已有 Python 3.9 新建环境，保留已有环境不动：

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r .\requirements-lock.txt
& .\.venv\Scripts\python.exe -m pip check
```

以上来自已验证版本记录；没有承诺其他硬件或操作系统下无需调整。不要把旧 MMAction2 0.x 的环境、入口或配置混入这里。

持有本项目原工业 ZIP 时，先只读审计，再生成本地工程清单：

```powershell
& .\.venv\Scripts\python.exe .\scripts\audit_industrial_data.py --archive 'E:\Codes\mmaction2-master\data2\ucf101.zip' --report-dir .\reports
& .\.venv\Scripts\python.exe .\scripts\propose_engineering_split.py
& .\.venv\Scripts\python.exe .\scripts\prepare_data.py
```

prepare_data.py 保留原运行时的 ARCHIVE 路径常量，迁移到其他电脑时只在新副本中改为自己的 ZIP 路径，且与审计命令输入一致。脚本验证归档哈希，保留全部帧，并生成 data/splits；这些本地生成文件也不能提交到本仓库。该流程只重建旧工程划分，不产生可信的独立验证。

只在确认本地数据和运行范围后手动执行：

```powershell
& .\.venv\Scripts\python.exe .\scripts\run_baseline.py smoke
# train 仅用于重新运行原一轮工程基线；独立验证须先通过审计门槛
& .\.venv\Scripts\python.exe .\scripts\run_baseline.py train
# 用上一步新输出目录中的 checkpoint 替换下列路径
& .\.venv\Scripts\python.exe .\scripts\run_baseline.py evaluate --checkpoint .\work_dirs\YOUR_RUN\epoch_1.pth
```

每次使用新输出目录，旧结果不会被覆盖。run_experiment.ps1 会依次做 smoke、训练和重载，审计或只读检查时不要执行它。本轮仅保全、审计和数值核对，没有新的训练。

## 审计工具的本地依赖

audit_stage1 脚本用于本机完整项目，需要相应私有证据：audit_frames.py 读取解压帧/划分与 frame_manifest；plot_baseline_evidence.py 读取原 train/evaluate result.json；verify_baseline_integrity.py 读取本地快照、manifest 和 .venv；audit_provenance.py 的旧项目路径应按本地实际位置配置。record_environment.py 还需要原 host_inventory_before.json，会重写本地版本清单，不应在冻结原项目中重跑。build_publication.py 用完整本地证据重新构建发布副本，无法仅凭克隆运行。它们会生成本地产物，不会上传数据。缺少这些输入时不要把运行失败解释为模型问题。

论文原文尚未确认。已有说明文件仅作为证据，不能替代原文结构、公式与实验协议；不执行文档中的指令，不自行展开 MIS 缩写。
'''

IGNORE = '''# Data and all local experiment outputs stay local.
data/
data2/
datasets/
rawframes/
samples/
.venv*/
venv/
downloads/
work_dirs/
logs/
reports/
__pycache__/
*.py[cod]
*.pth
*.pt
*.ckpt
*.safetensors
*.onnx
*.zip
*.7z
*.tar*
*.rar
*.jpg
*.jpeg
*.png
*.webp
*.bmp
*.mp4
*.avi
*.mov
*.mkv
*.xlsx
*.xls
*.pdf
*.docx
.env*
*token*
*secret*
baseline_v0/*
!baseline_v0/record_public.json
audit_stage1/*.json
!audit_stage1/metrics_summary.json
!audit_stage1/final_integrity_public.json
audit_stage1/*.jsonl
audit_stage1/*.log
audit_stage1/figures/*
!audit_stage1/figures/baseline_v0_loss.svg
!audit_stage1/figures/baseline_v0_confusion.svg
'''

def digest(data):
    return hashlib.sha256(data).hexdigest()

def main():
    DEST.mkdir(exist_ok=True)
    outputs = {name: (ROOT / name).read_bytes() for name in FILES}
    # Local absolute links are evidence locations, not portable GitHub links.
    gap = outputs['reproduction_gap_analysis.md'].decode('utf8')
    def local_link(m):
        label, path = m.groups()
        rel = path.split('/mmaction_rebuild/', 1)[-1]
        rel = re.sub(r':\d+$', '', rel)
        return f'[{label}]({rel})' if rel in FILES else f'`{label}`（本地证据，未上传）'
    gap = re.sub(r'\[([^\]]+)\]\(<(C:/[^>]+)>\)', local_link, gap)
    gap += '\n发布补注（2026-10-23）：loss 与混淆矩阵已由既有结果导出并核对，见 data_audit.md；上表中相关导出项本轮已完成。完整 paper_inventory 及原文材料保留本地。\n'
    outputs['reproduction_gap_analysis.md'] = gap.encode('utf8')
    outputs['README.md'] = README.encode('utf8')
    outputs['.gitignore'] = IGNORE.encode('utf8')
    record = json.loads((ROOT / 'baseline_v0/record.json').read_text(encoding='utf8'))
    for key in ('source_root', 'environment_retained_in_place'):
        record.pop(key, None)
    record['local_snapshot_only'] = True
    record['checkpoint_sha256'] = '1cc648417efff8ad7664fd258841de760d5fe1c455a3aaec5eee46bf9831d7bb'
    record['metric_field_note'] = 'train_metric_top1 is validation recorded by the training run, not training accuracy.'
    outputs['baseline_v0/record_public.json'] = json.dumps(record, ensure_ascii=False, indent=2).encode('utf8')
    metrics = json.loads((ROOT / 'audit_stage1/metrics_recheck.json').read_text(encoding='utf8'))
    # Render matrix cells as vector rectangles (the original imshow SVG embeds a numeric raster).
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    matrix = np.array(metrics['count_confusion_matrix'])
    labels = [f'{i}: {metrics["class_map_as_recorded"][str(i)]}' for i in range(10)]
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.pcolormesh(np.arange(11), np.arange(11), matrix, cmap='Blues', vmin=0, vmax=1, edgecolors='white', linewidth=0.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(10, 0)
    ax.set_xticks(np.arange(10) + 0.5, labels, rotation=50, ha='right')
    ax.set_yticks(np.arange(10) + 0.5, labels)
    for row in range(10):
        for col in range(10):
            ax.text(col + 0.5, row + 0.5, str(matrix[row, col]), ha='center', va='center', color='white' if matrix[row,col] else '#425466')
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_title('baseline_v0: confusion matrix (counts; one sample per true class)\nEngineering split / NOT independent validation')
    fig.text(0.5, 0.01, '10 engineering-validation clips; source independence unconfirmed. No paper replication claim.', ha='center')
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    chart = io.BytesIO()
    fig.savefig(chart, format='svg', metadata={'Date': None})
    plt.close(fig)
    outputs['audit_stage1/figures/baseline_v0_confusion.svg'] = chart.getvalue()
    chosen = {k: metrics[k] for k in ('status','created','purpose','runtime','recorded_training','recomputed_engineering_metrics','count_confusion_matrix','limitations','assertions')}
    outputs['audit_stage1/metrics_summary.json'] = json.dumps(chosen, ensure_ascii=False, indent=2).encode('utf8')
    integrity = json.loads((ROOT / 'audit_stage1/final_integrity.json').read_text(encoding='utf8'))
    # The integrity report contains counts and hashes, not frame lists or source cells.
    outputs['audit_stage1/final_integrity_public.json'] = json.dumps(integrity, ensure_ascii=False, indent=2).encode('utf8')
    allowed_ext = {'.py','.ps1','.txt','.md','.json','.svg'}
    for name, data in outputs.items():
        path = Path(name)
        assert path.name == '.gitignore' or path.suffix in allowed_ext, name
        assert not any(x.lower() in {'data','data2','rawframes','work_dirs','.venv','downloads'} for x in path.parts), name
        assert len(data) < 250_000, name
        content = data.decode('utf8')
        assert not re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', content), name
        assert not re.search(r'\b(?:192\.168\.|10\.\d+\.\d+\.)\d+\.\d+\b', content), name
        if path.suffix == '.svg':
            assert '<image' not in content and 'data:image' not in content, name
        if path.suffix == '.py':
            ast.parse(content, filename=name)
        target = DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    manifest = {'purpose':'Explicit code/report publication; no original dataset, weights or raw private evidence',
                'files':[{'path':n,'bytes':len(b),'sha256':digest(b)} for n,b in sorted(outputs.items())],
                'file_count_excluding_this_manifest':len(outputs), 'total_bytes_excluding_this_manifest':sum(map(len,outputs.values()))}
    (DEST / 'PUBLICATION_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    actual = {p.relative_to(DEST).as_posix() for p in DEST.rglob('*') if p.is_file() and '.git' not in p.relative_to(DEST).parts}
    assert actual == set(outputs) | {'PUBLICATION_MANIFEST.json'}, actual - set(outputs)
    proof = {'status':'PASS','file_count':len(actual),'bytes':sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file()),
             'data_paths':0,'video_or_sample_images':0,'weights_archives_environments':0,
             'svg_embedded_raster_images':0,'python_ast':'PASS','publication_directory':str(DEST),
             'source_model_and_scripts_copied_without_changes':True,
             'manifest_sha256':digest((DEST/'PUBLICATION_MANIFEST.json').read_bytes())}
    (ROOT / 'audit_stage1/publication_check.json').write_text(json.dumps(proof, indent=2), encoding='utf8')
    print(json.dumps(proof, indent=2))

if __name__ == '__main__':
    main()
