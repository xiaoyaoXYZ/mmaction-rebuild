"""Freeze the existing baseline without editing any baseline file."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone, timedelta
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'baseline_v0'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    files = [ROOT / n for n in ('README.md', 'requirements-lock.txt',
             'requirements-support.txt', 'run_experiment.ps1', 'reproduction_records.zip',
             '.venv/pyvenv.cfg')]
    for folder in ('configs', 'scripts', 'reports', 'logs', 'work_dirs', 'data'):
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files = sorted(set(files))
    old_manifest = json.loads((ROOT/'reports/artifact_manifest.json').read_text(encoding='utf8'))
    mismatches = [r['path'] for r in old_manifest if not (ROOT/r['path']).is_file() or sha(ROOT/r['path']) != r['sha256']]
    assert not mismatches, ('Baseline differs from prior final manifest', mismatches)
    manifest = []
    temporary = OUT/'snapshot.zip.partial'
    with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for p in files:
            name = p.relative_to(ROOT).as_posix()
            digest = sha(p)
            z.write(p, name)
            manifest.append(dict(path=name, bytes=p.stat().st_size, sha256=digest))
    with zipfile.ZipFile(temporary) as z:
        for item in manifest:
            h = hashlib.sha256()
            with z.open(item['path']) as f:
                for b in iter(lambda:f.read(1024*1024), b''):
                    h.update(b)
            assert h.hexdigest() == item['sha256'], item['path']
            assert sha(ROOT/item['path']) == item['sha256'], ('Source changed', item['path'])
    final = OUT/'snapshot.zip'
    temporary.replace(final)
    freeze = subprocess.run([sys.executable,'-m','pip','freeze','--all'],capture_output=True,text=True,check=True).stdout
    original = (ROOT/'requirements-lock.txt').read_text(encoding='utf8')
    assert sorted(freeze.splitlines()) == sorted(original.splitlines()), 'Environment package versions changed'
    (OUT/'environment-freeze.txt').write_text(freeze,encoding='utf8')
    record = dict(status='VERIFIED', recorded=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                  source_root=str(ROOT), snapshot='snapshot.zip', snapshot_bytes=final.stat().st_size,
                  snapshot_sha256=sha(final), file_count=len(manifest),
                  previous_artifact_manifest_matches=True, all_archive_entries_sha256_verified=True,
                  source_files_unchanged_during_snapshot=True, environment_versions_unchanged=True,
                  environment_retained_in_place=str(ROOT/'.venv'),
                  environment_binary_copy_in_archive=False, train_metric_top1=0.2,
                  meaning='Engineering baseline only; not an independent or paper reproduction result')
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    (OUT/'record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
    (OUT/'README.md').write_text('''# baseline_v0 封存记录

现有分类基线完整记录：代码、配置、版本清单、原始/当前标注、3037张帧图、日志、逐片预测和两份模型检查点均在 snapshot.zip。
所有条目逐一校验SHA256；清单见 manifest.json，总包校验见 record.json。旧文件在封存时与上轮产物清单一致。

原 `.venv` 环境原地保留，未安装或改动依赖；包版本冻结记录另存 environment-freeze.txt，ZIP含旧环境报告和pyvenv.cfg。ZIP不包含数GB的环境二进制或下载缓存，不能单独当作可迁移Python环境。

恢复时先校验ZIP的SHA256，将内容解压到新的空目录，不覆盖本项目；用Python3.9与requirements-lock.txt重建独立环境。配置中的文件夹模板按新目录展开；旧日志里的绝对路径仅作历史记录。原始外部ZIP的来源路径与哈希记录在归档报告中。

本记录只读使用。后续改动和测试写到audit_stage1或新的版本目录，不重写snapshot.zip。
Top-1 20%仅是当前20片段工程划分上的一轮结果，不是论文复现或可信泛化性能。
''',encoding='utf8')
    print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
