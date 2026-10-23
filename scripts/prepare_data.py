"""Preserve source frames; prepare an exact-frame-disjoint engineering split.

This is NOT an independent generalization benchmark: source-session grouping is
unknown and the original clip pairs contain overlapping frame sequences.
"""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path(r'E:\Codes\mmaction2-master\data2\ucf101.zip')


def main():
    report = json.loads((ROOT / 'reports/industrial_data_audit.json').read_text(encoding='utf8'))
    frames = json.loads((ROOT / 'reports/frame_manifest.json').read_text(encoding='utf8'))
    for key in ('unsafe_archive_paths', 'decode_or_crc_errors', 'frame_sequence_errors',
                'original_references_missing_from_archive'):
        assert not report[key], (key, report[key])
    digest = hashlib.sha256()
    with ARCHIVE.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == report['archive_sha256'], 'Archive changed since audit'
    train_hashes = {f['sha256'] for clip in report['proposed_train'] for f in frames[clip]}
    active_counts = {x['clip']: x['frames'] for x in report['clips']}
    original_counts = dict(active_counts)
    trimmed = []
    for clip in report['proposed_val']:
        ordered = sorted(frames[clip], key=lambda f: f['file'])
        repeated = [index + 1 for index, frame in enumerate(ordered) if frame['sha256'] in train_hashes]
        if repeated:
            active_counts[clip] = min(repeated) - 1
            assert active_counts[clip] > 0, 'No contiguous prefix remains: ' + clip
            trimmed.append(dict(clip=clip, original_frames=len(ordered),
                                retained_start=1, retained_end=active_counts[clip],
                                excluded_tail_frames=len(ordered) - active_counts[clip],
                                cross_train_exact_overlap_positions=repeated))
    val_hashes = {f['sha256'] for clip in report['proposed_val']
                  for f in sorted(frames[clip], key=lambda f: f['file'])[:active_counts[clip]]}
    assert not (train_hashes & val_hashes), 'Active train/val exact frame overlap'
    target = (ROOT / 'data').resolve()
    target.mkdir(exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        # Validate ALL paths before extracting any member.
        for item in archive.infolist():
            relative = PurePosixPath(item.filename)
            assert not relative.is_absolute() and '..' not in relative.parts
            assert '\\' not in item.filename and ':' not in item.filename
            assert target in (target / str(relative)).resolve().parents
            assert (item.external_attr >> 16) & 0o170000 != 0o120000, 'Archive symlink rejected'
        for item in archive.infolist():
            relative = PurePosixPath(item.filename)
            assert not relative.is_absolute() and '..' not in relative.parts
            assert '\\' not in item.filename and ':' not in item.filename
            dest = (target / str(relative)).resolve()
            assert target in dest.parents, dest
            if item.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    # Idempotent reuse only when bytes match the source archive.
                    assert hashlib.sha256(dest.read_bytes()).digest() == hashlib.sha256(archive.read(item)).digest()
                else:
                    with archive.open(item) as source, dest.open('xb') as output:
                        shutil.copyfileobj(source, output)
    classes = {int(k): v for k, v in report['class_index_one_based'].items()}
    by_name = {v: k - 1 for k, v in classes.items()}
    alias = report['class_folder_alias']
    # Confirm each extracted source JPG retains its audited bytes, including
    # the 91 excluded tail frames. They remain on disk and are never renamed.
    checked_files = 0
    for clip, items in frames.items():
        for item in items:
            path = target / 'ucf101/rawframes' / clip / item['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], path
            checked_files += 1
    assert checked_files == report['frame_count']
    splitdir = target / 'splits'
    splitdir.mkdir(exist_ok=True)
    manifest = []
    for split in ('train', 'val'):
        lines = []
        for clip in report['proposed_' + split]:
            folder = clip.split('/')[0]
            name = alias.get(folder, folder)
            label = by_name[name]
            lines.append(f'{clip} {active_counts[clip]} {label}')
            manifest.append(dict(clip=clip, split=split, frames=active_counts[clip], label=label,
                                 original_frames=original_counts[clip], source_start=1,
                                 source_end=active_counts[clip], class_name=name,
                                 source_group='unknown', independence='not_established'))
        assert len(lines) == 10
        assert {x['label'] for x in manifest if x['split'] == split} == set(range(10))
        (splitdir / (split + '.txt')).write_text('\n'.join(lines) + '\n', encoding='utf8')
    (splitdir / 'classes.json').write_text(json.dumps({str(k-1): v for k, v in classes.items()}, indent=2), encoding='utf8')
    (splitdir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    (splitdir / 'README.md').write_text(
        '# 工业帧序列划分\n\n原始 annotations 原样保留在 data/ucf101/annotations。'
        '这里使用 RawframeDataset 格式：片段目录 帧数 从0开始的类别编号。\n\n'
        '按原类别顺序，每类排序后的第1段用于训练，第2段用于工程验证。每类各1段，总计10/10段。'
        'anya_mopping 对应实际目录 an_mopping，不改原目录。没有独立测试集。\n\n'
        '原始3037帧全部保留。原定10/10划分存在26个跨集完全重复帧哈希，涉及3类。'
        '仅在验证标注中使用首次重复帧之前的连续前缀：an_mopping第2段1–72；'
        'cha_fang_you第2段1–143；cha_fang_zuo第2段1–123。'
        '共排除91个尾部帧，参与采样的帧为2946（训练1545，验证1401）。'
        'MMAction2必须使用start_index=1及这里的total_frames；不要重新扫描目录覆盖帧数。\n\n'
        '未按帧随机拆分或删除中间帧。活动帧SHA-256跨集交集为0，但已知3类原片段含连续复用画面。'
        '同源关系没有因此消失，原视频、采集场次、人员和帧率未知。'
        '此划分仅用于确认工程训练与验证流程，不是独立泛化评测。'
        '类别含左右方向，基线应关闭水平翻转。\n', encoding='utf8')
    summary = {'status': 'PASS', 'purpose': 'engineering_pipeline_validation_only',
               'archive_sha256': digest.hexdigest(), 'train_clips': 10, 'val_clips': 10,
               'classes': len(classes), 'original_frames_preserved_and_hash_verified': checked_files,
               'active_frames': sum(active_counts.values()),
               'train_frames': sum(active_counts[c] for c in report['proposed_train']),
               'val_frames': sum(active_counts[c] for c in report['proposed_val']),
               'excluded_frames': sum(original_counts.values()) - sum(active_counts.values()),
               'original_train_val_shared_exact_hashes': report['proposed_split_shared_exact_frame_hash_count'],
               'active_train_val_shared_exact_hashes': len(train_hashes & val_hashes),
               'trimmed_validation_clips': trimmed,
               'source_independence': 'not_established',
               'annotations': str(splitdir), 'rawframes': str(target / 'ucf101/rawframes')}
    (ROOT / 'reports/data_preparation.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
