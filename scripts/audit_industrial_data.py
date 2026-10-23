"""Read-only archive audit. Original data and annotations are never modified."""
import argparse
import collections
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True)
    parser.add_argument('--report-dir', required=True)
    args = parser.parse_args()
    archive = Path(args.archive)
    out = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    clips = collections.defaultdict(list)
    hashes = collections.defaultdict(list)
    sizes = collections.Counter()
    modes = collections.Counter()
    errors = []
    samples = []
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        unsafe = [m.filename for m in members if PurePosixPath(m.filename).is_absolute() or '..' in PurePosixPath(m.filename).parts or '\\' in m.filename]
        annotation_text = {m.filename: z.read(m).decode('utf-8-sig') for m in members if m.filename.endswith('.txt')}
        classes = {}
        for line in annotation_text['ucf101/annotations/classInd.txt'].splitlines():
            if line.strip():
                number, name = line.split()
                classes[int(number)] = name
        original_train = []
        for line in annotation_text['ucf101/annotations/trainlist01.txt'].splitlines():
            if line.strip():
                name, label = line.split()
                original_train.append({'clip': str(PurePosixPath(name).with_suffix('')), 'label_one_based': int(label)})
        original_test = [str(PurePosixPath(line.strip()).with_suffix('')) for line in annotation_text['ucf101/annotations/testlist01.txt'].splitlines() if line.strip()]
        for m in members:
            if not m.filename.lower().endswith('.jpg'):
                continue
            relative = PurePosixPath(m.filename).relative_to('ucf101/rawframes')
            clip = str(relative.parent)
            try:
                data = z.read(m)  # validates CRC while reading
                with Image.open(io.BytesIO(data)) as im:
                    im.load()  # fully decode every image
                    sizes[f'{im.width}x{im.height}'] += 1
                    modes[im.mode] += 1
                frame_hash = hashlib.sha256(data).hexdigest()
                hashes[frame_hash].append(str(relative))
                clips[clip].append({'file': relative.name, 'sha256': frame_hash, 'bytes': len(data)})
                if relative.name == 'img_00001.jpg':
                    sample_file = out / 'samples' / (clip.replace('/', '__') + '.jpg')
                    sample_file.parent.mkdir(parents=True, exist_ok=True)
                    sample_file.write_bytes(data)
                    samples.append(str(sample_file))
            except Exception as exc:
                errors.append({'path': m.filename, 'error': str(exc)})
        member_count = len(members)
        uncompressed_bytes = sum(m.file_size for m in members)
        file_extensions = dict(collections.Counter(PurePosixPath(m.filename).suffix.lower() for m in members if not m.is_dir()))
    clip_summary = []
    sequence_errors = []
    for clip, frames in sorted(clips.items()):
        numbers = sorted(int(re.fullmatch(r'img_(\d+)\.jpg', f['file']).group(1)) for f in frames)
        contiguous = numbers == list(range(1, len(frames) + 1))
        if not contiguous:
            sequence_errors.append(clip)
        clip_summary.append({'clip': clip, 'frames': len(frames), 'contiguous_from_one': contiguous})
    clip_to_label = {x['clip']: x['label_one_based'] for x in original_train}
    actual_folders = {clip.split('/')[0] for clip in clips}
    class_to_label = {name: label for label, name in classes.items()}
    alias_map = {'an_mopping': 'anya_mopping'}
    for clip in clips:
        folder = clip.split('/')[0]
        clip_to_label[clip] = class_to_label[alias_map.get(folder, folder)]
    train_names = {x['clip'] for x in original_train}
    test_names = set(original_test)
    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    cross_clip_duplicates = [paths for paths in duplicate_groups if len({str(PurePosixPath(p).parent) for p in paths}) > 1]
    proposed_train = []
    proposed_val = []
    for label in sorted(classes):
        class_clips = sorted(clip for clip, value in clip_to_label.items() if value == label)
        if len(class_clips) != 2:
            raise ValueError(f'Expected two clips for class {label}, got {class_clips}')
        proposed_train.append(class_clips[0])
        proposed_val.append(class_clips[1])
    train_frame_hashes = {f['sha256'] for clip in proposed_train for f in clips[clip]}
    val_frame_hashes = {f['sha256'] for clip in proposed_val for f in clips[clip]}
    report = {
        'archive': str(archive), 'archive_bytes': archive.stat().st_size,
        'archive_sha256': digest.hexdigest(), 'member_count': member_count,
        'uncompressed_bytes': uncompressed_bytes, 'file_extensions': file_extensions,
        'unsafe_archive_paths': unsafe, 'class_index_one_based': classes,
        'class_folder_alias': alias_map,
        'actual_class_folders': sorted(actual_folders), 'clips': clip_summary,
        'frame_count': sum(len(fs) for fs in clips.values()), 'image_sizes': dict(sizes),
        'image_modes': dict(modes), 'decode_or_crc_errors': errors,
        'frame_sequence_errors': sequence_errors,
        'original_annotations': annotation_text,
        'original_train_count': len(train_names), 'original_test_count': len(test_names),
        'original_train_class_counts': dict(collections.Counter(clip_to_label[c] for c in train_names)),
        'original_test_class_counts': dict(collections.Counter(clip_to_label[c] for c in test_names)),
        'original_clip_overlap': sorted(train_names & test_names),
        'original_references_missing_from_archive': sorted((train_names | test_names) - set(clips)),
        'unreferenced_archive_clips': sorted(set(clips) - (train_names | test_names)),
        'exact_duplicate_frame_groups': duplicate_groups,
        'exact_duplicate_frames_across_clips': cross_clip_duplicates,
        'proposed_train': proposed_train, 'proposed_val': proposed_val,
        'proposed_split_shared_exact_frame_hash_count': len(train_frame_hashes & val_frame_hashes),
        'limitations': [
            'This archive contains RGB frame sequences, not original videos, and is not the public UCF101 dataset.',
            'Original split excludes classes 9 and 10 from training; a new balanced split is needed for pipeline validation.',
            'Each class has only two clips. Source video, subject, camera, production session, timestamps and FPS are unknown.',
            'Clip-disjoint split and exact frame hashes cannot prove independence at source-video or acquisition-session level.',
            'A one-epoch run is a pipeline validation baseline, not evidence of deployment or generalization accuracy.',
        ],
        'sample_files': samples,
    }
    (out / 'industrial_data_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'frame_manifest.json').write_text(json.dumps(clips, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('original_annotations', 'sample_files', 'exact_duplicate_frame_groups')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
