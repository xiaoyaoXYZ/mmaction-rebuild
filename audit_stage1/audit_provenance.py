"""Read-only, bounded provenance audit; writes only into audit_stage1."""
import collections
import hashlib
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'audit_stage1'
BASE = Path(r'E:\Codes\mmaction2-master')
ACTION = Path(r'E:\Codes\action-analysis_package\action-analysis')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def xlsx_cells(path):
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main', 'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
    with zipfile.ZipFile(path) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(si.itertext()) for si in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        rels = {r.attrib['Id']: r.attrib['Target'] for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        result = []
        for sheet in ET.fromstring(z.read('xl/workbook.xml')).findall('s:sheets/s:sheet', ns):
            target = rels[sheet.attrib['{' + ns['r'] + '}id']]
            target = target.lstrip('/') if target.startswith('/') else 'xl/' + target
            cells = []
            for c in ET.fromstring(z.read(target)).findall('.//s:c', ns):
                v = c.find('s:v', ns)
                if c.attrib.get('t') == 's' and v is not None:
                    value = strings[int(v.text)]
                elif c.attrib.get('t') == 'inlineStr':
                    value = ''.join(c.find('s:is', ns).itertext())
                elif v is not None:
                    value = v.text
                else:
                    continue
                cells.append({'cell': c.attrib['r'], 'value': value})
            result.append({'sheet': sheet.attrib['name'], 'cells': cells})
        return result

manifest = json.loads((ROOT / 'reports/frame_manifest.json').read_text(encoding='utf8'))
split = json.loads((ROOT / 'data/splits/manifest.json').read_text(encoding='utf8'))
archive = BASE / 'data2/ucf101.zip'
inventory = {
    'audit_date_local': '2026-10-22',
    'scope': [str(BASE / 'data2'), str(BASE / 'data'), str(ACTION)],
    'search_limit': 'Only explicitly related directories; no whole-drive search. No training or source modification.',
    'source_archive': {'path': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha(archive)},
    'confirmed_original_video_mappings': [],
    'unknown_for_all_20_clips': ['original_video_id', 'original_video_path', 'source_frame_interval', 'source_fps', 'extraction_stride', 'capture_datetime', 'session_id', 'subject_id', 'camera_id'],
}

with zipfile.ZipFile(archive) as z:
    inventory['source_archive']['extension_counts'] = dict(collections.Counter(PurePosixPath(x.filename).suffix.lower() for x in z.infolist() if not x.is_dir()))
    annotations = {PurePosixPath(n).name: z.read(n).decode('utf-8-sig') for n in z.namelist() if n.startswith('ucf101/annotations/') and n.endswith('.txt')}
inventory['original_annotations'] = annotations
classmap = {int(line.split()[0]): line.split()[1] for line in annotations['classInd.txt'].splitlines() if line.strip()}
original_train = {str(PurePosixPath(line.split()[0]).with_suffix('')): int(line.split()[1]) for line in annotations['trainlist01.txt'].splitlines() if line.strip()}
original_test = {str(PurePosixPath(line.strip()).with_suffix('')) for line in annotations['testlist01.txt'].splitlines() if line.strip()}
alias = {'an_mopping': 'anya_mopping'}
clip_records = []
for row in split:
    clip = row['clip']
    names = [f['file'] for f in manifest[clip]]
    actual_names = sorted(p.name for p in (ROOT / 'data/ucf101/rawframes' / clip).glob('*.jpg'))
    numbers = [int(re.fullmatch(r'img_(\d+)\.jpg', name).group(1)) for name in actual_names]
    mapped_class = alias.get(clip.split('/')[0], clip.split('/')[0])
    expected_label = next(k - 1 for k, v in classmap.items() if v == mapped_class)
    clip_records.append({**row, 'original_split': 'train' if clip in original_train else 'test' if clip in original_test else 'unreferenced',
        'original_explicit_label_one_based': original_train.get(clip),
        'test_label_derivation': 'folder + classInd.txt (explicit alias an_mopping -> anya_mopping)' if clip in original_test else None,
        'current_label_matches_original_class_mapping': row['label'] == expected_label and (clip not in original_train or original_train[clip] - 1 == row['label']),
        'disk_frame_count': len(actual_names), 'names_match_old_frame_manifest': sorted(names) == actual_names,
        'filename_contiguous_from_one': numbers == list(range(1, len(numbers) + 1)),
        'true_temporal_order_verified': False,
        'original_video_id': 'unknown', 'session_id': 'unknown', 'subject_id': 'unknown', 'fps': 'unknown'})
inventory['clips'] = clip_records

inventory['same_named_rawframe_copies'] = []
for rel in ['data2/Aucf101/rawframes', 'data2/1001/rawframes']:
    root = BASE / rel
    checks = []
    for clip, frames in manifest.items():
        expected_names = {f['file'] for f in frames}
        d = root / clip
        names = {p.name for p in d.glob('*.jpg')} if d.is_dir() else set()
        missing, mismatch, matched = [], [], 0
        for frame in frames:
            p = d / frame['file']
            if not p.is_file():
                missing.append(frame['file'])
            elif sha(p) != frame['sha256']:
                mismatch.append(frame['file'])
            else:
                matched += 1
        checks.append({'clip': clip, 'matched_sha256_frames': matched, 'missing': missing,
            'mismatched': mismatch, 'extra_jpg_names': sorted(names - expected_names)})
    inventory['same_named_rawframe_copies'].append({'directory': str(root), 'checks': checks,
        'interpretation': 'A full byte match establishes an identical frame copy, not which copy came first or the original video/session.'})

inventory['associated_annotation_comparisons'] = []
for rel in ['data2/Aucf101/annotations', 'data2/1001/annotations']:
    d = BASE / rel
    texts = {name: (d / name).read_text(encoding='utf-8-sig') for name in annotations if (d / name).is_file()}
    inventory['associated_annotation_comparisons'].append({'directory': str(d),
        'text_matches_archive': {name: texts.get(name) == value for name, value in annotations.items()},
        'classInd_text': texts.get('classInd.txt'),
        'note': 'The 1001 class list is broader and spells class 1 an_mopping; this alone does not identify an original video.'})

videos = sorted((BASE / 'data2/bai-you').glob('*.mp4'))
videos += [ACTION / 'bai(3).mp4', ACTION / 'project/Model_Train/object-detection-model/video/bai(3).mp4',
           ACTION / 'project/Model_Train/object-detection-model/video/bai-zuo(1).mp4', ACTION / 'experiments/test_video/output/29.avi']
inventory['unverified_video_candidates'] = [{'path': str(p), 'bytes': p.stat().st_size,
    'sha256': None, 'sha256_reason': 'Not hashed: no confirmed association with the 20 clips.',
    'relation_to_current_20_clips': 'unknown',
    'candidate_basis': 'Located under related project directory; filename similarity is not proof of origin.'} for p in videos if p.is_file()]
inventory['other_dataset_notice'] = {
    'path': str(BASE / 'data/ucf101'),
    'classInd_first_ten_lines': (BASE / 'data/ucf101/annotations/classInd.txt').read_text(encoding='utf-8-sig').splitlines()[:10],
    'note': 'Separate directory uses public-UCF101-style class names. It must not be merged with the supplied industrial 10-class archive by basename.'}
with zipfile.ZipFile(BASE / 'data/ucf101.zip') as z:
    names = [x.filename for x in z.infolist() if not x.is_dir()]
    inventory['other_dataset_notice']['other_archive'] = {'path': str(BASE / 'data/ucf101.zip'),
        'bytes': (BASE / 'data/ucf101.zip').stat().st_size, 'extension_counts': dict(collections.Counter(PurePosixPath(n).suffix.lower() for n in names)), 'first_ten_members': names[:10]}

scripts = [BASE / 'data2/rename.py', BASE / 'data/label.py',
 ACTION / 'project/Model_Train/object-detection-model/data_processing/generator_train_and_test_data.py',
 ACTION / 'project/Model_Train/object-detection-model/calculate_and_record/calculate_mp4time.py']
inventory['inspected_scripts'] = [{'path': str(p), 'sha256': sha(p), 'text': p.read_text(encoding='utf-8-sig')} for p in scripts]

workbooks = [ACTION / 'doc/视频时间统计.xlsx', ACTION / 'doc/验证阶段/驗證data 20200120.xlsx', ACTION / 'doc/验证阶段/基本單元動作资料20190322.xlsx']
inventory['workbook_checks'] = []
exact_tokens = [PurePosixPath(c).name for c in manifest]
for p in workbooks:
    sheets = xlsx_cells(p)
    exact_matches, related = [], []
    for s in sheets:
        for c in s['cells']:
            value = str(c['value'])
            entry = {'sheet': s['sheet'], **c}
            if any(token.lower() in value.lower() for token in exact_tokens):
                exact_matches.append(entry)
            if re.search(r'\.mp4|\.avi|anya|mopping|bai|cha|帧|幀|拍|來源|来源|視頻|视频|工人', value, re.I):
                related.append(entry)
    inventory['workbook_checks'].append({'path': str(p), 'sheet_sizes': [{'sheet': s['sheet'], 'nonempty_cells': len(s['cells'])} for s in sheets],
        'first_nonempty_cells_by_sheet': [{'sheet': s['sheet'], 'cells': s['cells'][:25]} for s in sheets],
        'exact_current_clip_name_matches': exact_matches, 'related_text_cells': related,
        'relation_to_current_clips': 'unconfirmed; class-name similarity does not establish source mapping'})

(OUT / 'source_inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps({
    'archive': inventory['source_archive'],
    'copies': [{'path': x['directory'], 'matched_frames': sum(c['matched_sha256_frames'] for c in x['checks']), 'missing': sum(len(c['missing']) for c in x['checks']), 'mismatched': sum(len(c['mismatched']) for c in x['checks'])} for x in inventory['same_named_rawframe_copies']],
    'candidate_videos': len(inventory['unverified_video_candidates']),
    'labels_consistent': all(c['current_label_matches_original_class_mapping'] for c in clip_records),
    'filename_sequences_contiguous': all(c['filename_contiguous_from_one'] for c in clip_records),
    'workbooks': [{'path': x['path'], 'sheets': x['sheet_sizes'], 'exact_matches': x['exact_current_clip_name_matches'], 'related_examples': x['related_text_cells'][:12]} for x in inventory['workbook_checks']],
}, ensure_ascii=False, indent=2))
