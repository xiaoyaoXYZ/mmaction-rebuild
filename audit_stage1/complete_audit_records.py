"""Create an evidence template; every unconfirmed source field stays unknown."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
audit = json.loads((ROOT / 'audit_stage1/frame_audit.json').read_text(encoding='utf8'))
splits = json.loads((ROOT / 'data/splits/manifest.json').read_text(encoding='utf8'))
components = {clip: f'content_{i:02}' for i, clips in enumerate(audit['conservative_content_components'], 1) for clip in clips}
rows = []
for item in splits:
    row = {'clip': item['clip'], 'class_id': item['label'], 'existing_engineering_split': item['split'],
           'content_component': components[item['clip']],
           'component_meaning': 'Known exact shared-image connectivity only; NOT a source-video identity or independence proof'}
    row.update({key: 'unknown' for key in ('source_video_id','source_video_path','source_video_sha256',
                'source_start_frame','source_end_frame','source_start_time','source_end_time',
                'source_fps','extraction_stride','recording_session_id','person_id','camera_id',
                'capture_time','extraction_script_or_log','semantic_label_review','evidence_reference')})
    rows.append(row)
assert len(rows) == 20 and len(components) == 20
target = ROOT / 'audit_stage1/provenance_template.jsonl'
if target.exists():
    raise FileExistsError('Never overwrite a provenance template that may have been filled in')
target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf8')
print('PASS: 20 source records created, all unconfirmed fields unknown; local only')
