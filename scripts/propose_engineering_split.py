"""Plan contiguous validation prefixes; do not edit source data or annotations."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    audit = json.loads((ROOT / 'reports/industrial_data_audit.json').read_text(encoding='utf-8'))
    frames = json.loads((ROOT / 'reports/frame_manifest.json').read_text(encoding='utf-8'))
    train = audit['proposed_train']
    val = audit['proposed_val']
    train_hashes = {f['sha256'] for clip in train for f in frames[clip]}
    entries = []
    retained_val_hashes = set()
    for split, clips in (('train', train), ('val', val)):
        for clip in clips:
            ordered = sorted(frames[clip], key=lambda f: f['file'])
            shared_positions = [index + 1 for index, frame in enumerate(ordered)
                                if frame['sha256'] in train_hashes] if split == 'val' else []
            end = min(shared_positions) - 1 if shared_positions else len(ordered)
            if split == 'val':
                retained_val_hashes.update(f['sha256'] for f in ordered[:end])
            entries.append(dict(clip=clip, split=split, source_start=1, source_end=end,
                                original_frames=len(ordered), retained_frames=end,
                                excluded_tail_frames=len(ordered) - end,
                                original_exact_overlap_positions=shared_positions,
                                source_group='unknown',
                                independence='not_established'))
    shared = train_hashes & retained_val_hashes
    assert not shared
    summary = dict(
        purpose='engineering_pipeline_validation_only',
        status='proposal',
        policy='Train clip_1 complete; val clip_2 contiguous prefix before first exact overlap with train.',
        original_shared_exact_hashes=audit['proposed_split_shared_exact_frame_hash_count'],
        retained_shared_exact_hashes=len(shared),
        total_original_frames=sum(x['original_frames'] for x in entries),
        total_retained_frames=sum(x['retained_frames'] for x in entries),
        train_frames=sum(x['retained_frames'] for x in entries if x['split'] == 'train'),
        val_frames=sum(x['retained_frames'] for x in entries if x['split'] == 'val'),
        train_clips=len(train), val_clips=len(val), entries=entries,
        caveat='Exact-file disjointness does not establish source-video/session/person independence. '
               'This split must not be reported as an independent test/generalization result.')
    destination = ROOT / 'reports/engineering_split_proposal.json'
    destination.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'entries'}, indent=2))


if __name__ == '__main__':
    main()
