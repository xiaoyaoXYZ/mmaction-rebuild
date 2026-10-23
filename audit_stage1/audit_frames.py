"""Read-only full-frame audit; perceptual matches are candidates, not source IDs."""
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'audit_stage1'
RAW = ROOT / 'data/ucf101/rawframes'

def sha(data): return hashlib.sha256(data).hexdigest()

def runs(pairs):
    offsets=defaultdict(list)
    for a,b in pairs: offsets[b-a].append(a)
    result=[]
    for offset, positions in offsets.items():
        positions=sorted(set(positions))
        first=last=positions[0]
        for p in positions[1:]+[None]:
            if p is not None and p==last+1:
                last=p
            else:
                result.append(dict(a_start=first,a_end=last,b_start=first+offset,
                                   b_end=last+offset,length=last-first+1,offset=offset))
                first=last=p
    return sorted(result,key=lambda r:(-r['length'],r['a_start'],r['b_start']))

def main():
    started=time.monotonic()
    cv2.setNumThreads(1)
    previous=json.loads((ROOT/'reports/frame_manifest.json').read_text(encoding='utf8'))
    selected=json.loads((ROOT/'data/splits/manifest.json').read_text(encoding='utf8'))
    roles={x['clip']:x for x in selected}
    files=sorted(RAW.rglob('*.jpg'))
    def read(p):
        content=p.read_bytes()
        img=cv2.imdecode(np.frombuffer(content,np.uint8),cv2.IMREAD_COLOR)
        if img is None: raise ValueError(str(p))
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        tiny=cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)
        coeff=cv2.dct(tiny)[:8,:8].ravel()[1:]
        bits=coeff>np.median(coeff)
        phash=sum(int(v)<<i for i,v in enumerate(bits))
        thumb=cv2.resize(img,(64,36),interpolation=cv2.INTER_AREA)
        relative=p.relative_to(RAW).as_posix()
        clip=p.parent.relative_to(RAW).as_posix()
        frame=int(re.fullmatch(r'img_(\d+)\.jpg',p.name).group(1))
        return dict(path=relative,clip=clip,frame=frame,bytes=len(content),
                    file_sha256=sha(content),pixel_sha256=sha(img.tobytes()),
                    width=img.shape[1],height=img.shape[0],phash=phash,
                    selected=frame<=roles[clip]['frames'],split=roles[clip]['split']),thumb
    records=[]; thumbnails=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n,(record,thumbnail) in enumerate(pool.map(read,files),1):
            records.append(record); thumbnails.append(thumbnail)
            if n%500==0: print(f'decoded {n}/{len(files)}',flush=True)
    assert len(records)==3037
    old={f'{c}/{f["file"]}':f['sha256'] for c,fs in previous.items() for f in fs}
    changed=[r['path'] for r in records if old.get(r['path'])!=r['file_sha256']]
    assert not changed,changed
    byclip=defaultdict(list); byfile=defaultdict(list); bypixel=defaultdict(list)
    for i,r in enumerate(records):
        byclip[r['clip']].append(i); byfile[r['file_sha256']].append(i); bypixel[r['pixel_sha256']].append(i)
    duplicate_groups=[[records[i]['path'] for i in ids] for ids in byfile.values() if len(ids)>1]
    pixel_groups=[[records[i]['path'] for i in ids] for ids in bypixel.values() if len(ids)>1]
    clips=sorted(byclip)
    parent={c:c for c in clips}
    def find(c):
        while parent[c]!=c: c=parent[c]
        return c
    exact_pairs=[]
    for ia,a in enumerate(clips):
        for b in clips[ia+1:]:
            pairs=[(records[i]['frame'],records[j]['frame']) for i in byclip[a] for j in byclip[b]
                   if records[i]['pixel_sha256']==records[j]['pixel_sha256']]
            if pairs:
                parent[find(b)]=find(a)
                exact_pairs.append(dict(clip_a=a,clip_b=b,frame_pairs=pairs,runs=runs(pairs),
                    currently_selected_pairs=[(x,y) for x,y in pairs if x<=roles[a]['frames'] and y<=roles[b]['frames']]))
    thumbs=np.stack(thumbnails).astype(np.int16)
    phashes=np.array([r['phash'] for r in records],dtype=np.uint64)
    lut=np.array([int(i).bit_count() if hasattr(int,'bit_count') else bin(i).count('1') for i in range(256)],dtype=np.uint8)
    near=[]
    with (OUT/'near_frame_candidates.jsonl').open('w',encoding='utf8') as log:
        for ia,a in enumerate(clips):
            for b in clips[ia+1:]:
                ids=np.array(byclip[b],dtype=np.int64)
                matches=[]; active=[]; pixel_different=0
                for i in byclip[a]:
                    xor=np.bitwise_xor(phashes[i],phashes[ids])
                    dist=lut[xor.view(np.uint8).reshape(-1,8)].sum(axis=1)
                    candidate=ids[dist<=6]
                    if len(candidate)==0: continue
                    diffs=np.abs(thumbs[candidate]-thumbs[i])
                    maes=diffs.mean(axis=(1,2,3))/255.0
                    for j,mae in zip(candidate[maes<=0.01],maes[maes<=0.01]):
                        ra,rb=records[i],records[j]
                        pair=(ra['frame'],rb['frame'])
                        matches.append(pair)
                        different=ra['pixel_sha256']!=rb['pixel_sha256']
                        pixel_different+=int(different)
                        current=ra['selected'] and rb['selected']
                        if current: active.append(pair)
                        log.write(json.dumps(dict(a=ra['path'],b=rb['path'],thumbnail_mae=float(mae),
                            pixel_equal=not different,both_currently_selected=current))+'\n')
                if matches:
                    nr=runs(matches); ar=runs(active) if active else []
                    near.append(dict(clip_a=a,clip_b=b,cross_current_split=roles[a]['split']!=roles[b]['split'],
                        candidate_pairs=len(matches),pixel_different_candidate_pairs=pixel_different,
                        active_candidate_pairs=len(active),longest_run=nr[0],
                        longest_active_run=ar[0] if ar else None,
                        long_runs=[r for r in nr if r['length']>=8][:12],
                        long_active_runs=[r for r in ar if r['length']>=8][:12]))
            print(f'compared clip {ia+1}/{len(clips)}',flush=True)
    components=defaultdict(list)
    for c in clips: components[find(c)].append(c)
    clip_rows=[]; cuts=[]
    for c in clips:
        ids=byclip[c]
        seq=[records[i]['frame'] for i in ids]
        assert seq==list(range(1,len(ids)+1)),c
        differences=np.abs(thumbs[np.array(ids[1:])]-thumbs[np.array(ids[:-1])]).mean(axis=(1,2,3))/255.0
        transitions=[dict(before=k+1,after=k+2,thumbnail_mae=float(v)) for k,v in enumerate(differences) if v>0.12]
        if transitions: cuts.append(dict(clip=c,candidate_transitions=transitions))
        duplicate_extra=len(ids)-len({records[i]['file_sha256'] for i in ids})
        clip_rows.append(dict(clip=c,frames=len(ids),split=roles[c]['split'],active_frames=roles[c]['frames'],
                              label=roles[c]['label'],class_name=roles[c]['class_name'],
                              numbering_contiguous=True,extra_byte_duplicate_occurrences=duplicate_extra,
                              source_video_id='unknown',capture_session='unknown'))
    trainhash={r['pixel_sha256'] for r in records if r['selected'] and r['split']=='train'}
    valhash={r['pixel_sha256'] for r in records if r['selected'] and r['split']=='val'}
    result=dict(status='PASS',frames=len(records),all_current_file_sha256_match_previous_manifest=True,
        image_sizes=dict(Counter(f"{r['width']}x{r['height']}" for r in records)),clips=clip_rows,
        byte_duplicate_groups=duplicate_groups,decoded_pixel_duplicate_groups=pixel_groups,
        exact_cross_clip_pairs=exact_pairs,conservative_content_components=list(components.values()),
        current_selected_cross_split_decoded_pixel_hash_count=len(trainhash&valhash),
        perceptual_method=dict(phash='63 AC DCT bits from gray32x32',hamming_threshold=6,
            color_thumbnail='64x36 BGR INTER_AREA',normalized_thumbnail_mae_threshold=0.01,
            long_run_minimum_frames=8,interpretation='Heuristic candidates only; not proof of same source or independent negatives'),
        near_duplicate_clip_pairs=near,candidate_scene_transitions=cuts,
        warnings=['Pixel/byte equality confirms shared image content, not the identity of the original recording.',
                  'Low-resolution perceptual matches can be caused by static cameras or repeated actions.',
                  'No heuristic matches does not establish source independence.',
                  'Contiguous file numbers do not prove continuous acquisition or absence of edits.'],
        elapsed_seconds=time.monotonic()-started)
    (OUT/'frame_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    (OUT/'frame_fingerprints.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:result[k] for k in ['status','frames','current_selected_cross_split_decoded_pixel_hash_count','elapsed_seconds']}),flush=True)

if __name__=='__main__':main()
