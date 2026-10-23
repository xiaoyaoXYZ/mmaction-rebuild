"""Resumable HTTP Range download with mandatory publisher SHA256 verification."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(a.url, headers={'Range': 'bytes=0-0'})
    with urllib.request.urlopen(req, timeout=90) as r:
        assert r.status == 206, ('Range unsupported', r.status)
        size = int(r.headers['Content-Range'].split('/')[-1])
    chunks = a.out.parent / (a.out.name + '.parts')
    chunks.mkdir(exist_ok=True)
    chunk_size = 16 * 1024 * 1024
    count = (size + chunk_size - 1) // chunk_size
    started = time.monotonic()
    def fetch(i):
        start = i * chunk_size
        end = min(start + chunk_size, size) - 1
        path = chunks / f'{i:05}.part'
        length = end - start + 1
        if path.exists() and path.stat().st_size == length:
            return i
        for attempt in range(6):
            try:
                req = urllib.request.Request(a.url, headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(req, timeout=120) as response:
                    assert response.status == 206
                    assert response.headers['Content-Range'] == f'bytes {start}-{end}/{size}'
                    data = response.read()
                assert len(data) == length, (len(data), length)
                temp = path.with_suffix('.tmp')
                temp.write_bytes(data)
                temp.replace(path)
                return i
            except Exception as exc:
                print(f'chunk={i} attempt={attempt+1}: {exc}', flush=True)
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 10))
    print(f'Download {size} bytes in {count} chunks; official SHA256={a.sha256}', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for done, _ in enumerate(concurrent.futures.as_completed(ex.submit(fetch, i) for i in range(count)), 1):
            if done % 5 == 0 or done == count:
                print(f'{done}/{count} chunks ({100*done/count:.1f}%), elapsed={time.monotonic()-started:.0f}s', flush=True)
    temp = a.out.with_suffix(a.out.suffix + '.assembling')
    digest = hashlib.sha256()
    with temp.open('wb') as f:
        for i in range(count):
            data = (chunks / f'{i:05}.part').read_bytes()
            digest.update(data)
            f.write(data)
    actual = digest.hexdigest()
    assert actual == a.sha256, (actual, a.sha256)
    temp.replace(a.out)
    record = dict(url=a.url, bytes=size, sha256=actual, status='VERIFIED',
                  elapsed_seconds=time.monotonic()-started)
    a.out.with_suffix('.verified.json').write_text(json.dumps(record, indent=2), encoding='utf8')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
