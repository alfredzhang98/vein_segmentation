"""Download original public video datasets, resume transfers and verify checksums.

From the repository root:
python -m data.pipeline.download_video_datasets --datasets Regional-US --dry-run
Original archives and source metadata stay under each ignored dataset directory.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import shutil
from pathlib import Path
import threading
import time
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[2] / 'data/datasets'
RECORDS = {'ThrombUS': [17659415, 17664207],
           'TUS-REC2024': [11178509, 11180795, 11355500, 12979481]}
LOCK = threading.Lock()


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def status(root, name, **fields):
    with LOCK:
        path = root / 'download_status.json'
        data = json.loads(path.read_text()) if path.exists() else {}
        data.setdefault(name, {}).update(fields, updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        write_json(path, data)


def checksum(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def ranged_transfer(root, entry, path, part):
    """Bounded parallel ranges for large files; persist completed chunks on retry."""
    chunks = path.with_suffix(path.suffix+'.chunks'); chunks.mkdir(exist_ok=True)
    block = 64 * 1024**2; total = entry['size']
    count = (total + block - 1) // block
    if part.exists():
        # Reuse only complete blocks from a previous sequential transfer.
        with part.open('rb') as f:
            for i in range(part.stat().st_size // block):
                p = chunks/f'{i:06d}'
                if p.exists():
                    f.seek(block, 1)
                else:
                    p.write_bytes(f.read(block))
        part.unlink()
    done = sum(p.stat().st_size for p in chunks.iterdir() if p.name.isdigit())
    progress_lock = threading.Lock()

    def fetch(i):
        nonlocal done
        start = i*block; end = min(total, start+block)-1
        dest = chunks/f'{i:06d}'; tmp = dest.with_suffix('.tmp')
        if dest.exists() and dest.stat().st_size == end-start+1:
            return
        for attempt in range(8):
            try:
                with requests.get(entry['url'], headers={'Range':f'bytes={start}-{end}'},
                                  stream=True, timeout=(30, 90)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'):
                        raise ValueError('Server did not honor exact download range')
                    with tmp.open('wb') as f:
                        for chunk in response.iter_content(4*1024**2):
                            f.write(chunk)
                if tmp.stat().st_size != end-start+1:
                    raise ValueError('Incomplete range')
                tmp.replace(dest)
                with progress_lock:
                    done += dest.stat().st_size
                    status(root, entry['name'], state='downloading', bytes=done, total=total)
                    print(root.name, entry['name'], f'{done/1e9:.2f}/{total/1e9:.2f} GB', flush=True)
                return
            except (requests.RequestException, ValueError):
                if attempt == 7:
                    raise
                time.sleep(min(5*(attempt+1), 30))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, range(count)))
    status(root, entry['name'], state='assembling', bytes=total, total=total)
    with part.open('wb') as out:
        for i in range(count):
            with (chunks/f'{i:06d}').open('rb') as f:
                shutil.copyfileobj(f, out, length=8*1024**2)
    part.replace(path)


def transfer(root, entry):
    path = root / 'archives' / entry['name']
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + '.part')
    expected = entry.get('checksum')
    if not path.exists() and entry.get('size', 0) > 1024**3:
        ranged_transfer(root, entry, path, part)
    if not path.exists():
        for attempt in range(8):
            offset = part.stat().st_size if part.exists() else 0
            try:
                headers = {'Range': f'bytes={offset}-'} if offset else {}
                with requests.get(entry['url'], headers=headers, stream=True, timeout=(30, 90)) as response:
                    response.raise_for_status()
                    if offset and response.status_code == 206:
                        if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                            raise ValueError('Invalid resume range')
                    else:
                        offset = 0
                    total = entry.get('size') or (offset + int(response.headers.get('Content-Length', 0)))
                    progress_time = 0
                    with part.open('ab' if offset else 'wb') as f:
                        for chunk in response.iter_content(4 * 1024**2):
                            if not chunk:
                                continue
                            f.write(chunk); offset += len(chunk)
                            if time.monotonic() - progress_time > 15:
                                status(root, entry['name'], state='downloading', bytes=offset, total=total)
                                print(root.name, entry['name'], f'{offset/1e9:.2f}/{total/1e9:.2f} GB', flush=True)
                                progress_time = time.monotonic()
                if entry.get('size') and part.stat().st_size != entry['size']:
                    raise ValueError('Incomplete download')
                part.replace(path)
                break
            except (requests.RequestException, ValueError) as e:
                status(root, entry['name'], state='retrying', error=str(e), attempt=attempt+1)
                if attempt == 7:
                    raise
                time.sleep(min(5 * (attempt+1), 30))
    status(root, entry['name'], state='verifying', bytes=path.stat().st_size)
    if expected:
        alg, wanted = expected.split(':', 1)
        actual = checksum(path, alg)
        if actual != wanted:
            status(root, entry['name'], state='checksum_failed', actual=actual)
            raise ValueError(f'Checksum mismatch: {path}')
    else:
        actual = checksum(path, 'sha256')
    status(root, entry['name'], state='verified', checksum=expected or 'sha256:'+actual,
           bytes=path.stat().st_size)
    chunks = path.with_suffix(path.suffix+'.chunks')
    if chunks.exists():
        shutil.rmtree(chunks)  # verified archive now contains all downloaded bytes
    if path.suffix == '.zip':
        target = root / 'raw' / path.stem
        marker = target / '.extracted.json'
        if not marker.exists():
            target.mkdir(parents=True, exist_ok=True)
            status(root, entry['name'], state='extracting')
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    dest = (target / info.filename).resolve()
                    if not dest.is_relative_to(target.resolve()) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError('Unsafe archive path')
                    archive.extract(info, target)
            write_json(marker, {'archive': path.name, 'checksum': expected or actual})
    status(root, entry['name'], state='complete')
    print('COMPLETE', root.name, path.name, flush=True)


def inventory(name, *, write=False):
    root = ROOT / name
    entries = []; sources = []
    if name == 'Regional-US':
        # Pin the inspected release so new clones get the documented 227 videos.
        commit = 'ac4bf778db01cc81338773485e59a8ac254f665c'
        sources.append({'repository': 'https://github.com/Regional-US/brachial_plexus', 'commit': commit})
        entries.append({'name': f'brachial_plexus-{commit}.zip',
                        'url': f'https://codeload.github.com/Regional-US/brachial_plexus/zip/{commit}',
                        'size': 652244758,
                        'checksum': 'sha256:093cab36f70e0ee9be0942c0b8d0b3558ff688852a8cb1f88159f429ea6feea6'})
    else:
        for record in RECORDS[name]:
            response = requests.get(f'https://zenodo.org/api/records/{record}', timeout=60)
            response.raise_for_status(); data = response.json(); sources.append(data)
            for f in data['files']:
                entry = {'name': f['key'], 'url': f['links']['self'],
                         'size': f['size'], 'checksum': f['checksum']}
                # The two TUS training records repeat the identical calibration file.
                if entry['name'] not in [e['name'] for e in entries]:
                    entries.append(entry)
    if write:
        root.mkdir(parents=True, exist_ok=True)
        write_json(root/'download_sources.json', {'dataset': name, 'sources': sources, 'files': entries})
    return [(root, entry) for entry in entries]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', required=True,
                        choices=['ThrombUS', 'Regional-US', 'TUS-REC2024'])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true', help='List public URLs and sizes without downloading or creating directories')
    parser.add_argument('--files', nargs='+', help='Download only these exact archive filenames (for example Freehand_US_data_val.zip)')
    args = parser.parse_args(); jobs = []
    if args.workers < 1:
        parser.error('--workers must be positive')
    for name in args.datasets:
        jobs.extend(inventory(name, write=False))
    if args.files:
        missing = set(args.files) - {entry['name'] for _, entry in jobs}
        if missing:
            parser.error('Unknown filenames: ' + ', '.join(sorted(missing)))
        jobs = [(root, entry) for root, entry in jobs if entry['name'] in args.files]
    for root, entry in jobs:
        print(f"{root.name}\t{entry['name']}\t{entry['size']/1e9:.3f} GB\t{entry['url']}", flush=True)
    print(f"Download total: {sum(entry['size'] for _, entry in jobs)/1e9:.3f} GB. Extraction and caches need additional space.", flush=True)
    if args.dry_run:
        return
    for name in args.datasets:
        root = ROOT/name; root.mkdir(parents=True, exist_ok=True)
        write_json(root/'download_sources.json', {'dataset': name,
                   'files': [entry for selected_root, entry in jobs if selected_root == root]})
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(transfer, root, entry):(root, entry) for root, entry in jobs}
        for result in as_completed(pending):
            root, entry = pending[result]
            try:
                result.result()
            except Exception as e:
                status(root, entry['name'], state='failed', error=str(e))
                errors.append(str(e)); print('FAILED', root.name, entry['name'], str(e), flush=True)
    if errors:
        raise RuntimeError('; '.join(errors))


if __name__ == '__main__':
    main()
