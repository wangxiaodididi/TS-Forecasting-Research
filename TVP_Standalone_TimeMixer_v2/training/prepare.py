#!/usr/bin/env python3
"""Prepare a separate, checksum-verified copy of official TimeMixer.

Official files in user workspaces are never modified. Standard library only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
WORK = ROOT / 'workspace'
PIN = 'e24610583b36fdd8c76cc17a8df4e65759a5f460'
EXPECTED = {
    'models/TimeMixer.py': 'ad1847df86646848cdfecab568eb49f7099ef1ce',
    'exp/exp_basic.py': '6d093c4c9e2f66bc4ea4907a44274bc23aaec3d9',
    'run.py': 'acf120c652c880351b688ef99a1aa70032f98a74',
}


def git_blob_sha(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()


def validate_official(src):
    for rel, expect in EXPECTED.items():
        p = src / rel
        if not p.is_file():
            raise RuntimeError(f'Official file missing: {p}')
        got = git_blob_sha(p.read_bytes())
        if got != expect:
            raise RuntimeError(f'Official version mismatch: {rel}, got={got}, expected={expect}. Use pinned commit {PIN}.')


def unzip_source(archive, temp):
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if any(n.startswith('/') or '..' in Path(n).parts for n in names):
            raise ValueError('Unsafe archive paths')
        z.extractall(temp)
    candidates = [p for p in temp.iterdir() if p.is_dir() and (p / 'run.py').is_file()]
    if len(candidates) != 1:
        raise RuntimeError('ZIP should have exactly one TimeMixer root directory')
    return candidates[0]


def replace_once(text, old, new, file):
    if text.count(old) != 1:
        raise RuntimeError(f'Patch anchor not unique in {file}: {old!r}')
    return text.replace(old, new, 1)


def patch_repo(dest):
    # Leave TimeMixer architecture completely unchanged.
    # In a disposable copy only: seed is set by runner environment, so both
    # arms can use identical random seed / training protocol.
    run = dest / 'run.py'
    txt = run.read_text()
    txt = replace_once(txt, 'import argparse\n', 'import argparse\nimport os\n', str(run))
    txt = replace_once(txt, 'fix_seed = 2021', "fix_seed = int(os.environ.get('TIMEMIXER_SEED', '2021'))", str(run))
    txt = replace_once(
        txt,
        "            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))\n            exp.test(setting)",
        "            if os.environ.get('TIMEMIXER_ENABLE_TEST', '0') == '1':\n                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))\n                exp.test(setting)",
        str(run))
    run.write_text(txt)
    # Prevent test leakage: the official train() evaluates Test Loss every epoch.
    exp = dest / 'exp' / 'exp_long_term_forecasting.py'
    txt = exp.read_text()
    txt = replace_once(
        txt,
        "            test_loss = self.vali(test_data, test_loader, criterion)",
        "            test_loss = float('nan')  # intentionally hidden during development",
        str(exp))
    txt = replace_once(
        txt,
        "self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))",
        "self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth'), map_location=self.device))",
        str(exp))
    exp.write_text(txt)


def main():
    p = argparse.ArgumentParser(description='Prepare isolated official TimeMixer comparison worktree')
    p.add_argument('--source', help='Directory containing unmodified official repo at the pinned commit')
    p.add_argument('--archive', help='ZIP of pinned official source, if GitHub is blocked')
    p.add_argument('--force', action='store_true', help='Recreate experiment worktree; old logs remain in separate root')
    a = p.parse_args()
    dst = WORK / 'official'
    if dst.exists() and not a.force:
        print(f'[READY] {dst}; already prepared. Add --force to replace disposable source copy.')
        return
    if a.source and a.archive:
        p.error('Use either --source or --archive')
    with tempfile.TemporaryDirectory(prefix='tmofficial_') as tmp:
        td = Path(tmp)
        if a.source:
            src = Path(a.source).expanduser().resolve()
        else:
            archive = Path(a.archive).resolve() if a.archive else td / 'official.zip'
            if not a.archive:
                url = f'https://codeload.github.com/kwuking/TimeMixer/zip/{PIN}'
                print('[DOWNLOAD]', url, flush=True)
                try:
                    urllib.request.urlretrieve(url, archive)
                except Exception as e:
                    raise RuntimeError('Download failed. Download the pinned ZIP on your PC and upload it, then use --archive, or use --source.') from e
            extract = td / 'extract'
            extract.mkdir(parents=True, exist_ok=True)
            src = unzip_source(archive, extract)
        validate_official(src)
        WORK.mkdir(parents=True, exist_ok=True)
        fresh = WORK / 'official_new'
        if fresh.exists():
            shutil.rmtree(fresh)
        shutil.copytree(src, fresh, ignore=shutil.ignore_patterns('.git', '__pycache__', 'dataset', 'checkpoints', 'results', 'test_results'))
        patch_repo(fresh)
        if dst.exists():
            shutil.rmtree(dst)
        fresh.rename(dst)
        data = {'official_repo':'https://github.com/kwuking/TimeMixer','official_commit':PIN,'verified_original_blobs':EXPECTED,'changes':['original TimeMixer model unchanged','seed environment override','disable test eval in training and after training by default']}
        (WORK / 'manifest.json').write_text(json.dumps(data, ensure_ascii=False, indent=2))
        print('[READY]', dst)

if __name__ == '__main__':
    main()
