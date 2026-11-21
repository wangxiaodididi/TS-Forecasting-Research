"""Official upstream acquisition, integrity checks, and immutable experiment snapshots.

Avoid git clone on low-bandwidth/fragile AutoDL mounts: fetch the precise upstream
commit as a release-independent GitHub source archive, verify known upstream Git
blob hashes, and install atomically. Existing invalid source is backed up, never
silently deleted. `vendor/GTR` remains untouched by experiment modifications.
"""
import hashlib
import shutil
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path
from .paths import ROOT, VENDOR, DATA

REPO = 'https://github.com/macovaseas/GTR.git'
UPSTREAM_COMMIT = 'd94161906151bdc0c94f4d21c2b0ca356aeb3135'
UPSTREAM_BLOBS = {
    'run.py': '86495d8031efa9e9bba31b3b440e25cade91ef43',
    'models/GTR.py': '548ad0bff2173fc6858fb32ef3f7af6a1a5789b4',
    'exp/exp_main.py': 'a0c082833dadb714b6ef7cff8e34baa7174838aa',
    'data_provider/data_factory.py': '12fb66f8329bc0570d8084165f0d4633dd77f03f',
}


def git_blob(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def verify(root):
    root = Path(root)
    if not (root / 'models/GTR.py').is_file():
        raise RuntimeError(f'Official models/GTR.py not found in {root}')
    for rel, expected in UPSTREAM_BLOBS.items():
        path = root / rel
        if not path.is_file():
            raise RuntimeError(f'Missing official file: {path}')
        actual = git_blob(path.read_bytes())
        if actual != expected:
            raise RuntimeError(f'Upstream source drift: {rel}: {actual} != {expected}')
    if (root / '.git').exists():
        try:
            commit = subprocess.check_output(
                ['git', '-C', str(root), 'rev-parse', 'HEAD'],
                text=True, stderr=subprocess.PIPE).strip()
        except (subprocess.CalledProcessError, OSError) as exc:
            raise RuntimeError(f'Broken upstream git directory at {root}: {exc}') from exc
        if commit != UPSTREAM_COMMIT:
            raise RuntimeError(f'Expected upstream commit {UPSTREAM_COMMIT}, got {commit}')
    return True


def _backup_invalid_vendor():
    """Preserve a failed checkout; do not destroy user files."""
    if not VENDOR.exists():
        return None
    try:
        verify(VENDOR)
    except (RuntimeError, OSError) as exc:
        stamp = time.strftime('%Y%m%d_%H%M%S')
        backup = VENDOR.with_name('GTR_unverified_backup_' + stamp)
        suffix = 0
        while backup.exists():
            suffix += 1
            backup = VENDOR.with_name(f'GTR_unverified_backup_{stamp}_{suffix}')
        VENDOR.rename(backup)
        print(f'Existing invalid/incomplete GTR preserved at {backup} ({exc})', flush=True)
        return backup
    return None


def _extract_archive(src: Path, dst: Path):
    with zipfile.ZipFile(src) as z:
        entries = z.infolist()
        if not entries:
            raise RuntimeError('Empty upstream ZIP')
        for info in entries:
            parts = Path(info.filename).parts
            if len(parts) < 2 or info.is_dir() and len(parts) == 1:
                continue
            # GitHub's archive contains exactly one root directory.
            if parts[0] != 'GTR-' + UPSTREAM_COMMIT:
                raise RuntimeError(f'Unexpected root in official source archive: {parts[0]}')
            rel = Path(*parts[1:])
            if not rel.parts:
                continue
            if rel.is_absolute() or '..' in rel.parts:
                raise RuntimeError('Unsafe archive entry')
            output = dst / rel
            if info.is_dir():
                output.mkdir(parents=True, exist_ok=True)
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as inp, output.open('wb') as out:
                    shutil.copyfileobj(inp, out)


def _download(url: str, dst: Path):
    request = urllib.request.Request(url, headers={'User-Agent':'GTR-ResearchWorkbench/1.1'})
    with urllib.request.urlopen(request, timeout=50) as response, dst.open('wb') as out:
        while True:
            part = response.read(1024 * 256)
            if not part:
                break
            out.write(part)
    if dst.stat().st_size < 1000:
        raise RuntimeError('Downloaded source archive is suspiciously small')


def install_official():
    """ZIP-first pinned offline/online install; never run git clone implicitly."""
    VENDOR.parent.mkdir(parents=True, exist_ok=True)
    if VENDOR.exists():
        try:
            verify(VENDOR)
            return 'Official GTR source is already installed and verified'
        except (RuntimeError, OSError):
            _backup_invalid_vendor()

    local = [ROOT / 'GTR_official_source.zip', VENDOR.parent / 'GTR_official_source.zip']
    sources = [('local file', p) for p in local if p.is_file()]
    sources += [
        ('GitHub codeload', f'https://codeload.github.com/macovaseas/GTR/zip/{UPSTREAM_COMMIT}'),
        ('GitHub archive', f'https://github.com/macovaseas/GTR/archive/{UPSTREAM_COMMIT}.zip'),
    ]
    errors = []
    for label, source in sources:
        for attempt in range(1 if isinstance(source, Path) else 2):
            staging = VENDOR.parent / '.GTR_verified_staging'
            archive = VENDOR.parent / '.GTR_archive_part.zip'
            if staging.exists():
                shutil.rmtree(staging)
            if archive.exists():
                archive.unlink()
            try:
                print(f'Installing official GTR: {label} (try {attempt+1})', flush=True)
                if isinstance(source, Path):
                    _extract_archive(source, staging)
                else:
                    _download(source, archive)
                    _extract_archive(archive, staging)
                verify(staging)
                staging.rename(VENDOR)
                return f'Installed & verified upstream GTR at pinned commit {UPSTREAM_COMMIT} via {label}'
            except (Exception) as exc:
                errors.append(f'{label} attempt {attempt+1}: {exc}')
                print(f'Source attempt failed: {exc}', flush=True)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
                if archive.exists():
                    archive.unlink()

    raise RuntimeError('Could not obtain the pinned official source. ' + ' | '.join(errors)
        + '\nOffline solution: download the official ZIP on your own PC, upload as '
          'GTR_official_source.zip into project root, then rerun install-official.')


def adopt_official(path):
    """Offline route: import a matching official source checkout."""
    verify(path)
    if VENDOR.exists():
        raise FileExistsError(f'Already exists: {VENDOR}')
    shutil.copytree(path, VENDOR, ignore=shutil.ignore_patterns(
        'dataset', 'checkpoints', 'results', 'test_results', '__pycache__', '.venv'))
    verify(VENDOR)
    return str(VENDOR)


def snapshot(dst, model, mode, smoke_batches=0):
    verify(VENDOR)
    if dst.exists():
        raise FileExistsError(dst)
    shutil.copytree(VENDOR, dst, ignore=shutil.ignore_patterns(
        '.git', 'checkpoints', 'results', 'test_results', 'dataset', '__pycache__', '.venv'))
    (dst / 'dataset').symlink_to(DATA, target_is_directory=True)
    if model != 'GTR':
        from .extend import install_snapshot_extension
        install_snapshot_extension(dst)
    if mode == 'smoke':
        from .extend import enable_smoke_subset
        enable_smoke_subset(dst, max_samples=64)
    return dst
