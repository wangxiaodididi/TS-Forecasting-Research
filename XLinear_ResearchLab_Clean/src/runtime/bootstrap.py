"""官方源码、公开数据集准备：校验固定 Git blob，不静默替换现有仓库。"""
from __future__ import annotations
import argparse
import hashlib
import io
import shutil
from pathlib import Path
import tempfile
import urllib.request
import zipfile
from src.core.config import ROOT, read_config

UPSTREAM = ROOT / 'vendor' / 'XLinear'
# 固定版本上游训练器、数据处理器的 Git blob SHA（不是用户改造文件的校验和）。
REQUIRED_BLOBS = {
    'run_longExp.py': 'a062ce361fa3b95c6aea4b39b1a5b6b1812c5f38',
    'exp/exp_main.py': '1bd11747069462b346944044f2c329c906222613',
    'data_provider/data_loader.py': '6f14b2e3d982b9ab4425b8e53e9ff2b5345d4ccf',
    'data_provider/data_factory.py': '9a60c1fdabc5fe550e2a2681928c4c90a080a181',
    'utils/tools.py': '39e0ca6e90e23e4ddf6f2e706f3c4f89d1cf8f49',
}


def git_blob(payload: bytes) -> str:
    return hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()


def verify_source(cfg: dict, upstream: Path = UPSTREAM) -> None:
    source=upstream/'models'/'XLinear.py'
    if not source.is_file(): raise FileNotFoundError(f'找不到原版模型: {source}')
    actual=git_blob(source.read_bytes())
    expected=cfg['source']['model_git_blob']
    if actual != expected:
        raise RuntimeError(f'官方 XLinear.py 校验失败: expected={expected}; actual={actual}. 不覆盖文件。')
    for relative,sha in REQUIRED_BLOBS.items():
        file=upstream/relative
        if not file.is_file() or git_blob(file.read_bytes())!=sha:
            raise RuntimeError(f'作者固定版本重要文件校验失败: {relative}；不要使用当前源码作为官方基线。')


def _extract_safe(raw: bytes, target: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        members=[m for m in z.infolist() if not m.is_dir()]
        prefixes={Path(m.filename).parts[0] for m in members}
        if len(prefixes)!=1: raise RuntimeError('下载压缩包根目录数量不正确')
        for item in members:
            parts=Path(item.filename).parts[1:]
            if not parts or '..' in parts: raise RuntimeError('下载 ZIP 路径不安全')
            path=target.joinpath(*parts)
            path.parent.mkdir(parents=True,exist_ok=True)
            with z.open(item) as a,path.open('wb') as b: shutil.copyfileobj(a,b)


def ensure_source(cfg:dict, reuse:Path|None=None, allow_download:bool=True):
    if UPSTREAM.exists():
        verify_source(cfg)
        print('已校验官方源码:',cfg['source']['commit'],flush=True)
        return
    UPSTREAM.parent.mkdir(parents=True,exist_ok=True)
    if reuse:
        old=Path(reuse).expanduser().resolve()
        if not old.is_dir(): raise FileNotFoundError(f'指定的旧官方源码不存在: {old}')
        verify_source(cfg,old)
        print('从旧备份复制已校验的官方源码:',old,flush=True)
        shutil.copytree(old,UPSTREAM,ignore=shutil.ignore_patterns('__pycache__','*.pyc','results','test_results','checkpoints','result.txt'))
    elif allow_download:
        url=f"https://codeload.github.com/Zaiwen/XLinear/zip/{cfg['source']['commit']}"
        print('首次下载官方固定版本:',url,flush=True)
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'XLinear-ResearchLab/1.0'}),timeout=120) as reply:
            raw=reply.read()
        with tempfile.TemporaryDirectory(prefix='xlinear_source_',dir=UPSTREAM.parent) as folder:
            temp=Path(folder)/'source';temp.mkdir()
            _extract_safe(raw,temp)
            verify_source(cfg,temp)
            temp.rename(UPSTREAM)
    else:
        raise FileNotFoundError('尚未准备官方源码。请执行 bash run.sh setup 或复制旧版 upstream_XLinear。')
    verify_source(cfg)
    print('官方源码就绪并通过校验',flush=True)


def prepare_datasets(cfg:dict) -> dict:
    src=ROOT/cfg['source']['archive']
    if not src.is_file(): raise FileNotFoundError(f'公开数据归档不存在: {src}')
    mappings={
        'ETTh1.csv': 'dataset/ETT-small/ETTh1.csv',
        'ETTm1.csv': 'dataset/ETT-small/ETTm1.csv',
        'weather.csv': 'dataset/weather/weather.csv',
        'electricity.csv': 'dataset/electricity/electricity.csv',
    }
    dest=UPSTREAM/'dataset';dest.mkdir(parents=True,exist_ok=True)
    checks={}
    with zipfile.ZipFile(src) as archive:
        for filename,member in mappings.items():
            data=archive.read(member)
            target=dest/filename
            expected=hashlib.sha256(data).hexdigest()
            if target.exists():
                actual=hashlib.sha256(target.read_bytes()).hexdigest()
                if actual!=expected:raise RuntimeError(f'数据文件内容不匹配，拒绝覆盖: {target}')
            else:
                target.write_bytes(data)
            checks[filename]=expected
    print('数据准备完成:',', '.join(checks),flush=True)
    return checks


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reuse-upstream',type=Path,help='从旧项目复制已校验的官方源码，而不是联网下载')
    parser.add_argument('--no-download',action='store_true')
    args=parser.parse_args()
    cfg=read_config()
    ensure_source(cfg,reuse=args.reuse_upstream,allow_download=not args.no_download)
    prepare_datasets(cfg)
    print('SETUP PASS：官方模型与公开数据已就绪。')


if __name__=='__main__':main()
