#!/usr/bin/env python3
import argparse, hashlib, json, shutil, subprocess, tempfile, zipfile
from pathlib import Path

SOURCES = {
    "TStruct": {
        "repo": "yangzeli/TStruct",
        "commit": "5b9eb393a092c14d28d08395b63de809accd115a",
        "model": "models/TStruct.py",
    },
    "TimeBase": {
        "repo": "hqh0728/TimeBase",
        "commit": "369b330f3d77371fcc7e8c75c808d01330c40899",
        "model": "models/TimeBase.py",
    },
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def download_source(name: str, spec: dict, root: Path) -> Path:
    dest = root / name
    marker = dest / "SOURCE_COMMIT.txt"
    if marker.exists() and marker.read_text().strip() == spec["commit"] and (dest / spec["model"]).exists():
        print(f"[SOURCE] {name}: pinned source already present ({spec['commit'][:12]})")
        return dest
    if dest.exists():
        raise RuntimeError(f"{dest} exists but is not the expected pinned source. Move it aside manually; refusing to overwrite.")
    root.mkdir(parents=True, exist_ok=True)
    url = f"https://codeload.github.com/{spec['repo']}/zip/{spec['commit']}"
    with tempfile.TemporaryDirectory(prefix=f"{name}_") as td:
        td = Path(td)
        archive = td / f"{name}.zip"
        print(f"[DOWNLOAD] {url}")
        subprocess.run(["curl", "-fL", "--retry", "4", "--retry-delay", "2", url, "-o", str(archive)], check=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(td / "extract")
        candidates = [p for p in (td / "extract").iterdir() if p.is_dir()]
        if len(candidates) != 1 or not (candidates[0] / spec["model"]).exists():
            raise RuntimeError(f"Unexpected archive layout for {name}")
        shutil.move(str(candidates[0]), str(dest))
    marker.write_text(spec["commit"] + "\n")
    print(f"[SOURCE] {name}: downloaded and pinned")
    return dest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, spec in SOURCES.items():
        repo_dir = download_source(name, spec, root)
        license_file = repo_dir / "LICENSE"
        if not license_file.exists():
            raise RuntimeError(f"Missing LICENSE in {name} source archive")
        manifest[name] = {
            "repo": spec["repo"],
            "commit": spec["commit"],
            "license": "MIT",
            "license_sha256": sha256(license_file),
            "model_sha256_before_patch": sha256(repo_dir / spec["model"]),
        }
    (root / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("[SOURCE] both pinned repositories are ready")

if __name__ == "__main__":
    main()
