"""Download Google's MediaPipe Pose Landmarker (lite) model and verify its SHA-256 checksum.

    python scripts/download_model.py            # saves pretrained_models/pose_landmarker_lite.task
    python scripts/download_model.py --check    # only verify an existing file

The model is published by Google under the Apache License 2.0 (see CREDITS.md). It is not stored in
this repository because it is a binary we did not create. The URL points at version 1 of the float16
model, so the file (and therefore the checksum and the landmark cache's model hash) does not change.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import urllib.request
from pathlib import Path

URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
       "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
SHA256 = "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"
DEST = Path(__file__).resolve().parents[1] / "pretrained_models" / "pose_landmarker_lite.task"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=DEST)
    parser.add_argument("--check", action="store_true", help="verify the existing file, do not download")
    args = parser.parse_args()

    if args.dest.is_file():
        digest = sha256(args.dest)
        if digest == SHA256:
            print(f"OK: {args.dest} (sha256 {digest[:12]}…)")
            return 0
        print(f"Checksum mismatch for {args.dest}: {digest}", file=sys.stderr)
        if args.check:
            return 1
    elif args.check:
        print(f"Missing: {args.dest}", file=sys.stderr)
        return 1

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    with tempfile.NamedTemporaryFile(dir=args.dest.parent, delete=False, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urllib.request.urlopen(URL, timeout=60) as response:
                while block := response.read(1 << 16):
                    tmp.write(block)
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            print(f"Download failed: {exc}", file=sys.stderr)
            return 1
    digest = sha256(tmp_path)
    if digest != SHA256:
        tmp_path.unlink(missing_ok=True)
        print(f"Checksum mismatch: got {digest}, expected {SHA256}. The file was not saved.", file=sys.stderr)
        return 1
    tmp_path.replace(args.dest)
    print(f"Saved {args.dest} ({args.dest.stat().st_size / 1e6:.1f} MB, checksum verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
