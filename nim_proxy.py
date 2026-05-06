#!/usr/bin/env python3
"""Backward-compatible launcher for the packaged NVIDIA NIM proxy."""

from pathlib import Path
import sys

src_path = Path(__file__).resolve().parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from nvidia_nim_proxy.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
