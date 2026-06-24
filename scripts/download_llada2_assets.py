#!/usr/bin/env python3
from __future__ import annotations

"""Download LLaDA2 public assets for real-weight validation.

The remote Ascend machines used for validation often reach ModelScope more
reliably than Hugging Face. This helper keeps the runbook reproducible while
leaving the large model files outside the git repository.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_REPO_ID = "inclusionAI/LLaDA2.0-mini-preview"
DEFAULT_INCLUDE = ["*.json", "*.safetensors", "*.py", "README.md", ".gitattributes"]


def parse_args():
    parser = argparse.ArgumentParser(description="Download LLaDA2 assets for validation.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--local-dir", required=True, help="Directory to place the downloaded snapshot.")
    parser.add_argument("--backend", default="auto", choices=["auto", "modelscope", "huggingface"])
    parser.add_argument("--revision", default=None)
    parser.add_argument("--include", nargs="*", default=DEFAULT_INCLUDE)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--proxy", default=None, help="Optional HTTP(S) proxy URL.")
    parser.add_argument("--modelscope-bin", default=None, help="Path to modelscope CLI.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_env(proxy: str | None):
    env = os.environ.copy()
    env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    if proxy:
        env["HTTP_PROXY"] = proxy
        env["HTTPS_PROXY"] = proxy
        env["http_proxy"] = proxy
        env["https_proxy"] = proxy
    return env


def run_modelscope(args, env):
    modelscope_bin = args.modelscope_bin or shutil.which("modelscope")
    if not modelscope_bin and Path("/home/miniconda3/bin/modelscope").exists():
        modelscope_bin = "/home/miniconda3/bin/modelscope"
    if not modelscope_bin and args.dry_run:
        modelscope_bin = "modelscope"
    if not modelscope_bin:
        raise FileNotFoundError("modelscope CLI not found")

    cmd = [
        modelscope_bin,
        "download",
        "--model",
        args.repo_id,
        "--local_dir",
        args.local_dir,
        "--max-workers",
        str(args.max_workers),
    ]
    if args.revision:
        cmd.extend(["--revision", args.revision])
    if args.include:
        cmd.append("--include")
        cmd.extend(args.include)

    print("Running:", " ".join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def run_huggingface(args, env):
    if args.dry_run:
        print(
            "Running: huggingface_hub.snapshot_download("
            f"repo_id={args.repo_id!r}, local_dir={args.local_dir!r})"
        )
        return

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required for --backend huggingface") from exc

    old_env = os.environ.copy()
    os.environ.update(env)
    try:
        snapshot_download(
            repo_id=args.repo_id,
            revision=args.revision,
            local_dir=args.local_dir,
            allow_patterns=args.include,
            max_workers=args.max_workers,
        )
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def main():
    args = parse_args()
    Path(args.local_dir).mkdir(parents=True, exist_ok=True)
    env = build_env(args.proxy)

    if args.backend == "modelscope":
        run_modelscope(args, env)
        return
    if args.backend == "huggingface":
        run_huggingface(args, env)
        return

    try:
        run_modelscope(args, env)
    except Exception as exc:
        print(f"ModelScope backend failed, falling back to Hugging Face: {exc}", file=sys.stderr)
        run_huggingface(args, env)


if __name__ == "__main__":
    main()
