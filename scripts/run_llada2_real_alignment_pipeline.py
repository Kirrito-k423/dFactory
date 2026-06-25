#!/usr/bin/env python3
from __future__ import annotations

"""Gate and run the real-weight LLaDA2 old/new alignment workflow."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full real-weight LLaDA2 alignment pipeline.")
    parser.add_argument("--legacy-repo", required=True)
    parser.add_argument("--current-repo", default=str(REPO_ROOT))
    parser.add_argument("--config-path", default=str(REPO_ROOT / "configs/model_configs/llada2_mini"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sample-path", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    parser.add_argument("--device", default="npu", choices=["cpu", "cuda", "npu"])
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--validate-safetensors", action="store_true")
    parser.add_argument("--wait", action="store_true", help="Poll until assets are complete before running alignment.")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--timeout-seconds", type=int, default=0, help="0 means no timeout when --wait is used.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_command(cmd, *, capture: bool):
    if capture:
        return subprocess.run(cmd, check=False, text=True, capture_output=True)
    return subprocess.run(cmd, check=False)


def check_assets(args):
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/check_llada2_assets.py"),
        "--model-path",
        args.model_path,
        "--allow-incomplete",
    ]
    if args.validate_safetensors:
        cmd.append("--validate-safetensors")
    result = run_command(cmd, capture=True)
    if result.returncode != 0:
        raise RuntimeError(f"asset check failed:\n{result.stderr}\n{result.stdout}")
    return json.loads(result.stdout)


def wait_for_assets(args):
    start = time.monotonic()
    while True:
        status = check_assets(args)
        if status["complete"]:
            return status
        summary = {
            "complete": False,
            "missing_count": len(status["missing"]),
            "missing": status["missing"][:20],
            "temp_file_count": len(status["temp_files"]),
            "temp_total_bytes": sum(item["size"] for item in status["temp_files"]),
        }
        if not args.wait:
            return status
        print(json.dumps({"status": "waiting_for_assets", **summary}, sort_keys=True), flush=True)
        if args.timeout_seconds and time.monotonic() - start >= args.timeout_seconds:
            raise TimeoutError("timed out waiting for LLaDA2 assets")
        time.sleep(args.poll_seconds)


def build_sample_cmd(args):
    return [
        sys.executable,
        str(REPO_ROOT / "scripts/create_llada2_alignment_sample.py"),
        "--model-path",
        args.model_path,
        "--output-path",
        args.sample_path,
        "--max-length",
        str(args.max_seq_len),
    ]


def build_alignment_cmd(args):
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/run_llada2_real_precision_alignment.py"),
        "--legacy-repo",
        args.legacy_repo,
        "--current-repo",
        args.current_repo,
        "--config-path",
        args.config_path,
        "--model-path",
        args.model_path,
        "--sample-path",
        args.sample_path,
        "--sample-index",
        "0",
        "--max-seq-len",
        str(args.max_seq_len),
        "--attn",
        args.attn,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
    ]
    if args.backward:
        cmd.append("--backward")
    return cmd


def main():
    args = parse_args()
    sample_cmd = build_sample_cmd(args)
    alignment_cmd = build_alignment_cmd(args)

    if args.dry_run:
        print(json.dumps({"sample_command": sample_cmd, "alignment_command": alignment_cmd}, indent=2))
        return

    status = wait_for_assets(args)
    if not status["complete"]:
        print(
            json.dumps(
                {
                    "status": "assets_incomplete",
                    "missing": status["missing"],
                    "temp_file_count": len(status["temp_files"]),
                    "temp_total_bytes": sum(item["size"] for item in status["temp_files"]),
                },
                indent=2,
                sort_keys=True,
            )
        )
        sys.exit(2)

    sample_result = run_command(sample_cmd, capture=True)
    if sample_result.returncode != 0:
        raise RuntimeError(f"sample creation failed:\n{sample_result.stderr}\n{sample_result.stdout}")
    print(sample_result.stdout, end="")

    alignment_result = run_command(alignment_cmd, capture=True)
    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if alignment_result.returncode != 0:
        result_path.write_text(
            json.dumps(
                {
                    "status": "alignment_failed",
                    "returncode": alignment_result.returncode,
                    "stdout": alignment_result.stdout,
                    "stderr": alignment_result.stderr,
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise RuntimeError(
            f"alignment failed; stdout saved to {result_path}\n{alignment_result.stderr}\n{alignment_result.stdout}"
        )
    result_path.write_text(alignment_result.stdout)
    print(alignment_result.stdout, end="")
    print(json.dumps({"status": "alignment_complete", "result_path": str(result_path)}, indent=2))


if __name__ == "__main__":
    main()
