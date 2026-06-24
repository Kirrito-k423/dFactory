#!/usr/bin/env python3
from __future__ import annotations

"""Check whether LLaDA2 validation assets are complete on disk."""

import argparse
import json
import sys
from pathlib import Path


DEFAULT_REQUIRED_FILES = [
    "config.json",
    "configuration_llada2_moe.py",
    "generation_config.json",
    "model.safetensors.index.json",
    "modeling_llada2_moe.py",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Check LLaDA2 downloaded asset completeness.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--expected-shards", type=int, default=7)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--validate-safetensors", action="store_true")
    return parser.parse_args()


def read_index_shards(model_path: Path):
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        return []
    data = json.loads(index_path.read_text())
    return sorted(set(data.get("weight_map", {}).values()))


def expected_shards(args, model_path: Path):
    indexed = read_index_shards(model_path)
    if indexed:
        return indexed
    return [f"model-{idx:05d}-of-{args.expected_shards:05d}.safetensors" for idx in range(1, args.expected_shards + 1)]


def check_safetensors(path: Path):
    try:
        from safetensors import safe_open
    except Exception as exc:
        return {"ok": False, "error": f"safetensors import failed: {exc}"}

    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return {"ok": True, "tensors": len(handle.keys())}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def file_record(path: Path, validate: bool):
    record = {"exists": path.exists(), "size": path.stat().st_size if path.exists() else 0}
    if validate and path.exists() and path.suffix == ".safetensors":
        record["safetensors"] = check_safetensors(path)
    return record


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    required = list(DEFAULT_REQUIRED_FILES)
    shards = expected_shards(args, model_path)
    required.extend(shards)

    files = {name: file_record(model_path / name, args.validate_safetensors) for name in required}
    missing = [name for name, record in files.items() if not record["exists"]]
    empty = [name for name, record in files.items() if record["exists"] and record["size"] <= 0]
    invalid = [
        name
        for name, record in files.items()
        if record.get("safetensors") and not record["safetensors"].get("ok", False)
    ]
    temp_files = []
    for temp_dir in [model_path / "._____temp", model_path / ".cache" / "huggingface" / "download"]:
        if not temp_dir.exists():
            continue
        for path in temp_dir.rglob("*"):
            if path.is_file():
                temp_files.append({"path": str(path), "size": path.stat().st_size})
    temp_files.sort(key=lambda record: record["path"])

    complete = not missing and not empty and not invalid
    result = {
        "model_path": str(model_path),
        "complete": complete,
        "missing": missing,
        "empty": empty,
        "invalid": invalid,
        "required_file_count": len(required),
        "shard_count": len(shards),
        "files": files,
        "temp_files": temp_files,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not complete and not args.allow_incomplete:
        sys.exit(2)


if __name__ == "__main__":
    main()
