#!/usr/bin/env python3
from __future__ import annotations

"""Create a deterministic JSONL sample for LLaDA2 precision alignment."""

import argparse
import json
from pathlib import Path

DEFAULT_TEXT = "A short deterministic LLaDA2 precision alignment sample."


def parse_args():
    parser = argparse.ArgumentParser(description="Create a fixed input_ids sample for LLaDA2 alignment.")
    parser.add_argument("--model-path", required=True, help="Directory containing the tokenizer files.")
    parser.add_argument("--output-path", required=True, help="JSONL file to write.")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--add-special-tokens", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    input_ids = tokenizer(
        args.text,
        add_special_tokens=args.add_special_tokens,
        truncation=True,
        max_length=args.max_length,
    )["input_ids"]
    if len(input_ids) < 2:
        raise ValueError("Alignment sample must contain at least two tokens.")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"input_ids": input_ids}) + "\n")
    print(json.dumps({"output_path": str(output_path), "tokens": len(input_ids)}, indent=2))


if __name__ == "__main__":
    main()
