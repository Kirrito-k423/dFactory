#!/usr/bin/env python3
from __future__ import annotations

"""Run old-vs-new LLaDA2 precision alignment when real assets are available.

This script is intentionally asset-driven: it does not download model weights
or datasets. Point it at a legacy dFactory checkout, the current checkout,
LLaDA2 config/weights/tokenizer paths, and one JSON/JSONL sample file.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


WORKER_CODE = r"""
import json
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


repo = Path(sys.argv[1]).resolve()
config_path = Path(sys.argv[2]).resolve()
model_path_arg = sys.argv[3]
sample_path = Path(sys.argv[4]).resolve()
output_path = Path(sys.argv[5]).resolve()
mode = sys.argv[6]
attn = sys.argv[7]
max_seq_len = int(sys.argv[8])
sample_index = int(sys.argv[9])
text_key = sys.argv[10]
mask_token_id = int(sys.argv[11])
device = sys.argv[12]
dtype_name = sys.argv[13]
enable_backward = sys.argv[14].lower() == "true"

sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "VeOmni"))


def install_transformers_compat():
    import transformers.pytorch_utils as pytorch_utils
    import transformers.utils.import_utils as import_utils

    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: True
    if not hasattr(pytorch_utils, "is_torch_greater_or_equal_than_1_13"):
        pytorch_utils.is_torch_greater_or_equal_than_1_13 = lambda: True

    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" not in ROPE_INIT_FUNCTIONS:
        def default_rope_init(config, device=None, **kwargs):
            del kwargs
            head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
            dim = int(head_dim * partial_rotary_factor)
            inv_freq = 1.0 / (
                config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
            )
            return inv_freq, 1.0

        ROPE_INIT_FUNCTIONS["default"] = default_rope_init


def read_sample(path: Path, index: int):
    if path.suffix == ".jsonl":
        with path.open() as handle:
            for row_idx, line in enumerate(handle):
                if row_idx == index:
                    return json.loads(line)
        raise IndexError(f"sample_index={index} out of range for {path}")
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data[index]
    return data


def reference_fused_moe_forward(module, num_experts, routing_weights, selected_experts, hidden_states, fc1_1_weight, fc1_2_weight, fc2_weight):
    output = torch.zeros_like(hidden_states)
    act_fn = getattr(module, "act_fn", F.silu)
    for expert_idx in range(num_experts):
        token_mask = selected_experts == expert_idx
        if not token_mask.any():
            continue
        token_indices, topk_indices = torch.where(token_mask)
        expert_input = hidden_states[token_indices]
        gate = F.linear(expert_input, fc1_1_weight[expert_idx])
        up = F.linear(expert_input, fc1_2_weight[expert_idx])
        expert_output = F.linear(act_fn(gate) * up, fc2_weight[expert_idx])
        weighted_output = expert_output * routing_weights[token_indices, topk_indices].unsqueeze(-1)
        output.index_add_(0, token_indices, weighted_output.to(output.dtype))
    return output


install_transformers_compat()

if mode == "current":
    import models.llada2_moe  # noqa: F401
    from veomni.arguments import OpsImplementationConfig
    from veomni.ops import apply_ops_config

    apply_ops_config(
        OpsImplementationConfig(
            attn_implementation=attn,
            moe_implementation="eager",
            cross_entropy_loss_implementation="eager",
            rms_norm_implementation="eager",
            swiglu_mlp_implementation="eager",
            rotary_pos_emb_implementation="eager",
            rotary_pos_emb_vision_implementation="eager",
            load_balancing_loss_implementation="eager",
            rms_norm_gated_implementation="eager",
            causal_conv1d_implementation="eager",
            chunk_gated_delta_rule_implementation="eager",
        )
    )

from models.llada2_moe.configuration_llada2_moe import LLaDA2MoeConfig
from models.llada2_moe.modeling_llada2_moe import LLaDA2MoeModelLM

if mode == "legacy":
    import models.llada2_moe.modeling_llada2_moe as modeling

    modeling.fused_moe_forward = reference_fused_moe_forward

from transformers import AutoTokenizer


def resolve_dtype(name: str):
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def load_weights(model, model_path: Path):
    if str(model_path) == "":
        return {"missing": [], "unexpected": []}
    try:
        from safetensors.torch import load_file
    except Exception as exc:
        raise RuntimeError("safetensors is required for real-weight alignment") from exc

    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
    elif (model_path / "model.safetensors").exists():
        weight_map = {}
        shards = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"No safetensors weights found under {model_path}")

    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    loaded_keys = set()
    unexpected_keys = set()
    expert_key_pattern = re.compile(r"^(?P<prefix>.+\.mlp\.experts)\.(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$")
    grouped_expert_pattern = re.compile(r"^(?P<prefix>.+\.mlp\.experts)\.(?P<proj>gate_proj|up_proj|down_proj)$")
    grouped_expert_keys = {
        key
        for key in model_keys
        if grouped_expert_pattern.match(key)
    }

    def grouped_key_for_checkpoint_key(key: str):
        match = expert_key_pattern.match(key)
        if not match:
            return None
        grouped_key = f"{match.group('prefix')}.{match.group('proj')}"
        if grouped_key not in grouped_expert_keys:
            return None
        return grouped_key, int(match.group("expert"))

    for shard in shards:
        shard_state = load_file(str(model_path / shard), device="cpu")
        direct_state = {}
        for key, value in shard_state.items():
            if key in model_keys:
                direct_state[key] = value
            elif grouped_key_for_checkpoint_key(key) is None:
                unexpected_keys.add(key)
        loaded_keys.update(direct_state.keys())
        result = model.load_state_dict(direct_state, strict=False)
        unexpected_keys.update(result.unexpected_keys)
        del shard_state

    if grouped_expert_keys:
        layer_locations = {}
        grouped_keys_by_layer = {}
        for key, shard in weight_map.items():
            mapped = grouped_key_for_checkpoint_key(key)
            if mapped is None:
                continue
            grouped_key, expert_id = mapped
            layer_key = grouped_key.rsplit(".", 1)[0]
            grouped_keys_by_layer.setdefault(layer_key, set()).add(grouped_key)
            layer_locations.setdefault(layer_key, {}).setdefault(shard, []).append((grouped_key, expert_id, key))

        for layer_key in sorted(grouped_keys_by_layer):
            locations = layer_locations.get(layer_key, {})
            if not locations:
                continue
            grouped_tensors = {
                grouped_key: torch.empty_like(model_state[grouped_key], device="cpu")
                for grouped_key in sorted(grouped_keys_by_layer[layer_key])
            }
            seen_experts = {grouped_key: set() for grouped_key in grouped_tensors}
            for shard, entries in sorted(locations.items()):
                shard_state = load_file(str(model_path / shard), device="cpu")
                for grouped_key, expert_id, checkpoint_key in entries:
                    if checkpoint_key not in shard_state:
                        continue
                    grouped_tensor = grouped_tensors[grouped_key]
                    grouped_tensor[expert_id].copy_(shard_state[checkpoint_key].to(grouped_tensor.dtype))
                    seen_experts[grouped_key].add(expert_id)
                del shard_state
            layer_state = {
                grouped_key: grouped_tensor
                for grouped_key, grouped_tensor in grouped_tensors.items()
                if len(seen_experts[grouped_key]) == grouped_tensor.shape[0]
            }
            if layer_state:
                result = model.load_state_dict(layer_state, strict=False)
                loaded_keys.update(layer_state.keys())
                unexpected_keys.update(result.unexpected_keys)
            del grouped_tensors

    return {
        "missing": sorted(model_keys - loaded_keys),
        "unexpected": sorted(unexpected_keys),
        "shards_loaded": shards,
    }


config = LLaDA2MoeConfig.from_pretrained(str(config_path))
config._attn_implementation = attn
tokenizer = AutoTokenizer.from_pretrained(str(Path(model_path_arg) if model_path_arg else config_path), trust_remote_code=True)
if device == "npu":
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)

dtype = resolve_dtype(dtype_name)
model = LLaDA2MoeModelLM(config).to(dtype=dtype)
load_result = load_weights(model, Path(model_path_arg)) if model_path_arg else {"missing": [], "unexpected": []}
model = model.to(device=device, dtype=dtype)
model.eval()

sample = read_sample(sample_path, sample_index)
if "input_ids" in sample:
    input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)[:max_seq_len].unsqueeze(0)
elif text_key in sample:
    value = sample[text_key]
    if isinstance(value, list):
        text = tokenizer.apply_chat_template(value, tokenize=False)
    else:
        text = str(value)
    input_ids = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
        padding="max_length",
        add_special_tokens=False,
    ).input_ids
else:
    raise KeyError(f"Sample does not contain 'input_ids' or {text_key!r}")

labels = input_ids.clone()
labels[:, : input_ids.shape[1] // 2] = -100
input_ids = input_ids.to(device)
labels = labels.to(device)
model.zero_grad(set_to_none=True)
with torch.enable_grad() if enable_backward else torch.no_grad():
    outputs = model(input_ids=input_ids, use_cache=False, output_router_logits=False)
    logits = outputs.logits.float()
    loss = F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )
    if enable_backward:
        loss.backward()

grads = []
if enable_backward:
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.detach().float().cpu().reshape(-1))

torch.save(
    {
        "loss": torch.tensor(float(loss.detach().cpu())),
        "logits": logits.detach().cpu(),
        "grad": torch.cat(grads) if grads else torch.empty(0),
        "load_result": load_result,
        "backward": enable_backward,
    },
    output_path,
)

print(json.dumps({"repo": str(repo), "mode": mode, "loss": float(loss.detach().cpu()), "load_result": load_result}))
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Run real-asset LLaDA2 old/new precision alignment.")
    parser.add_argument("--legacy-repo", required=True, help="Path to dFactory checkout before this migration.")
    parser.add_argument("--current-repo", default=str(REPO_ROOT), help="Path to current migrated dFactory checkout.")
    parser.add_argument("--config-path", required=True, help="LLaDA2 config directory.")
    parser.add_argument("--model-path", required=True, help="LLaDA2 safetensors weight directory.")
    parser.add_argument("--sample-path", required=True, help="JSON or JSONL sample file.")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--text-key", default="messages")
    parser.add_argument("--mask-token-id", type=int, default=156895)
    parser.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "npu"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--backward", action="store_true")
    return parser.parse_args()


def run_worker(repo: Path, args, output_path: Path, mode: str):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo}:{repo / 'VeOmni'}"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            WORKER_CODE,
            str(repo),
            args.config_path,
            args.model_path,
            args.sample_path,
            str(output_path),
            mode,
            args.attn,
            str(args.max_seq_len),
            str(args.sample_index),
            args.text_key,
            str(args.mask_token_id),
            args.device,
            args.dtype,
            str(args.backward),
        ],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{mode} worker failed with rc={result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def diff_metrics(actual: torch.Tensor, expected: torch.Tensor):
    diff = (actual - expected).abs()
    denom = expected.abs().clamp_min(1e-6)
    return {
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "max_rel": float((diff / denom).max().item()) if diff.numel() else 0.0,
    }


def main():
    args = parse_args()
    import torch

    current_repo = Path(args.current_repo).resolve()
    legacy_repo = Path(args.legacy_repo).resolve()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        current_out = tmp / "current.pt"
        legacy_out = tmp / "legacy.pt"

        current_log = run_worker(current_repo, args, current_out, "current")
        legacy_log = run_worker(legacy_repo, args, legacy_out, "legacy")

        current = torch.load(current_out, map_location="cpu")
        legacy = torch.load(legacy_out, map_location="cpu")
        result = {
            "current_repo": str(current_repo),
            "legacy_repo": str(legacy_repo),
            "config_path": args.config_path,
            "model_path": args.model_path,
            "sample_path": args.sample_path,
            "sample_index": args.sample_index,
            "attn": args.attn,
            "device": args.device,
            "dtype": args.dtype,
            "backward": args.backward,
            "current_worker": current_log,
            "legacy_worker": legacy_log,
            "current_loss": float(current["loss"].item()),
            "legacy_loss": float(legacy["loss"].item()),
            "loss_abs_diff": float(abs(current["loss"].item() - legacy["loss"].item())),
            "logits": diff_metrics(current["logits"], legacy["logits"]),
            "grad": diff_metrics(current["grad"], legacy["grad"]),
            "current_load_result": current["load_result"],
            "legacy_load_result": legacy["load_result"],
        }
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
