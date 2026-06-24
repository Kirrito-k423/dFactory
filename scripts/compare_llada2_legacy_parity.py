#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


WORKER_CODE = r"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


repo = Path(sys.argv[1]).resolve()
state_path = Path(sys.argv[2]).resolve()
input_path = Path(sys.argv[3]).resolve()
output_path = Path(sys.argv[4]).resolve()
mode = sys.argv[5]
seed = int(sys.argv[6])
attn = sys.argv[7]

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


install_transformers_compat()

from models.llada2_moe.configuration_llada2_moe import LLaDA2MoeConfig
from models.llada2_moe.modeling_llada2_moe import LLaDA2MoeModelLM


def make_config(seq_len):
    config = LLaDA2MoeConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        attention_dropout=0.0,
        embedding_dropout=0.0,
        output_dropout=0.0,
        max_position_embeddings=max(128, seq_len),
        num_experts=4,
        num_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        routed_scaling_factor=1.0,
        moe_intermediate_size=32,
        first_k_dense_replace=0,
        pad_token_id=0,
        tie_word_embeddings=False,
        output_router_logits=False,
    )
    config.model_type = "llada2_moe_legacy_eager"
    config._attn_implementation = attn
    return config


payload = torch.load(input_path, map_location="cpu")
torch.manual_seed(seed)
config = make_config(payload["input_ids"].shape[1])
model = LLaDA2MoeModelLM(config)
model.train()

if mode == "init":
    torch.save(model.state_dict(), state_path)
else:
    model.load_state_dict(torch.load(state_path, map_location="cpu"), strict=True)

input_ids = payload["input_ids"]
labels = payload["labels"]
model.zero_grad(set_to_none=True)
outputs = model(input_ids=input_ids, use_cache=False, output_router_logits=False)
logits = outputs.logits.float()
loss = F.cross_entropy(
    logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
    labels[:, 1:].contiguous().view(-1),
    ignore_index=-100,
)
loss.backward()

grads = []
for param in model.parameters():
    if param.grad is not None:
        grads.append(param.grad.detach().float().cpu().reshape(-1))

torch.save(
    {
        "loss": torch.tensor(float(loss.detach().cpu())),
        "logits": logits.detach().cpu(),
        "grad": torch.cat(grads) if grads else torch.empty(0),
    },
    output_path,
)
print(json.dumps({"repo": str(repo), "loss": float(loss.detach().cpu()), "grad_numel": int(sum(g.numel() for g in grads))}))
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Compare current LLaDA2 tiny eager path against a legacy checkout.")
    parser.add_argument("--legacy-repo", required=True, help="Path to a pre-migration dFactory checkout.")
    parser.add_argument("--current-repo", default=str(REPO_ROOT), help="Path to the current migrated dFactory checkout.")
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    return parser.parse_args()


def run_worker(repo: Path, state_path: Path, input_path: Path, output_path: Path, mode: str, seed: int, attn: str):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo}:{repo / 'VeOmni'}"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            WORKER_CODE,
            str(repo),
            str(state_path),
            str(input_path),
            str(output_path),
            mode,
            str(seed),
            attn,
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
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
    current_repo = Path(args.current_repo).resolve()
    legacy_repo = Path(args.legacy_repo).resolve()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        input_path = tmp / "inputs.pt"
        state_path = tmp / "state.pt"
        current_out = tmp / "current.pt"
        legacy_out = tmp / "legacy.pt"

        torch.manual_seed(args.seed)
        input_ids = torch.randint(1, 256, (args.batch_size, args.seq_len))
        labels = input_ids.clone()
        labels[:, : args.seq_len // 2] = -100
        torch.save({"input_ids": input_ids, "labels": labels}, input_path)

        current_log = run_worker(current_repo, state_path, input_path, current_out, "init", args.seed, args.attn)
        legacy_log = run_worker(legacy_repo, state_path, input_path, legacy_out, "load", args.seed, args.attn)

        current = torch.load(current_out, map_location="cpu")
        legacy = torch.load(legacy_out, map_location="cpu")
        result = {
            "current_repo": str(current_repo),
            "legacy_repo": str(legacy_repo),
            "attn": args.attn,
            "current_worker": current_log,
            "legacy_worker": legacy_log,
            "current_loss": float(current["loss"].item()),
            "legacy_loss": float(legacy["loss"].item()),
            "loss_abs_diff": float(abs(current["loss"].item() - legacy["loss"].item())),
            "logits": diff_metrics(current["logits"], legacy["logits"]),
            "grad": diff_metrics(current["grad"], legacy["grad"]),
        }
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
