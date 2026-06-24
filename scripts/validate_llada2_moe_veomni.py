#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "VeOmni"))

import torch
import torch.nn.functional as F

import models.llada2_moe  # noqa: F401 - registers model/config with VeOmni.
from models.llada2_moe.configuration_llada2_moe import LLaDA2MoeConfig
from models.llada2_moe.modeling_llada2_moe import LLaDA2MoeModelLM
from veomni.arguments import OpsImplementationConfig
from veomni.ops import apply_ops_config


def parse_args():
    parser = argparse.ArgumentParser(description="Validate LLaDA2 MoE against latest VeOmni ops dispatch.")
    parser.add_argument("--backend", default="eager", choices=["eager", "fused_npu", "fused_triton", "fused_quack"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "npu"])
    parser.add_argument("--compare-eager", action="store_true")
    parser.add_argument("--attn", default=None, choices=["eager", "sdpa"])
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def resolve_device(requested: str, backend: str):
    if requested != "auto":
        return requested
    if backend == "fused_npu":
        return "npu"
    if backend in {"fused_triton", "fused_quack"} and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def make_ops_config(backend: str, device: str, attn: str):
    if device == "npu":
        return OpsImplementationConfig(
            attn_implementation=attn,
            moe_implementation=backend,
            cross_entropy_loss_implementation="npu",
            rms_norm_implementation="npu",
            swiglu_mlp_implementation="eager",
            rotary_pos_emb_implementation="npu",
            rotary_pos_emb_vision_implementation="eager",
            load_balancing_loss_implementation="eager",
            rms_norm_gated_implementation="eager",
            causal_conv1d_implementation="eager",
            chunk_gated_delta_rule_implementation="eager",
        )

    return OpsImplementationConfig(
        attn_implementation=attn,
        moe_implementation=backend,
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


def make_config(seq_len: int, attn: str):
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
    config._attn_implementation = attn
    return config


def move_model(model, device: str, backend: str):
    del backend
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    return model.to(device=device, dtype=dtype)


def run_once(backend: str, device: str, attn: str, config, input_ids, labels, state_dict=None):
    apply_ops_config(make_ops_config(backend, device, attn))
    model = move_model(LLaDA2MoeModelLM(config), device, backend)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    model.train()
    model.zero_grad(set_to_none=True)

    outputs = model(input_ids=input_ids, use_cache=False, output_router_logits=False)
    logits = outputs.logits.float()
    loss = F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )
    loss.backward()

    grad_name = "model.layers.0.mlp.experts.gate_proj"
    grad = dict(model.named_parameters())[grad_name].grad.detach().float().cpu()
    return {
        "model": model,
        "logits": logits.detach().float().cpu(),
        "loss": float(loss.detach().cpu()),
        "grad": grad,
        "kernel": _current_moe_kernel_name(),
    }


def _current_moe_kernel_name():
    try:
        from veomni.ops.kernels import moe
    except Exception:
        return "unavailable"

    kernel = getattr(moe, "_fused_moe_forward", None)
    return getattr(kernel, "__name__", "None") if kernel is not None else "None"


def diff_metrics(actual: torch.Tensor, expected: torch.Tensor):
    diff = (actual - expected).abs()
    denom = expected.abs().clamp_min(1e-6)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_rel": float((diff / denom).max().item()),
    }


def main():
    args = parse_args()
    device = resolve_device(args.device, args.backend)
    if device == "npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)

    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    elif device == "npu":
        torch.npu.manual_seed_all(args.seed)

    attn = args.attn or ("sdpa" if device == "npu" else "eager")
    config = make_config(args.seq_len, attn)
    input_ids = torch.randint(1, config.vocab_size, (args.batch_size, args.seq_len), device=device)
    labels = input_ids.clone()
    labels[:, : args.seq_len // 2] = -100

    if args.compare_eager and args.backend != "eager":
        eager = run_once("eager", device, attn, config, input_ids, labels)
        state_dict = {k: v.detach().clone() for k, v in eager["model"].state_dict().items()}
        actual = run_once(args.backend, device, attn, config, input_ids, labels, state_dict=state_dict)
        result = {
            "device": device,
            "attn": attn,
            "backend": args.backend,
            "reference_backend": "eager",
            "kernel": actual["kernel"],
            "reference_kernel": eager["kernel"],
            "loss": actual["loss"],
            "reference_loss": eager["loss"],
            "loss_abs_diff": abs(actual["loss"] - eager["loss"]),
            "logits": diff_metrics(actual["logits"], eager["logits"]),
            "grad": diff_metrics(actual["grad"], eager["grad"]),
        }
    else:
        actual = run_once(args.backend, device, attn, config, input_ids, labels)
        result = {
            "device": device,
            "attn": attn,
            "backend": args.backend,
            "kernel": actual["kernel"],
            "loss": actual["loss"],
            "logits_shape": list(actual["logits"].shape),
            "grad_max_abs": float(actual["grad"].abs().max().item()),
        }

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
