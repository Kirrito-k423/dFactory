#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="Compare LLaDA2 SP/EP behavior between legacy and current dFactory.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    worker = subparsers.add_parser("worker", help="Run one distributed worker under torchrun.")
    worker.add_argument("--repo", required=True)
    worker.add_argument("--case", required=True, choices=["sp", "ep"])
    worker.add_argument("--out-dir", required=True)
    worker.add_argument("--seed", type=int, default=20260625)
    worker.add_argument("--seq-len", type=int, default=16)
    worker.add_argument("--batch-size", type=int, default=2)
    worker.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    worker.add_argument("--device", default="npu", choices=["npu", "cpu"])

    compare = subparsers.add_parser("compare", help="Compare two worker output directories.")
    compare.add_argument("--current-dir", required=True)
    compare.add_argument("--legacy-dir", required=True)
    compare.add_argument("--case", required=True, choices=["sp", "ep"])

    return parser.parse_args()


def install_transformers_compat():
    import transformers
    import transformers.modeling_utils as modeling_utils
    import transformers.pytorch_utils as pytorch_utils
    import transformers.utils.import_utils as import_utils

    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: True
    if not hasattr(pytorch_utils, "is_torch_greater_or_equal_than_1_13"):
        pytorch_utils.is_torch_greater_or_equal_than_1_13 = lambda: True
    if not hasattr(modeling_utils, "no_init_weights"):
        @contextlib.contextmanager
        def no_init_weights(*_args, **_kwargs):
            yield

        modeling_utils.no_init_weights = no_init_weights
    if not hasattr(transformers, "AutoModelForVision2Seq") and hasattr(transformers, "AutoModelForImageTextToText"):
        transformers.AutoModelForVision2Seq = transformers.AutoModelForImageTextToText

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


def make_config(seq_len: int, attn: str, fused: bool):
    from models.llada2_moe.configuration_llada2_moe import LLaDA2MoeConfig

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
    config.model_type = "llada2_moe_veomni" if fused else "llada2_moe_legacy_eager"
    config._attn_implementation = attn
    return config


def init_dist_and_parallel(case: str, device: str):
    if device == "npu":
        import torch_npu  # noqa: F401

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.npu.set_device(local_rank)
        backend = "hccl"
        device_type = "npu"
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        backend = "gloo"
        device_type = "cpu"

    import torch.distributed as dist

    dist.init_process_group(backend=backend)
    world_size = dist.get_world_size()

    from veomni.distributed.parallel_state import init_parallel_state

    if case == "sp":
        dp_size = world_size // 2
        ulysses_size = 2
        ep_size = 1
    else:
        dp_size = world_size
        ulysses_size = 1
        ep_size = 2

    try:
        init_parallel_state(
            dp_size=dp_size,
            dp_replicate_size=1,
            dp_shard_size=dp_size,
            tp_size=1,
            pp_size=1,
            cp_size=1,
            ulysses_size=ulysses_size,
            dp_mode="ddp",
            device_type=device_type,
            extra_parallel_sizes=(ep_size,),
            extra_parallel_placement_innermost=(False,),
            extra_parallel_names=("ep",),
        )
    except TypeError:
        init_parallel_state(
            dp_size=dp_size,
            dp_replicate_size=1,
            dp_shard_size=dp_size,
            tp_size=1,
            ep_size=ep_size,
            pp_size=1,
            cp_size=1,
            ulysses_size=ulysses_size,
            dp_mode="ddp",
            device_type=device_type,
        )

    return dist, local_rank


def apply_current_ops(case: str, attn: str, device: str):
    from veomni.arguments import OpsImplementationConfig
    from veomni.ops import apply_ops_config

    if case == "ep" and device == "npu":
        moe_impl = "fused_npu"
        norm_impl = "npu"
        rotary_impl = "npu"
    else:
        moe_impl = "eager"
        norm_impl = "eager"
        rotary_impl = "eager"

    apply_ops_config(
        OpsImplementationConfig(
            attn_implementation=attn,
            moe_implementation=moe_impl,
            cross_entropy_loss_implementation="eager",
            rms_norm_implementation=norm_impl,
            swiglu_mlp_implementation="eager",
            rotary_pos_emb_implementation=rotary_impl,
            rotary_pos_emb_vision_implementation="eager",
            load_balancing_loss_implementation="eager",
            rms_norm_gated_implementation="eager",
            causal_conv1d_implementation="eager",
            chunk_gated_delta_rule_implementation="eager",
        )
    )


def apply_ep_plan(model):
    from veomni.distributed.parallel_state import get_parallel_state

    parallel_state = get_parallel_state()
    plan = model.get_parallel_plan()
    if hasattr(parallel_state, "extra_parallel_fsdp_device_mesh"):
        mesh = parallel_state.extra_parallel_fsdp_device_mesh
    else:
        mesh = parallel_state.ep_fsdp_device_mesh
    return plan.apply(model, mesh)


def parallel_summary():
    from veomni.distributed.parallel_state import get_parallel_state

    ps = get_parallel_state()
    def safe_int(name: str, default: int = 0):
        try:
            return int(getattr(ps, name))
        except Exception:
            return default

    summary = {
        "sp_enabled": bool(ps.sp_enabled),
        "sp_size": int(ps.sp_size),
        "sp_rank": safe_int("sp_rank"),
        "ulysses_enabled": bool(ps.ulysses_enabled),
        "ulysses_rank": safe_int("ulysses_rank"),
        "ep_enabled": bool(ps.ep_enabled),
        "ep_size": int(ps.ep_size),
        "ep_rank": safe_int("ep_rank"),
    }
    return summary


def diff_metrics(actual: torch.Tensor, expected: torch.Tensor):
    if actual.shape != expected.shape:
        return {"shape_mismatch": [list(actual.shape), list(expected.shape)]}
    diff = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1e-6)
    return {
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "max_rel": float((diff / denom).max().item()) if diff.numel() else 0.0,
    }


def run_worker(args):
    repo = Path(args.repo).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    result_path = out_dir / f"rank{rank}.pt"
    json_path = out_dir / f"rank{rank}.json"

    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "VeOmni"))
    install_transformers_compat()

    dist = None
    try:
        dist, local_rank = init_dist_and_parallel(args.case, args.device)
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", local_rank))
        torch.manual_seed(args.seed)
        if args.device == "npu":
            torch.npu.manual_seed_all(args.seed)

        is_current = (repo / "VeOmni" / "veomni" / "arguments").exists()
        if is_current:
            import models.llada2_moe  # noqa: F401

            apply_current_ops(args.case, args.attn, args.device)

        from models.llada2_moe.modeling_llada2_moe import LLaDA2MoeModelLM

        fused = args.case == "ep"
        config = make_config(args.seq_len, args.attn, fused=fused)
        device = torch.device(f"npu:{local_rank}") if args.device == "npu" else torch.device("cpu")
        dtype = torch.bfloat16 if args.device == "npu" else torch.float32
        model = LLaDA2MoeModelLM(config).to(device=device, dtype=dtype)
        model.train()

        ep_spec_count = 0
        if args.case == "ep":
            fqn2spec = apply_ep_plan(model)
            ep_spec_count = len(fqn2spec)

        torch.manual_seed(args.seed + 1)
        input_ids = torch.randint(1, config.vocab_size, (args.batch_size, args.seq_len), device=device)
        labels = input_ids.clone()
        labels[:, : args.seq_len // 2] = -100

        model.zero_grad(set_to_none=True)
        outputs = model(input_ids=input_ids, use_cache=False, output_router_logits=False)
        logits = outputs.logits.float()
        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
        )
        loss.backward()

        grad_chunks = []
        grad_shapes = {}
        param_shapes = {}
        for name, param in model.named_parameters():
            if "mlp.experts" in name or name in {"model.embed_tokens.weight", "lm_head.weight"}:
                param_shapes[name] = list(param.shape)
            if param.grad is not None and ("mlp.experts" in name or name.endswith("embed_tokens.weight")):
                grad_shapes[name] = list(param.grad.shape)
                grad_chunks.append(param.grad.detach().float().cpu().reshape(-1))

        kernel = "unknown"
        try:
            if is_current:
                from veomni.ops.kernels import moe

                kernel_fn = getattr(moe, "_fused_moe_forward", None)
                kernel = getattr(kernel_fn, "__name__", "None") if kernel_fn is not None else "None"
            else:
                kernel = "legacy_fused_moe_forward" if fused else "legacy_eager"
        except Exception as exc:
            kernel = f"unavailable:{exc}"

        result = {
            "ok": True,
            "repo": str(repo),
            "case": args.case,
            "rank": rank,
            "local_rank": local_rank,
            "device": str(device),
            "dtype": str(dtype),
            "is_current": is_current,
            "fused": fused,
            "kernel": kernel,
            "parallel": parallel_summary(),
            "ep_spec_count": ep_spec_count,
            "loss": torch.tensor(float(loss.detach().cpu())),
            "logits": logits.detach().cpu(),
            "grad": torch.cat(grad_chunks) if grad_chunks else torch.empty(0),
            "param_shapes": param_shapes,
            "grad_shapes": grad_shapes,
        }
        torch.save(result, result_path)
        json_path.write_text(
            json.dumps(
                {k: v for k, v in result.items() if k not in {"logits", "grad"}},
                indent=2,
                sort_keys=True,
                default=lambda x: float(x.item()) if torch.is_tensor(x) and x.numel() == 1 else str(x),
            )
        )
        if dist is not None:
            dist.barrier()
    except Exception as exc:
        failure = {
            "ok": False,
            "repo": str(repo),
            "case": args.case,
            "rank": rank,
            "local_rank": local_rank,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        torch.save(failure, result_path)
        json_path.write_text(json.dumps(failure, indent=2, sort_keys=True))
        raise
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def load_outputs(path: Path):
    outputs = []
    for file in sorted(path.glob("rank*.pt")):
        outputs.append(torch.load(file, map_location="cpu"))
    return outputs


def run_compare(args):
    current = load_outputs(Path(args.current_dir))
    legacy = load_outputs(Path(args.legacy_dir))
    result = {
        "case": args.case,
        "current_dir": str(Path(args.current_dir).resolve()),
        "legacy_dir": str(Path(args.legacy_dir).resolve()),
        "current_ok": all(item.get("ok") for item in current),
        "legacy_ok": all(item.get("ok") for item in legacy),
        "ranks": [],
    }
    if len(current) != len(legacy):
        result["rank_count_mismatch"] = [len(current), len(legacy)]

    for cur, old in zip(current, legacy):
        rank_result = {
            "rank": cur.get("rank"),
            "current_loss": float(cur["loss"].item()) if cur.get("ok") else None,
            "legacy_loss": float(old["loss"].item()) if old.get("ok") else None,
            "current_parallel": cur.get("parallel"),
            "legacy_parallel": old.get("parallel"),
            "current_kernel": cur.get("kernel"),
            "legacy_kernel": old.get("kernel"),
            "current_param_shapes": cur.get("param_shapes"),
            "legacy_param_shapes": old.get("param_shapes"),
        }
        if cur.get("ok") and old.get("ok"):
            rank_result["loss_abs_diff"] = abs(float(cur["loss"].item()) - float(old["loss"].item()))
            rank_result["logits"] = diff_metrics(cur["logits"], old["logits"])
            rank_result["grad"] = diff_metrics(cur["grad"], old["grad"])
        else:
            rank_result["current_error"] = cur.get("error")
            rank_result["legacy_error"] = old.get("error")
        result["ranks"].append(rank_result)

    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    args = parse_args()
    if args.cmd == "worker":
        run_worker(args)
    elif args.cmd == "compare":
        run_compare(args)


if __name__ == "__main__":
    main()
