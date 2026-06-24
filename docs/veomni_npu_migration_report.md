# dFactory 兼容最新 VeOmni 与 Ascend NPU 迁移评估报告

日期：2026-06-24  
仓库：[inclusionAI/dFactory](https://github.com/inclusionAI/dFactory)  
旧 VeOmni 基线：`600fe6d7442392fd3ddefad4b6c8d3c0002fed1c`（`v0.1.2`）  
目标 VeOmni 基线：`8ca09d7c87f06ee7c0f69b0ca0c9e9a6b37f2280`（`main`，`[model, ci] fix: GPT-OSS e2e parametrization (#861)`）

## 结论摘要

dFactory 基于 8 个月前的 VeOmni 开发，主要耦合点集中在训练参数 schema、模型注册机制、`build_foundation_model` / `build_parallelize_model` API、MoE fused kernel 调用签名和 checkpoint 保存链路。最新 VeOmni 已经具备 Ascend NPU 的核心 kernel dispatch 能力，但 LLaDA2 MoE 是 dFactory 自定义模型，不在 VeOmni 官方 NPU 支持模型矩阵内，因此需要额外的模型侧兼容层。

本次已完成基础迁移：

- VeOmni 子模块升级到最新 `main` 提交 `8ca09d7`。
- LLaDA2 MoE 接入最新 `MODEL_CONFIG_REGISTRY` / `MODELING_REGISTRY`。
- 训练脚本迁移到最新 `veomni.arguments`、`ops_implementation`、FSDP2、optimizer、dataloader、checkpoint API。
- LLaDA2 MoE 适配新版 `fused_moe_forward` 签名，并支持 `fused_npu` backend 绑定。
- 增加共享权重的 eager MoE fallback，用于基础精度对齐和无 fused kernel 环境验证。
- 新增迁移前/迁移后 tiny eager parity 脚本，用旧 `v0.1.2` checkout 与当前代码做同权重 logits/loss/grad 对齐。
- 禁用 NPU 环境下的 Liger RMSNorm / SwiGLU / RoPE 替换，避免 GPU-only kernel 误用。
- 修复 transformers v5 下 `is_torch_fx_available` 和默认 RoPE registry 的兼容问题。
- 新增 Ascend NPU 配置与 tiny smoke/对齐验证脚本。

基础验证已通过：在 910B2 单卡上，tiny LLaDA2 MoE 的 `fused_npu` 路径明确绑定 `npu_fused_moe_forward`，相同权重/输入下与 eager baseline 的 loss/logits 差异为 0。迁移前旧代码 eager 路径与当前代码 eager 路径在 tiny 模型上的 loss/logits/grad 差异也为 0。

完整生产化建议按 3-5 周排期；若只要求“能在 NPU 上启动小规模 SFT 并通过基础 loss 对齐”，预计 1-2 周。

## 迁移工作点

| 模块 | 旧实现 | 最新 VeOmni 要求 | 本次处理 |
| --- | --- | --- | --- |
| 子模块 | `VeOmni@600fe6d` | `VeOmni@8ca09d7` | 已升级子模块指针 |
| 参数入口 | `veomni.utils.arguments` | `veomni.arguments` | 已迁移 |
| 模型注册 | `ModelRegistry.register_modeling_path` | `MODEL_CONFIG_REGISTRY` / `MODELING_REGISTRY` | 已注册 `llada2_moe_veomni` |
| ops 配置 | `model.attn_implementation` / `model.moe_implementation` | `model.ops_implementation.*` | 已迁移 YAML，并保留旧字段解析兼容 |
| MoE kernel | `fused_moe_forward(module=..., ...)` | 无 `module` 参数，routing weights 需 bf16/fp16 | 已适配签名与 dtype |
| MoE backend 绑定 | 旧版隐式 fused | 最新通过 `ops_implementation.moe_implementation` | 已支持 `eager` / `fused_triton` / `fused_npu` |
| 训练循环 | 旧 FSDP/optimizer/checkpoint 字段 | `train.accelerator.*`、`train.optimizer.*`、`train.checkpoint.*` | 已抽象为 `tasks/train_llada2_common.py` |
| dataloader | 旧 `dataloader_type`、`rmpad` | `data.dataloader.*`、`dyn_bsz` | 已迁移，block diffusion 默认固定 batch |
| HF 权重保存 | `ckpt_to_state_dict + save_model_weights` | `save_hf_safetensor` | 已迁移 |
| 启动脚本 | 依赖 `nvidia-smi` | NPU 需要 `npu-smi` 或显式进程数 | 已支持 NPU fallback |

## Ascend NPU 算子 gap

| 算子/路径 | VeOmni 最新能力 | LLaDA2 当前状态 | Gap 与建议 |
| --- | --- | --- | --- |
| MoE Group GEMM | 支持 `fused_npu`，底层使用 `torch_npu` MoE permute/grouped matmul/unpermute | 已接入并验证 tiny forward/backward | 需要继续验证真实 LLaDA2 权重、EP>1、多卡 FSDP2 下的 routing 和性能 |
| Attention | VeOmni 文档推荐 NPU 使用 FA/SDPA/CANN 路径 | LLaDA2 自定义 attention 仅支持 `eager` / `sdpa` / `flex_attention` | 无 LLaDA2 专属 flash_attention_2/sequence-parallel patch；生产性能优化需补 LLaDA2 attention patchgen |
| RMSNorm | VeOmni 有 NPU RMSNorm kernel | LLaDA2 使用本地 RMSNorm；NPU 下已避免 Liger | 功能可跑，性能未用 NPU RMSNorm；建议后续接入 OpSlot 或替换为 VeOmni NPU RMSNorm |
| RoPE | VeOmni 有 NPU RoPE kernel | LLaDA2 使用本地 partial rotary；已补 transformers v5 default RoPE fallback | 功能可跑，性能未用 NPU RoPE；建议补 partial rotary 的 NPU kernel 适配 |
| SwiGLU MLP | VeOmni 明确 NPU 无专用 backend，推荐 eager | LLaDA2 dense/shared expert MLP 走 eager | 符合当前能力边界，后续性能优化可评估自定义 NPU fused SwiGLU |
| Cross entropy | VeOmni 支持 `npu` / `chunk_loss` | dFactory 训练循环手写 `F.cross_entropy` | 功能可跑；若要统一 loss dispatch，需要把 MDM loss 收敛到模型 loss 或 VeOmni loss 工具 |
| Load balancing loss | VeOmni 有 Triton/NPU Triton-Ascend 相关路径 | dFactory 当前训练未使用 MoE aux loss | NPU 配置先设 `eager`，后续若启用 aux loss 需专项验证 triton-ascend |
| Qwen3.5 gated trio | VeOmni 对 NPU 多数要求 eager/未支持 | LLaDA2 不涉及 | 与本模型无关 |
| Block diffusion 4D mask | VeOmni 通用训练未覆盖该自定义 mask | 已保留旧逻辑 | 需用真实 SFT 数据跑 1-10 step，确认 NPU SDPA 对 block diffusion mask 的行为和显存 |

## 工作量评估

| 阶段 | 目标 | 估算 |
| --- | --- | --- |
| 基础 API 迁移 | 子模块、注册、训练入口、YAML schema、启动脚本 | 3-5 人日，本次已完成主体 |
| 单卡 NPU 功能验证 | tiny 模型、fused_npu/eager 对齐、基本 forward/backward | 1-2 人日，本次已完成基础验证 |
| 旧代码 tiny parity | 旧 VeOmni v0.1.2 checkout vs 当前代码，同权重/输入 tiny eager loss/logits/grad 对齐 | 1 人日，本次已完成 |
| 真实权重小步 SFT | 准备 LLaDA2 权重/Tokenizer/GSM8K 数据，单卡或 8 卡跑 1-10 step | 2-4 人日，受权重和数据可用性影响 |
| 多卡 FSDP2/EP 验证 | 8 卡 FSDP2、可选 EP、checkpoint 保存/恢复、HF safetensor 导出 | 4-7 人日 |
| 生产级精度对齐 | 旧 VeOmni v0.1.2 baseline vs 新 VeOmni eager/fused，固定 seed/数据/权重，loss 曲线对齐 | 5-8 人日，必须有旧环境和真实权重 |
| 性能优化 | NPU profiling，attention/RMSNorm/RoPE kernel 替换，batch/sequence 并行策略 | 5-10 人日 |
| 文档和 CI | NPU README、smoke 脚本、最小 CI/手工验证矩阵 | 1-3 人日 |

建议排期：

- 基础可用版：1-2 周，目标是 NPU 上能完成小规模 SFT smoke。
- 生产训练版：3-5 周，目标是真实权重、多卡、checkpoint、精度和性能都可交付。

## 已完成验证

本地验证：

```bash
python3 -m py_compile \
  models/llada2_moe/__init__.py \
  models/llada2_moe/configuration_llada2_moe.py \
  models/llada2_moe/modeling_llada2_moe.py \
  tasks/train_llada2_common.py \
  tasks/train_llada2_bd.py \
  tasks/train_llada2_bd_with_dparallel.py \
  scripts/validate_llada2_moe_veomni.py

ruby -ryaml -e 'Dir["configs/sft/llada2_*bd_sft*.yaml"].sort.each { |p| YAML.load_file(p); puts p }'
```

远端 NPU 环境：

- 机器：8 x Ascend 910B2
- 环境：`veomni_qwen35`
- `torch`: `2.7.1+cpu`
- `torch_npu`: `2.7.1`
- `transformers`: `5.2.0`

注册验证：

```text
config_registered True
model_registered True
```

CPU eager smoke：

```json
{
  "attn": "eager",
  "backend": "eager",
  "device": "cpu",
  "logits_shape": [1, 8, 256],
  "loss": 5.636879920959473
}
```

迁移前旧代码 vs 当前代码 tiny eager parity：

```json
{
  "attn": "eager",
  "current_loss": 5.506927490234375,
  "legacy_loss": 5.506927490234375,
  "loss_abs_diff": 0.0,
  "logits": {
    "max_abs": 0.0,
    "mean_abs": 0.0,
    "max_rel": 0.0
  },
  "grad": {
    "max_abs": 0.0,
    "mean_abs": 0.0,
    "max_rel": 0.0
  }
}
```

NPU fused_npu vs eager 对齐：

```json
{
  "attn": "sdpa",
  "backend": "fused_npu",
  "device": "npu",
  "kernel": "npu_fused_moe_forward",
  "reference_backend": "eager",
  "reference_kernel": "None",
  "loss": 5.469104290008545,
  "reference_loss": 5.469104290008545,
  "loss_abs_diff": 0.0,
  "logits": {
    "max_abs": 0.0,
    "mean_abs": 0.0,
    "max_rel": 0.0
  },
  "grad": {
    "max_abs": 0.0,
    "mean_abs": 0.0,
    "max_rel": 0.0
  }
}
```

说明：

- 该验证是 tiny 随机模型，不依赖未公开的 LLaDA2 权重。
- 已完成迁移前旧代码与当前代码的 tiny eager parity；生产级真实权重精度对齐仍需旧 v0.1.2 环境、真实权重、固定数据切片和固定随机种子；当前仓库不包含这些资产。
- 已确认 `inclusionAI/LLaDA2.0-mini-preview` 为非 gated 模型，包含 17 个文件、7 个 safetensors 分片，总权重约 30GB。2026-06-24 在远端 910B2 机器上分别尝试 Hugging Face 反向代理下载和 ModelScope 直连下载；两条链路均可访问，但吞吐不足以在本次工作窗口内完成全量权重获取。当前已保留可断点续传的下载目录和真实权重对齐脚本，拿到完整权重后可直接复跑下面的 harness。
- transformers v5 会触发 `AttentionMaskConverter` deprecation warning，VeOmni logger 在该 warning 上有非阻塞格式化噪声，不影响结果。

真实权重对齐 harness：

```bash
python scripts/download_llada2_assets.py \
  --backend modelscope \
  --repo-id inclusionAI/LLaDA2.0-mini-preview \
  --local-dir /data/t00906153/modelscope_models/LLaDA2.0-mini-preview \
  --max-workers 4
```

```bash
python scripts/check_llada2_assets.py \
  --model-path /data/t00906153/modelscope_models/LLaDA2.0-mini-preview \
  --validate-safetensors
```

```bash
python scripts/create_llada2_alignment_sample.py \
  --model-path /data/t00906153/modelscope_models/LLaDA2.0-mini-preview \
  --output-path /data/t00906153/llada2_alignment_sample.jsonl \
  --max-length 128
```

```bash
python scripts/run_llada2_real_precision_alignment.py \
  --legacy-repo /path/to/dFactory-v0.1.2 \
  --current-repo /path/to/dFactory-current \
  --config-path /path/to/configs/model_configs/llada2_mini \
  --model-path /data/t00906153/modelscope_models/LLaDA2.0-mini-preview \
  --sample-path /data/t00906153/llada2_alignment_sample.jsonl \
  --sample-index 0 \
  --max-seq-len 128 \
  --attn eager \
  --device npu \
  --dtype bfloat16
```

该脚本会在旧代码路径中 monkeypatch 旧版 `fused_moe_forward` 为等价 PyTorch reference MoE，从而在没有旧 CUDA fused kernel 的环境里仍能比较同一真实权重和同一输入的 loss/logits。默认只做 forward 对齐以适配 16B 真实权重；需要梯度对齐时可额外传 `--backward`，但这对 HBM/内存要求显著更高。拿到真实权重和固定样本后，应把该结果作为生产级旧/新精度对齐的准入证据。

## 推荐下一步

1. 准备真实 LLaDA2 mini/flash 权重、tokenizer 与 GSM8K 或内部 SFT 数据，在单卡 NPU 上跑 `max_steps=1`。
2. 使用 `scripts/run_llada2_real_precision_alignment.py` 对相同 batch 固定 seed 跑旧 v0.1.2 与新 eager，记录 loss/logits/grad 差异。
3. 将新 eager baseline 与 `fused_npu` 对齐，阈值建议：loss 相对差 < 1%，关键 logits max_abs/mean_abs 结合 dtype 放宽评估。
4. 扩展到 8 卡 FSDP2，验证 checkpoint save/load 与 HF safetensor 导出。
5. 若性能不足，再投入 LLaDA2 attention/RMSNorm/RoPE 的 NPU OpSlot 化。

## 参考资料

- [VeOmni repository](https://github.com/ByteDance-Seed/VeOmni)
- [VeOmni Ascend NPU get started](https://github.com/ByteDance-Seed/VeOmni/blob/main/docs/hardware_support/get_started_npu.md)
- [VeOmni NPU typical usage](https://github.com/ByteDance-Seed/VeOmni/blob/main/docs/hardware_support/typical_usage.md)
- [VeOmni NPU FAQ](https://github.com/ByteDance-Seed/VeOmni/blob/main/docs/hardware_support/FAQ.md)
- [LLaDA2.0 mini preview weights](https://huggingface.co/inclusionAI/LLaDA2.0-mini-preview)
- [LLaDA2.0 flash preview weights](https://huggingface.co/inclusionAI/LLaDA2.0-flash-preview)
