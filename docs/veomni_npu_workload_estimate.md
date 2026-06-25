# dFactory / VeOmni LLaDA 系列 NPU 支持工作量评估

日期：2026-06-25

## 估算口径

- 本报告只做工作量与路线评估，不展开验证日志。
- 时间以区间估算，单位为人日；区间上界用于排期预留，下界用于理想条件下的最小投入判断。
- 不计入小规模验证项：单卡 NPU 功能验证、旧代码 tiny parity、真实权重小步 SFT。
- 交付目标直接按大规模训练准入口径估算：多卡 NPU、真实权重、真实数据、生产级 checkpoint、SP/EP 并行能力和精度/性能验收。
- 评估分两条路线：继续维护 dFactory 适配最新 VeOmni，以及引导客户使用 VeOmni 原仓并在原仓原生支持 LLaDA 系列模型。

## dFactory 当前使用的 VeOmni 能力

dFactory 不是只把 VeOmni 当子模块放在仓库里，而是直接依赖了 VeOmni 的训练基础设施。主要使用面如下：

| VeOmni 能力 | dFactory 使用位置 | 说明 |
| --- | --- | --- |
| 模型注册与加载 | `models/llada2_moe/__init__.py`，`tasks/train_llada2_common.py` | 通过 `MODEL_CONFIG_REGISTRY` / `MODELING_REGISTRY` 注册 `llada2_moe_veomni`，再由 `build_foundation_model` 统一加载模型和 ops config |
| 参数体系 | `tasks/train_llada2_common.py` | 继承 `ModelArguments` / `DataArguments` / `TrainingArguments` / `VeOmniArguments`，使用 `parse_args`、`save_args` |
| 数据集与 dataloader | `tasks/dataset/dataset.py`，`tasks/train_llada2_common.py` | 使用 `IterativeDataset`、`MappingDataset`、`build_dataset`、`build_dataloader`，dFactory 只补 LLaDA 的 MDM transform |
| 并行状态与 FSDP2 | `tasks/train_llada2_common.py`，`models/llada2_moe/modeling_llada2_moe.py` | 使用 `init_parallel_state`、`get_parallel_state`、`build_parallelize_model`，并在 MoE eager 路径显式判断 EP 状态 |
| 优化器与调度器 | `tasks/train_llada2_common.py` | 使用 `build_optimizer`、`build_lr_scheduler` 和 VeOmni 的梯度裁剪 |
| checkpoint 与 HF safetensor 保存 | `tasks/train_llada2_common.py` | 使用 `build_checkpointer`、`save_model_assets`、`save_hf_safetensor` |
| NPU/GPU ops dispatch | `models/llada2_moe/modeling_llada2_moe.py` | 使用 `fused_moe_forward`、`OpsImplementationConfig`，按 `fused_triton` / `fused_npu` 选择后端 |
| 设备与分布式工具 | `tasks/train_llada2_common.py` | 使用 `get_device_type`、`get_dist_comm_backend`、`get_torch_device`、`synchronize`、`all_reduce` |

结论：如果继续以 dFactory 为训练入口，迁移到最新 VeOmni 是有必要的。原因不是“追新”，而是 dFactory 的训练、分布式、checkpoint、ops dispatch 都绑定在 VeOmni API 上；停留在 8 个月前版本会继续积累 NPU、FSDP2、ops schema、transformers v5 和 checkpoint 兼容债。

但从客户导向看，更推荐把 LLaDA 系列支持沉到 VeOmni 原仓，而不是长期让客户使用 dFactory fork。这样可以减少客户认知成本、减少 dFactory 与 VeOmni API 漂移带来的二次维护，并复用 VeOmni 原仓已有的 NPU、SP、EP、checkpoint、trainer 生态。

## 权重转换问题

dFactory 当前的 LLaDA2 训练链路要求用户先把 Hugging Face 标准 separate-expert 权重转换为 merged-expert 权重，再在训练完成后 split 回标准 MoE 权重。这个转换由 `scripts/moe_convertor.py` 完成：

- merge：`model.layers.{layer}.mlp.experts.{expert}.{gate/up/down}_proj.weight` 堆叠成 `model.layers.{layer}.mlp.experts.{gate/up/down}_proj`
- split：将堆叠后的 expert tensor 再拆回每个 expert 的独立权重
- README 还要求训练后手动复制原始 `modeling_llada2_moe.py` 到导出目录

这套设计是为了让训练时的 expert 参数形状匹配高效 batched/grouped MoE GEMM，但它把权重形态转换暴露给了用户。若引导客户使用 VeOmni 原仓，建议把这部分能力内置到 VeOmni 的模型加载/保存链路中：

- 加载时支持 HF separate-expert 权重到 VeOmni grouped expert 参数的自动映射。
- 保存时支持 VeOmni grouped expert 参数自动导出为 HF separate-expert safetensors。
- 保留可选离线转换工具，但不再要求客户训练前后手动 merge/split。
- 将模型代码、config、tokenizer、权重导出收敛到 VeOmni 标准 `build_foundation_model` / `save_hf_safetensor` 语义。

这会把“权重转换”从客户操作步骤变成框架内部兼容能力，是引导客户使用 VeOmni 原仓的关键收益。

## 路线 A：继续维护 dFactory 适配最新 VeOmni

| 工作项 | 交付内容 | 估算 |
| --- | --- | ---: |
| 基础 API 迁移与模型接入收口 | VeOmni 子模块升级、模型注册、训练参数 schema、dataloader、optimizer、checkpoint API、启动脚本迁移 | 3-5 人日 |
| 大规模多卡 FSDP2 训练闭环 | 8 卡 NPU FSDP2 训练、真实 batch、checkpoint save/load、HF safetensor 导出、故障恢复验证 | 4-7 人日 |
| SP 迁移 | 接入并验证 sequence parallel 相关切分、attention/RoPE/mask 兼容、block diffusion 4D mask 在 SP 下的行为、显存与通信边界 | 5-7 人日 |
| EP 迁移 | 接入并验证 MoE expert parallel routing、expert weight 分片/聚合、fused_npu MoE dispatch、checkpoint 分片保存/恢复和 EP/FSDP2 组合行为 | 6-8 人日 |
| 生产级精度对齐 | 旧 VeOmni v0.1.2 baseline vs 最新 VeOmni，固定真实权重、真实数据切片、seed、loss 曲线、关键 logits 和必要梯度对齐 | 5-8 人日 |
| 大规模性能优化 | NPU profiling、MoE/attention/RMSNorm/RoPE 热点定位、batch/sequence 并行策略调参、通信与 HBM 利用率优化 | 6-10 人日 |
| 权重转换链路生产化 | merge/split 工具流式化、断点/校验、导出后可直接推理、减少手工复制 modeling 文件 | 3-5 人日 |
| 文档、Runbook 与准入 CI | NPU 部署文档、真实权重资产检查、对齐脚本、故障排查手册、最小 CI/手工准入矩阵 | 2-3 人日 |

### 路线 A 汇总

| 口径 | 估算 |
| --- | ---: |
| 总工作量 | 34-53 人日 |
| 单人串行排期 | 7-11 周 |
| 2 人并行排期 | 4-6 周 |
| 3 人并行排期 | 3-4 周 |

路线 A 的优点是改动局部、能最大化复用 dFactory 已有 LLaDA2 训练逻辑；缺点是客户仍面对 dFactory fork、额外权重转换、dFactory/VeOmni 双仓维护和未来 API 漂移。

## 路线 B：在 VeOmni 原仓原生支持 LLaDA 系列

该路线面向客户使用方式：客户直接 clone/安装 VeOmni 原仓，选择 LLaDA 系列配置即可训练，不需要理解 dFactory fork，也不需要手工做权重 merge/split。

| 工作项 | 交付内容 | 估算 |
| --- | --- | ---: |
| LLaDA 模型上游接入 | 将 LLaDA2 mini/flash 的 config、modeling、model registry 接入 VeOmni 标准目录和 loader，按原仓风格拆分 GPU/NPU patch 与模型资产 | 6-9 人日 |
| MDM / block diffusion 训练任务接入 | 将 dFactory 的 MDM SFT transform、block diffusion mask、loss 计算、confidence/consistency loss 接入 VeOmni trainer 或新增 LLaDA trainer | 7-10 人日 |
| 权重加载自动映射 | 支持 HF separate-expert safetensors 直接加载到 VeOmni grouped expert 参数，避免训练前手工 merge，并处理分片流式加载、strict key 检查和 dtype/device 策略 | 6-9 人日 |
| 权重导出自动反映射 | 支持训练后 grouped expert 参数直接导出为 HF separate-expert safetensors，避免训练后手工 split 和复制 modeling 文件，并保证推理侧可直接 `from_pretrained` | 6-9 人日 |
| NPU ops 与配置接入 | 在 VeOmni 原仓提供 LLaDA GPU/NPU ops preset，接入 `fused_npu` MoE、NPU RMSNorm/RoPE/CrossEntropy fallback 与禁用 GPU-only Liger | 5-8 人日 |
| 大规模 FSDP2/SP/EP 验证 | 8 卡 NPU 真实权重真实数据，验证 FSDP2、SP、EP、checkpoint、恢复训练、HF 导出 | 12-16 人日 |
| 生产级精度与性能验收 | 与 HF / dFactory baseline 对齐 loss/logits/必要梯度，补 NPU profiling 和瓶颈调优 | 8-12 人日 |
| 原仓文档、示例和 CI | LLaDA mini/flash YAML、下载/资产检查、训练/导出/推理 runbook、CI 或手工准入矩阵 | 4-6 人日 |
| 原仓 review 与维护收口 | 按 VeOmni 原仓代码规范拆 PR、补测试资产、处理 review、避免破坏现有模型矩阵和 checkpoint 语义 | 5-8 人日 |

### 路线 B 汇总

| 口径 | 估算 |
| --- | ---: |
| 总工作量 | 59-87 人日 |
| 单人串行排期 | 12-18 周 |
| 2 人并行排期 | 7-10 周 |
| 3 人并行排期 | 5-7 周 |

路线 B 明显重于路线 A。原因是它不是把 dFactory 代码搬到新 VeOmni 上，而是把 LLaDA 作为 VeOmni 原仓的一等模型能力产品化：需要原仓风格的模型接入、trainer 接入、权重自动映射、导出语义、文档/CI、review 和对现有模型矩阵的回归保护。投入更高，但沉淀到 VeOmni 原仓后，客户后续使用和维护成本更低。尤其是权重转换问题，路线 B 可以把 dFactory 暴露给用户的 merge/split 操作内置为 VeOmni loader/exporter 的自动映射能力。

## 推荐路线

| 目标 | 推荐 |
| --- | --- |
| 短期让现有 dFactory 训练继续可用 | 路线 A |
| 面向客户推广、降低使用门槛、减少 fork 维护 | 路线 B |
| 需要 SP/EP/NPU 持续演进 | 路线 B |
| 只做一次性验证或内部 PoC | 路线 A |

推荐对客户主推路线 B：在 VeOmni 原仓支持 LLaDA 系列。dFactory 迁移可以作为短期兼容与验证基线，但不建议作为长期对外主路径。

## 说明

- SP 与 EP 分别单独计入，未折叠进多卡 FSDP2 基础闭环。
- 以上估算默认已有可用 Ascend 910B2 多卡环境、完整 LLaDA2 权重、真实训练数据和旧版本 baseline 环境。
- 若 NPU 环境、数据访问、权重下载或远端网络不稳定，需要额外预留环境排障时间。
- 当前 VeOmni 子模块内未检索到 LLaDA/LLaDA2 原生模型支持，因此路线 B 需要新增模型与任务接入，而不是简单打开已有配置。
