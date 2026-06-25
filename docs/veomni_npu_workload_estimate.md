# dFactory / VeOmni LLaDA 系列 NPU 支持工作量评估

日期：2026-06-25

## 一页结论

本报告只做工作量评估。估算口径从“完整生产化交付”调整为“穿刺出可评审的大规模训练原型”：真实权重、真实数据、多卡 NPU、FSDP2/SP/EP 主链路、旧版本精度对齐、基础性能优化。所有工作量统一用人周表达，并按 0.5 人周作为最小估算粒度归整。单卡 NPU 功能验证、旧代码 tiny parity、真实权重小步 SFT 不作为单独工作项。

| 路线 | 定位 | 工作量 | 3 人并行排期 | 判断 |
| --- | --- | ---: | ---: | --- |
| 路线 A：继续维护 dFactory，适配最新 VeOmni + NPU | 内部兼容与短期交付路径 | 6 人周 | 2 周 | 投入较小，能最快拿到可训练原型，但客户仍面对 dFactory fork 和后续双仓维护 |
| 路线 B：在 VeOmni 原仓原生支持 LLaDA 系列 | 客户主推路径 | 8 人周 | 3 周 | 投入高于路线 A，但能把模型、trainer、NPU 配置和权重自动化沉到 VeOmni 原仓，长期维护成本更低 |

建议：短期用路线 A 作为兼容验证和风险收敛基线；对客户主推路线 B。路线 B 比路线 A 更耗时是合理的，因为它不是简单迁移 dFactory，而是把 LLaDA 系列做成 VeOmni 原仓的一等训练能力。

## 路线 A：继续维护 dFactory 适配最新 VeOmni

路线 A 的目标是在 dFactory 仓内完成最新 VeOmni 兼容、NPU 训练入口、多卡训练原型、精度对齐和基础性能优化。它适合已有 dFactory 用户继续训练，也适合作为路线 B 的 baseline。

| 关键工作 | 交付边界 | 估算 |
| --- | --- | ---: |
| 最新 VeOmni API 迁移 | 模型注册、参数解析、dataloader、optimizer、checkpoint、ops config 能在 dFactory 入口跑通 | 1 人周 |
| 多卡 NPU 训练主链路 | 真实权重/真实数据下跑通 8 卡 NPU FSDP2 训练、保存、恢复和 HF safetensor 导出 | 1 人周 |
| SP 最小可用穿刺 | 处理 block diffusion mask、attention/RoPE、sequence 切分与 loss 对齐问题 | 0.5 人周 |
| EP 最小可用穿刺 | 处理 MoE expert routing、fused_npu dispatch、expert 参数分片与 checkpoint 恢复 | 1 人周 |
| 精度对齐 | 固定 seed、真实权重和真实数据切片，对齐旧 dFactory/老 VeOmni baseline 的 loss 曲线和关键 logits | 1 人周 |
| 基础性能优化 | 做 NPU profiling，优先处理 MoE、attention、RMSNorm/RoPE、通信与显存瓶颈 | 1 人周 |
| 权重转换自动化与最小文档 | 将现有 merge/split 流程脚本化、加校验，补最小运行说明；不展开完整产品化 | 0.5 人周 |

路线 A 总计：6 人周。

主要风险：dFactory 仍跟随 VeOmni API 漂移；权重形态、NPU ops、SP/EP 组合验证会继续落在 dFactory fork 内维护。

## 路线 B：在 VeOmni 原仓原生支持 LLaDA 系列

路线 B 的目标是让客户直接使用 VeOmni 原仓训练 LLaDA 系列模型，不需要理解 dFactory fork，也不需要手工维护模型代码和权重转换步骤。该路线更接近客户长期使用方式，因此工作量高于路线 A。

| 关键工作 | 交付边界 | 估算 |
| --- | --- | ---: |
| LLaDA 模型原仓接入 | 将 LLaDA2 mini/flash 的 config、modeling、registry、ops preset 按 VeOmni 原仓风格接入 | 1.5 人周 |
| MDM / block diffusion trainer | 将 MDM SFT transform、block diffusion mask、loss、confidence/consistency loss 接入 VeOmni trainer 或新增 LLaDA trainer | 1.5 人周 |
| 权重加载/导出自动化 | 在 VeOmni loader/exporter 中自动处理 HF separate-expert 与 grouped expert 的映射；保留离线工具即可 | 1 人周 |
| 多卡 FSDP2/SP/EP 穿刺 | 原仓入口下跑通真实权重/真实数据 8 卡 NPU，覆盖 FSDP2、SP、EP、checkpoint 和恢复训练 | 2 人周 |
| 精度对齐与基础性能优化 | 与 dFactory baseline 对齐 loss/logits，并完成 NPU YAML、fused_npu MoE、fallback op 和首轮 profiling 调优 | 1.5 人周 |
| 最小文档与准入收口 | 补配置示例、运行命令、资产检查和必要 review 修改；不按完整 CI 产品化估算 | 0.5 人周 |

路线 B 总计：8 人周。

主要风险：需要在 VeOmni 原仓保持模型、trainer、checkpoint、ops 和现有模型矩阵的兼容性；review 和上游风格收口也会消耗额外时间。

## 推荐推进方式

| 阶段 | 建议 |
| --- | --- |
| 近期交付 | 先做路线 A，快速形成 dFactory + 最新 VeOmni + NPU 的可训练原型，作为精度和性能 baseline |
| 客户主路径 | 同步或随后推进路线 B，把 LLaDA 系列沉到 VeOmni 原仓，减少客户使用 fork 和手工权重转换的成本 |
| 验收重点 | 不看单卡/tiny demo，直接看 8 卡 NPU、真实权重、真实数据、FSDP2/SP/EP、loss 对齐和基础性能指标 |

## 附录 A：dFactory 使用了 VeOmni 的什么能力

dFactory 对 VeOmni 不是轻依赖，而是直接复用了训练基础设施：

| VeOmni 能力 | dFactory 使用方式 |
| --- | --- |
| 模型注册与加载 | 通过 `MODEL_CONFIG_REGISTRY` / `MODELING_REGISTRY` 注册 LLaDA2 MoE，再由 `build_foundation_model` 加载 |
| 参数体系 | 继承 `ModelArguments` / `DataArguments` / `TrainingArguments` / `VeOmniArguments`，使用 VeOmni 的 `parse_args` |
| 数据与 dataloader | 复用 VeOmni dataset/dataloader，dFactory 只补 LLaDA 的 MDM transform |
| 分布式与 FSDP2 | 使用 `init_parallel_state`、`get_parallel_state`、`build_parallelize_model` |
| 优化器、调度器、checkpoint | 使用 VeOmni 的 optimizer、lr scheduler、checkpointer、HF safetensor 保存链路 |
| GPU/NPU ops dispatch | 通过 `OpsImplementationConfig` 和 `fused_moe_forward` 选择 `fused_triton` / `fused_npu` |

因此，如果继续让 dFactory 作为训练入口，迁移到最新 VeOmni 是必要的；否则 NPU、FSDP2、ops schema、transformers v5 和 checkpoint 兼容债会继续累积。

## 附录 B：为什么仍建议引导客户使用 VeOmni 原仓

从客户视角看，长期使用 dFactory fork 会带来三类成本：

| 成本 | 说明 |
| --- | --- |
| 认知成本 | 客户需要知道 dFactory 与 VeOmni 的关系、版本差异和训练入口差异 |
| 维护成本 | VeOmni 原仓继续演进后，dFactory 需要持续追 API、分布式和 ops 变化 |
| 交付成本 | LLaDA 模型、权重转换、NPU 配置、SP/EP 验证分散在 fork 中，不利于复用 |

把 LLaDA 系列沉到 VeOmni 原仓后，客户可以直接使用 VeOmni 的模型、trainer、checkpoint、NPU 和并行生态，后续维护也更集中。

## 附录 C：权重转换为什么只按自动化估算

dFactory 现有 `scripts/moe_convertor.py` 的核心作用是把 HF separate-expert 权重和训练侧 grouped expert 权重互转。这个问题确实重要，但在本次“训练原型穿刺”口径下，不需要把它估成完整产品化项目：

- 路线 A：保留现有脚本，做流程自动化、校验和最小文档即可。
- 路线 B：在 VeOmni loader/exporter 内做自动映射，训练前后不再要求客户手工 merge/split。
- 完整的流式转换、断点续转、跨版本资产治理和 CI 覆盖可以放到后续生产化阶段。

## 附录 D：估算假设

- 已有可用 Ascend 910B2 多卡环境、真实 LLaDA2 权重、真实训练数据和旧版本 baseline 环境。
- 估算包含精度对齐和基础性能优化，但不包含长期性能压榨、完整 CI 矩阵、完整产品文档和长期维护值守。
- 若 NPU 环境、权重下载、数据访问或远端网络不稳定，需要额外预留排障时间。
