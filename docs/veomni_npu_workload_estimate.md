# dFactory 兼容最新 VeOmni 与 Ascend NPU 工作量评估

日期：2026-06-25

## 估算口径

- 本报告只做工作量评估，不展开代码变更、验证日志或算子细节。
- 所有时间均取上限，以人日计。
- 不计入小规模验证项：单卡 NPU 功能验证、旧代码 tiny parity、真实权重小步 SFT。
- 交付目标直接按大规模训练准入口径估算：多卡 NPU、真实权重、真实数据、生产级 checkpoint、SP/EP 并行能力和精度/性能验收。

## 工作量评估

| 工作项 | 交付内容 | 上限估算 |
| --- | --- | ---: |
| 基础 API 迁移与模型接入收口 | VeOmni 子模块升级、模型注册、训练参数 schema、dataloader、optimizer、checkpoint API、启动脚本迁移 | 5 人日 |
| 大规模多卡 FSDP2 训练闭环 | 8 卡 NPU FSDP2 训练、真实 batch、checkpoint save/load、HF safetensor 导出、故障恢复验证 | 7 人日 |
| SP 迁移 | 接入并验证 sequence parallel 相关切分、attention/RoPE/mask 兼容、block diffusion 4D mask 在 SP 下的行为、显存与通信边界 | 7 人日 |
| EP 迁移 | 接入并验证 MoE expert parallel routing、expert weight 分片/聚合、fused_npu MoE dispatch、checkpoint 分片保存/恢复和 EP/FSDP2 组合行为 | 8 人日 |
| 生产级精度对齐 | 旧 VeOmni v0.1.2 baseline vs 最新 VeOmni，固定真实权重、真实数据切片、seed、loss 曲线、关键 logits 和必要梯度对齐 | 8 人日 |
| 大规模性能优化 | NPU profiling、MoE/attention/RMSNorm/RoPE 热点定位、batch/sequence 并行策略调参、通信与 HBM 利用率优化 | 10 人日 |
| 文档、Runbook 与准入 CI | NPU 部署文档、真实权重资产检查、对齐脚本、故障排查手册、最小 CI/手工准入矩阵 | 3 人日 |

## 汇总

| 口径 | 上限估算 |
| --- | ---: |
| 总工作量 | 48 人日 |
| 单人串行排期 | 10 周 |
| 2 人并行排期 | 5 周 |
| 3 人并行排期 | 4 周 |

## 说明

- SP 与 EP 分别单独计入，未折叠进多卡 FSDP2 基础闭环。
- 以上估算默认已有可用 Ascend 910B2 多卡环境、完整 LLaDA2 权重、真实训练数据和旧版本 baseline 环境。
- 若 NPU 环境、数据访问、权重下载或远端网络不稳定，需要额外预留环境排障时间。
