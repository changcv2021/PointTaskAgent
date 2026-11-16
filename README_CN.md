# PointTaskAgent — coding experiment release preparation

这是从实际运行的 **GPT-6.1 Sol / high、100 个案例、每例最多 5 次代码尝试**
实验中整理的独立代码目录。原实验保留不变，公开代码副本由本目录整理生成。

核心流程是：读取冻结的跨任务候选与真实点云 → 生成 Python 诊断代码 →
在隔离的持久进程中执行 → 返回测量、真实点云视图与错误 → 修改代码 →
选择已有候选或 KEEP → 单独使用 GT 离线评测。

| 目录 | 内容 |
|---|---|
| `src/` | coding agent、沙箱、场景/诊断接口、渲染、四类候选工具原语、离线评测 |
| `prompts/` | 实验实际使用的 agent / planner / detection / segmentation 提示词 |
| `configs/` | 可填写的运行配置、类别映射、100 场景冻结样本清单 |
| `scripts/` | 初始化、依赖检查、原实验用过的中断恢复脚本 |
| `reproduction/backbones/` | LION-Mamba / LitePT-S / MCTrack 配置、依赖锁及源码提交号 |
| `reproduction/upstream/` | 候选、历史窗口与缓存准备的来源代码，供整理上游复现链 |
| `reproduction/results/` | 全部 100 例统计，包含有益、有害、KEEP，而非只保留成功样例 |
| `examples/` | 7 个有益且选中最优候选的案例：逐步代码、诊断记录、决策及离线结果 |
| `docs/` | 复现说明、数据接口、保留/排除清单、边界与已知限制 |
| `third_party/` | SeeGround 摘取函数的许可证和归属说明 |

从 [中文整理说明](docs/RELEASE_SCOPE.md) 和 [运行步骤](docs/REPRODUCTION.md) 开始。
逐例过程在 [七例文档](examples/SEVEN_OPTIMAL_CASES_STEP_BY_STEP.md)。
文件来源及改动记录见 `SOURCE_MANIFEST.json`；校验结果见 `validation/`。

**复现范围：**核心入口面向已准备好的 `observations.json/npz`。
本次没有重新运行基线模型、100 例付费实验或全量 nuScenes 评测。
上游模型权重、nuScenes 数据和准备好的大数组均未复制。上游历史代码包含
明确的路径占位符，需要按文档接通，不能把这个目录描述成已验证的一键全链路复现包。

API 密钥仅从环境变量读取。默认配置禁止付费调用。`api.md`、环境目录、
模型权重、原始点云和运行缓存不在包内。
