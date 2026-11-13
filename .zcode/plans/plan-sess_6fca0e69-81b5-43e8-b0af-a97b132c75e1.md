# VoiceVeritas — 面试读稿检测开源项目实施计划

## Context（背景与调研结论）

**目标**：按 `projectPlan.md` 实现"AI 面试语音读稿检测"（输入候选人回答音频 → 输出朗读/背诵概率 P∈[0,1]），并以 **GitHub 优秀开源项目**标准交付完整工程。

**调研核心结论**（已完成的领域研究）：
- **SOTA 方法**：SLT 2024 论文 *Classification of Spontaneous and Scripted Speech*（arXiv:2412.11896）—— 预训练 Transformer（Whisper/wav2vec2 类）编码器嵌入 + 轻量分类头 + 30s 分段打分中值聚合，显著优于手工特征（eGeMAPS+语速+VAD统计，115维）；**冻结骨干即可，微调无增益**
- **判别性特征**（文献共识）：静音/语音段时长分布、发音速率、F0 轨迹、填充词密度、说话轮次
- **市场空白**：GitHub 无同类开源项目（面试反读稿），差异化价值真实
- **免费直链数据**：LibriSpeech（朗读，OpenSLR 12）、SBCSAE（自发对话，OpenSLR 155）、CORAAL（访谈）、AMI（会议）
- **本机约束**：CPU-only（Ryzen 7 8745HS/32GB）→ 双轨方案：LightGBM 基线全程 CPU + 冻结骨干嵌入预计算（缓存到磁盘）后训练轻头

**用户已确认**：英文数据优先（架构多语言就绪，中文列路线图）；Apache-2.0；全流程含真实数据基准。

## 技术方案

### 双轨模型架构
| 轨道 | 方法 | 定位 |
|---|---|---|
| **A 基线** | 能量 VAD（自适应阈值，纯 numpy）+ 韵律特征（parselmouth：F0统计/Jitter/Shimmer/PVI/静音分布/语速方差，~50维）→ LightGBM + isotonic 概率校准 | 可解释、快、CPU 友好，输出特征贡献解释 |
| **B 深度** | 冻结 wav2vec2-base（HF）→ 分段嵌入 mean+std 池化 → **磁盘缓存 npz** → MLP头（BCE/Focal可选） | 性能上限，论文 SOTA 路线 |

- 分段评分（segmented_scores）+ 中值聚合；说话人**不相交**切分防泄漏
- 指标：AUC-ROC / AUC-PR / EER / **FPR@TPR95**（防误杀优秀候选人）+ 校准曲线
- 核心 audio-only；`reference_text`/强制对齐设计为可选 `[asr]` extra 扩展钩子（faster-whisper）

### 仓库结构（src 布局，uv + Python 3.11）
```
VoiceVeritas/
├── pyproject.toml          # 包 voiceveritas，CLI 入口 vv，extras: [api,asr,dev]
├── README.md               # 双语 + 徽章 + mermaid 架构图 + 快速开始
├── LICENSE(Apache-2.0) CONTRIBUTING.md CODE_OF_CONDUCT.md SECURITY.md CITATION.cff
├── .github/workflows/ci.yml  # ruff+mypy+pytest 矩阵 3.10-3.12；issue/PR 模板
├── .pre-commit-config.yaml
├── src/voiceveritas/
│   ├── config.py  audio/  vad/  features/  manifest/  augment/
│   ├── models/    train/  metrics/  infer/  api/
├── scripts/  (prepare_librispeech.py, prepare_sbcsae.py, bench_api.py, make_demo.py)
├── configs/  (baseline.yaml, deep.yaml)
├── tests/    (合成音频夹具：已知F0正弦、语音样脉冲 → 验证特征正确性)
├── docs/     (usage.md, data.md, benchmark.md, model_card.md, ethics.md 伦理章程)
└── data/ runs/ models/     # gitignore，不入库
```

### 工程质量标准（GitHub 优秀项目）
- **CI**：Actions 矩阵跑 ruff check/format + mypy + pytest（合成夹具，无网络无 GPU）
- **测试**：合成信号验证特征提取正确性（440Hz 正弦→F0≈440）、VAD、manifest 往返、训练冒烟、API TestClient
- **伦理章程**（此类双用途工具的关键）：人机协同定位（辅助而非自动拒人）、FPR 披露、禁止无告知监控/纯自动化雇佣决策、非母语者偏差风险文档化 —— 这是优秀开源项目的差异化标配
- 数据集许可证文档化（CORAAL 为 NC 需标注）；音频/模型永不入库

## 实施里程碑

| # | 内容 | 验证标准 |
|---|---|---|
| M0 | git init + 骨架 + CI + 社区文件 + README v0 | CI 配置就绪 |
| M1 | 音频 IO（wav/mp3/m4a→16k mono）+ manifest JSONL + 合成夹具 + 切分（说话人不相交） | pytest 绿 |
| M2 | VAD + 韵律特征提取器 | 合成信号特征正确性测试通过 |
| M3 | 训练/评测管线 + 指标模块 + 校准 + runs/ 实验记录 | 合成数据冒烟训练通过 |
| M4 | **真实基准**：下载 LibriSpeech dev-clean(337MB) + SBCSAE(~1-2GB) → 特征提取 → LightGBM 基线 | 真实 AUC 写入 docs/benchmark.md |
| M5 | 深度轨道：wav2vec2-base 冻结嵌入预计算（后台跑）+ MLP 头训练对比 | 与基线对比表 |
| M6 | CLI `vv predict`（含分段评分+解释）+ FastAPI 服务（/v1/predict, /v1/health, 上传限制, OpenAPI）+ Dockerfile | API 集成测试 + 延迟基准（10s音频≤500ms） |
| M7 | model card + benchmark 完善 + README 终版 + 语法检查 + tag v0.1.0 | 全量测试绿 + 演示可复现 |

## 验证方式
1. `uv run pytest` 全绿（每次里程碑）
2. M4/M5 真实基准：`vv prepare` → `vv train` → `vv eval` 一键复现，指标入 docs
3. M6：`vv predict demo.wav` 输出概率+分段评分；`uvicorn` 起服务后 TestClient + httpx 压测脚本
4. 最终 ruff/mypy 干净，提交历史按里程碑组织（conventional commits）