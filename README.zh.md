# qwen3.8_jev

[English](README.md) | **中文**

只通过**输出端约束**（`grammar` + `logit_bias` + `logprob`）把 Qwen 27B GGUF 改造成
**System One / JEV 式决策函数**：单次前向、单传、路径中**没有第二个模型**。

> `logit_bias` `logprob` `qwen3.8`

**纯 CPU（无 GPU）实测：48/50 = 96.0%**，ECE 0.060 — [完整技术报告](TECHNICAL_REPORT.md) · [验证页面](https://feikukuai.github.io/qwen3.8_jev/verify/)

---

## 快速开始 —— 一条命令

```bash
git clone https://github.com/feikukuai/qwen3.8_jev.git && cd qwen3.8_jev
python3 deploy.py                 # 下载模型、编译 llama.cpp、起服务、冒烟测试
```

这就是全部步骤。`deploy.py` 会自动下载模型、用本机 CPU 指令集编译 llama.cpp
（如果装了 OpenBLAS 就一起启用）、用正确的参数起服务，并跑一次冒烟决策证明能work。

```bash
python3 deploy.py --list          # 查看各档位与体积
python3 deploy.py --tier 4b       # 小档位（2.7 GB，笔记本可跑）
python3 deploy.py --tier 27b-gsq  # ISTA-DASLab GSQ-RCO IQ3_S
python3 deploy.py --mirror        # 国内/慢链路走 hf-mirror.com
```

跑基准：

```bash
python3 bench/bench.py            # 50 题测试集
```

## 核心思路

聊天模型把解码器浪费在你还要再解析的文本上。System One 模型接收 *state* + 带类型的问题，
在一次前向里返回**带校准概率的决策**。**不需要任何训练**：

* **grammar (GBNF)** —— 让只有选项字母可达；
* **logit_bias** —— 把所有非声明字母的 token 硬掩掉；
* **logprob** —— 在声明的字母上重新归一化第一个 token 的分布。

这个分布**就是**答案，并且自带置信度。模型永远不输出文本，所以不会幻觉、不会跑偏，
也没有 JSON 要解析。

## 模型

| 档位 | 模型 | 体积 | 下载 |
|---|---|---|---|
| `27b` | [Qwen3.5-27B-Q3_K_S](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF) | 12.29 GB | [直链](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/resolve/main/Qwen3.5-27B-Q3_K_S.gguf) · [镜像](https://hf-mirror.com/unsloth/Qwen3.5-27B-GGUF/resolve/main/Qwen3.5-27B-Q3_K_S.gguf) |
| `27b-gsq` | [Qwen3.8-27B-GSQ-RCO-IQ3_S](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) | 11.77 GB | [直链](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/resolve/main/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf) · [镜像](https://hf-mirror.com/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/resolve/main/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf) |
| `4b` | [Qwen3.5-4B-Q4_K_M](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) | 2.74 GB | [直链](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf) · [镜像](https://hf-mirror.com/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf) |

`deploy.py` 会自动下载。**只有 `27b` 档位做过基准测试**（48/50）；其余档位是提供的部署选项，
不是准确率声明——见[局限说明](TECHNICAL_REPORT.md#9-局限--引用这些数字前请先读)。

## 实测性能

| | 数值 |
|---|---|
| 准确率 | **48/50 = 96.0%**（T1 93.3%、T2 95.0%、T3 100%） |
| ECE | 0.060 |
| 每题一次前向的延迟 | ~50–56 秒 |
| 共用前缀分支的延迟 | **~1.6 秒** |
| 缓存命中 | ~0.34 秒 |

硬件：AMD EPYC 9K65，32 vCPU，64 GB 内存，**无 GPU**；llama.cpp 0.4.1；
`Qwen3.5-27B-Q3_K_S`。

### 成本断崖 —— 跑基准前必读

| prompt token 数 | 延迟 |
|---|---|
| 35 | 1.91 秒 |
| **36** | **49.03 秒** |
| 208 | 75.10 秒 |

在 **36 个 prompt token** 处有一个陡峭的不连续点。真实的决策 prompt 必然超过它，
所以那个 ~50 秒的截距永远存在——而且**缩短 prompt 没有意义**（200 个 token 只值约 3 秒）。
**正确做法是摊薄**：让多个问题共用一个已热的前缀，第一次之后的每次决策只要 ~1.6 秒。

两个硬性前提：

1. 前缀必须**完全一致**（同一个 state，每题特有的部分放在最后）；
2. 必须用**单个大 slot** —— `--parallel 1 -c 8192`。用 `--parallel 4` 会把上下文切碎，
   缓存反复失效（实测在 1.6 秒 / 58 秒之间抖动）。

## 文件

| 文件 | 用途 |
|---|---|
| [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) | 完整双语报告：结果、成本模型、局限 |
| [`deploy.py`](deploy.py) | 一键部署 |
| [`bench/system_one.py`](bench/system_one.py) | 单传分类器客户端 |
| [`bench/openjev_method.py`](bench/openjev_method.py) | 共用前缀快速路径 |
| [`bench/batched_system_one.py`](bench/batched_system_one.py) | 一次前向做 N 个决策 |
| [`bench/COST_MODEL.py`](bench/COST_MODEL.py) | 成本模型 |
| [`bench/questions.py`](bench/questions.py) | 50 题测试集 |
| [`bench/bench.py`](bench/bench.py) | 基准测试运行器 |
| [`bench/results.json`](bench/results.json) | 原始运行数据 |
| [`verify/index.html`](https://feikukuai.github.io/qwen3.8_jev/verify/) | 静态验证页面 |

## 最容易踩的坑

必须用 `/v1/completions`，**不能**用 `/v1/chat/completions`。chat 端点返回的是
**未约束**分布的 `top_logprobs`，选项字母可能完全不在候选里、无法归一化
（实测：在 `logit_bias` 下采样器输出了 `A`，而 ` A` 与 ` B` 都不在 top-20 中）。
`/v1/completions` 即使应用了 `logit_bias`，仍会返回**两个**声明字母的真实 logit logprob。

## 相关工作

[TypeSafe AI 的 Jev / System One](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 是概念来源。
[ekzhang/openjev-sglang](https://github.com/ekzhang/openjev-sglang) 与 [SemIf](https://openjev.com)
在 GPU 上用 `token_ids_logprob` 实现了同一套解码器侧原理。
[smanx/llm2jev](https://github.com/smanx/llm2jev) 走的是**另一条路**：把 LLM 包在 Jev **API 契约**后面、
要求输出 JSON 文本——那是协议仿真，不是解码器侧分类。

## 许可与诚实声明

50 题测试集由运行它的同一个 agent 编写——没有留出集，没有独立作者。请把 96.0% 当作
冒烟级质量检查，**不是**可与公开基准对比的数字。完整局限见
[报告 §9](TECHNICAL_REPORT.md#9-局限--引用这些数字前请先读)。

## 换 GPU 会更快吗？

**会——而且那 50 秒确实是 CPU 的问题。** 这是**预测**：本报告所用机器没有 GPU。
详见[附录 A](TECHNICAL_REPORT.md#appendix-a--gpu-forecast-what-should-be-faster-and-by-how-much)。

| 加速器 | 显存 / 带宽 | 每次决策延迟 | 证据状态 |
|---|---|---:|---|
| **纯 CPU 基线**（AMD EPYC 9K65, 32 vCPU） | DDR5，约 0.58 TB/s | **53.3 秒** | **实测** |
| 1 × NVIDIA RTX 3090 | 24 GB GDDR6X，0.94 TB/s | ~0.9 秒 | 外推 |
| 1 × NVIDIA RTX 4090 | 24 GB GDDR6X，1.01 TB/s | ~0.6 秒 | 外推 |
| 1 × NVIDIA A100 | 80 GB HBM2e，2.04 TB/s | ~0.3 秒 | 外推 |
| NVIDIA B200 + SGLang | 192 GB HBM3e，8.0 TB/s | 0.07–0.5 秒 | 已发表区间 |

倍数一律**相对于本报告定义的纯 CPU 基线**而言，而非相对于任何特定部署：3090 约为 59 倍加速，
A100 约为 163 倍。下表中**只有第一行是实测**，其余为外推，推导方式见下文。

那 50 秒是**算力受限，不是带宽受限**：朴素的「读 12.6 GB 权重一次」模型预测 0.04 秒，错了 1000 倍。
预填充会反复读取权重，这台机器只能提供约 **1.2 TFLOP/s** 的有效算力——GPU 打的正是这一项。

**GPU 不会改变的是准确率。** 同样的权重、同样的单 token 读出、同样的答案。GPU 只是让同样的 48/50
更快到达，不会让模型变聪明。

作为参照：openjev 在 1000 题 MMLU-Pro 单 token 评测里，Qwen3.8-27B 约 60%，而 **Jev 是 82.9%**。
速度是已解决的问题；准确率差距不是靠硬件补的。
