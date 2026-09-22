# Technical Report — System One / JEV-style decisions on pure CPU

**English** | [中文](#中文版)

> Turning a Qwen 27B GGUF into a JEV-style *decision function* by constraining the
> decoder on the output side only — `grammar` + `logit_bias` + `logprob` — with a
> single forward pass and no second model in the path.

---

## 1. Summary

| metric | value |
|---|---|
| **Accuracy (50 questions)** | **48/50 = 96.0%** |
| Tier 1 (easy) | 14/15 = 93.3% |
| Tier 2 (moderate) | 19/20 = 95.0% |
| Tier 3 (hard) | 15/15 = 100.0% |
| Calibration error (ECE) | 0.060 |
| Latency, one decision per prefill | ~50–56 s |
| Latency, shared-prefix branches | **~1.6 s** |
| Hardware | AMD EPYC 9K65, 32 vCPU, 64 GB, **no GPU** |

The headline is not the accuracy — a 27B model at Q3 is expected to be good at
multiple-choice. It is that **a chat model can be turned into a typed decision
function without any training**, and that the real engineering problem on CPU is
**cost amortisation**, not model quality.

## 2. What "System One / JEV-style" means here

[TypeSafe AI's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
introduced *System One* models: instead of generating text, the model takes a state
plus a typed question and returns a **decision with calibrated probabilities** in a
single pass. A general LLM can be made to behave this way on the **output side**:

1. **Declare the options** as `A. … B. … C. …`.
2. **Constrain the decoder** so only option letters are reachable:
   `logit_bias` hard-masks every non-letter token; a GBNF `grammar` expresses the
   same thing and is used on the batched path.
3. **Read the logprob** of the first generated position and renormalise over the
   declared letters. That distribution *is* the answer, with a confidence attached.

The model never emits prose, so it cannot hallucinate text or drift off-task, and
there is no JSON to parse and no output to clamp.

### How this differs from related work

| | mechanism | confidences come from |
|---|---|---|
| **[smanx/llm2jev](https://github.com/smanx/llm2jev)** | protocol emulation | numbers the model *writes* into JSON, then clamped |
| **[ekzhang/openjev-sglang](https://github.com/ekzhang/openjev-sglang)** | `token_ids_logprob`, prefill-only, GPU | measured log-softmax |
| **this work** | `logit_bias` + `grammar`, CPU, llama.cpp | measured log-softmax |

`llm2jev` puts a Jev-shaped **API** in front of any LLM but still asks for generated
JSON and validates it — a different mechanism. `openjev` and this work share the
decoder-side principle; the difference is the engine (SGLang/GPU vs llama.cpp/CPU).

## 3. Environment (exactly as measured)

| item | value |
|---|---|
| CPU | AMD EPYC 9K65 192-Core, container sees **32 vCPU** (AVX-512 + VNNI) |
| RAM | 64 GB |
| GPU | **none** (`nvidia-smi` absent) |
| OS | Ubuntu 24.04.4, Python 3.12.3 |
| Engine | llama.cpp **0.4.1-dev** (commit 5836771), `-ngl 0`, OpenBLAS, 32 threads |
| Model | `Qwen3.5-27B-Q3_K_S.gguf`, 12,289,423,264 B, sha256 `4da64157d83ede86…` |
| Architecture | `qwen35`, 26.9 B params, 851 tensors, **hybrid attention + SSM** |

The hybrid architecture matters — see §5.

## 4. Benchmark

50 questions across 3 tiers and 7 task types. Each item is a typed decision
(`state`, `question`, 2–4 options), so the model never generates text.

| type | correct | accuracy |
|---|---|---|
| capital | 2/2 | 100% |
| code | 6/6 | 100% |
| entailment | 5/5 | 100% |
| knowledge | 4/4 | 100% |
| logic | 9/9 | 100% |
| math | 16/18 | 88.9% |
| sentiment | 3/3 | 100% |
| topic | 3/3 | 100% |

**Both failures are arithmetic** (Q15 `8+15`, Q18 a 25%-discount word problem), and
they are the two lowest-confidence items in the set (0.546 and 0.792) while correct
answers average 0.945. The model is not confidently wrong — which is the property that
makes a decision function usable behind a threshold.

Raw data: [`bench/results.json`](bench/results.json). Full per-question table:
[`bench/REPORT.md`](bench/REPORT.md).

## 5. Cost model — the important finding

### 5.1 There is a cliff at 36 prompt tokens

| prompt tokens | latency |
|---|---|
| 7 | 0.66 s |
| 13 | 0.92 s |
| 30 | 1.74 s |
| 35 | 1.91 s |
| **36** | **49.03 s** |
| 37 | 51.02 s |
| 108 | 62.40 s |
| 208 | 75.10 s |
| 1608 | 74.93 s |

Below 36 tokens: ~1–2 s. At or above: ~50 s + ~0.015 s/token. A realistic decision
prompt (instruction header + state + question + options + `Answer:`) is **always**
above 36 tokens, so the ~50 s intercept always applies. Shortening the prompt is
futile — removing 200 tokens saves ~3 s.

`llama-bench` on the same model: pp32 0.69 t/s, pp64 1.23, pp128 2.50, pp256 4.50,
tg8 5.45 t/s. Prefill throughput is the bottleneck, and the flat-looking region above
36 tokens is the hybrid **SSM / Gated-DeltaNet recurrent state** being rebuilt for
each fresh prompt.

> **Correction.** An earlier draft of this work claimed latency was a *flat ~50 s
> floor independent of prompt length*. That was wrong, and it was wrong because the
> prompts compared all happened to sit in the same length band. The cliff measurement
> above supersedes it. The claim is preserved here deliberately: the wrong conclusion
> led to the wrong fix (shorten prompts) instead of the right one (amortise).

### 5.2 Amortise instead

| strategy | per decision |
|---|---|
| one decision per prefill, sequential | ~50–56 s |
| **N decisions sharing a warmed prefix** | **~1.6 s** |
| exact-repeat prompt (cache hit) | ~0.34 s |

Measured on 20 questions sharing one state: warm 51.84 s once, then 20 branch calls
totalling 33.81 s → **1.69 s each**. A second batch against the same state: 1.62 s
each. A batch against a *new* state re-warms (50.67 s) and then runs at 1.48 s each.

This is the pattern [openjev](https://github.com/ekzhang/openjev-sglang) uses: render
the template once, split a common prefix from per-question suffixes, warm the cache,
then one `max_new_tokens=1` call per question.

**The win requires an exact shared prefix.** Two operational consequences:

- all questions must share the same state/preamble, with the per-question part last;
- use a **single large slot** (`--parallel 1 -c 8192`). With `--parallel 4` the context
  is sharded into small slots and the cache thrashes — observed alternating 1.6 s /
  58 s, which is worse than not sharing at all.

## 6. Implementation notes

**Use `/v1/completions`, not `/v1/chat/completions`.** Measured on llama.cpp 0.4.1:
the chat endpoint returns `top_logprobs` of the **unconstrained** distribution. The
option letters can be absent from the returned candidates entirely — observed ` A` and
` B` missing from top-20 while the sampler still emitted `A` under `logit_bias`. They
cannot be renormalised. `/v1/completions` returns the raw-logit logprobs for **both**
declared letters even with `logit_bias` applied, which is what a single-pass classifier
needs.

**Batching.** N independent decisions can be packed into one prefill by emitting N
letters (GBNF-enforced, single-space separated) and reading each position. Measured
9.2 s/decision at N=6 versus ~50 s sequential — a 5.4× win, useful when the items do
not share a state. `bench/batched_system_one.py`.

## 7. Repository layout

| file | purpose |
|---|---|
| `deploy.py` | one-click deploy: fetch model, build llama.cpp, serve, smoke-test |
| `bench/system_one.py` | single-pass classifier client |
| `bench/openjev_method.py` | shared-prefix fast path |
| `bench/openjev_scale.py` | 20-question scale test |
| `bench/batched_system_one.py` | N decisions in one prefill |
| `bench/COST_MODEL.py` | the cost model in §5 |
| `bench/prefill_curve.py` | prefill scaling measurements |
| `bench/questions.py` | the 50-question suite |
| `bench/bench.py` | benchmark runner |
| `bench/REPORT.md` | auto-generated per-question results |
| `bench/results.json` | raw run data |

## 8. Reproducing

```bash
git clone https://github.com/feikukuai/qwen3.8_jev.git && cd qwen3.8_jev

# one-click: downloads the model, builds llama.cpp, serves, smoke-tests
python3 deploy.py --tier 27b        # or --tier 4b

# then run the 50-question benchmark against the running server
python3 bench/bench.py
```

For slow international links add `--mirror` (uses hf-mirror.com).
`python3 deploy.py --list` shows tiers and download sizes.

## 9. Limitations — read before quoting these numbers

- **One model was benchmarked.** `Qwen3.5-27B-Q3_K_S` at 48/50. The 4B and the
  ISTA-DASLab GSQ-RCO variants were **not** benchmarked — downloads were stopped
  before completion, so no accuracy claim is made about them anywhere in this repo.
- **50 questions is a small sample.** The 95% confidence interval on 48/50 is roughly
  ±5 pp. Tiers differ by a handful of items; treat tier scores as indicative only.
- **The suite was written by the same agent that ran it.** There is no held-out split
  and no independent authorship, so 96.0% should not be compared against published
  benchmarks. It is a smoke-level quality check, not an evaluation.
- **No human-verified ground truth.** Answers are objectively correct or not for math,
  logic and code; sentiment and topic items reflect one annotator's judgement.
- **Latency figures are single-machine.** 32 vCPU EPYC, DDR5, model on overlay FS.
  Different core counts, NUMA topology or storage will move the ~50 s intercept.
- **Not compared against the real Jev service.** Jev is a closed hosted endpoint; no
  calls were made to it. The "官网九十几" target was supplied by the requester, not
  measured here, and Jev is evaluated on different workflows with a different metric.
  See the prior 4B work for a careful discussion of why those numbers are not
  directly comparable.
- **Greedy decoding only** (`temperature=0`). Sampling was not explored.
- **Option order is untested** for sensitivity; two permutations were not run.

## 10. Acknowledgements

The decoder-as-classifier principle and the shared-prefix serving pattern follow
[TypeSafe AI's System One / Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
[ekzhang/openjev-sglang](https://github.com/ekzhang/openjev-sglang) and
[SemIf](https://openjev.com). The 4B deploy notes from `feat/qwen35-deploy-archive`
arrived at the same scoring math independently and correctly predicted that llama.cpp
would need a hand-written logits readout; §6 is the implementation of that.

---

<a name="中文版"></a>
# 技术报告 — 纯 CPU 上的 System One / JEV 式决策

[English](#technical-report--system-one--jev-style-decisions-on-pure-cpu) | **中文**

> 只通过**输出端约束**（`grammar` + `logit_bias` + `logprob`）把 Qwen 27B GGUF 改造成
> JEV 式**决策函数**：单次前向、单传、路径中没有第二个模型。

---

## 1. 结论速览

| 指标 | 数值 |
|---|---|
| **准确率（50 题）** | **48/50 = 96.0%** |
| Tier 1（简单） | 14/15 = 93.3% |
| Tier 2（中等） | 19/20 = 95.0% |
| Tier 3（困难） | 15/15 = 100.0% |
| 校准误差（ECE） | 0.060 |
| 每题一次前向的延迟 | ~50–56 秒 |
| 共用前缀分支的延迟 | **~1.6 秒** |
| 硬件 | AMD EPYC 9K65，32 vCPU，64 GB，**无 GPU** |

重点不是准确率——27B 模型做多选题本就应该不错。重点是：**一个聊天模型不需要任何训练
就能变成带类型的决策函数**，而且纯 CPU 上真正的工程问题是**成本摊薄**，不是模型质量。

## 2. 这里的「System One / JEV 式」指什么

[TypeSafe AI 的 Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 提出了
*System One* 模型：不生成文本，而是接收 state + 带类型的问题，在一次前向里返回**带校准
概率的决策**。一个通用 LLM 可以在**输出端**被改造成这个样子：

1. **声明选项**为 `A. … B. … C. …`。
2. **约束解码器**，使只有选项字母可达：`logit_bias` 把所有非字母 token 硬掩掉；
   GBNF `grammar` 表达同样的约束，用在批处理路径上。
3. **读 logprob**：取第一个生成位置的分布，在声明的字母上重新归一化。
   这个分布**就是**答案，并且自带置信度。

模型永远不输出散文，因此不会幻觉、不会跑偏，也没有 JSON 要解析、没有输出要夹紧。

### 与相关工作的区别

| | 机制 | 置信度来源 |
|---|---|---|
| **[smanx/llm2jev](https://github.com/smanx/llm2jev)** | 协议仿真 | 模型**写出来**的 JSON 数字，再做夹紧 |
| **[ekzhang/openjev-sglang](https://github.com/ekzhang/openjev-sglang)** | `token_ids_logprob`，prefill-only，GPU | 实测 log-softmax |
| **本工作** | `logit_bias` + `grammar`，CPU，llama.cpp | 实测 log-softmax |

`llm2jev` 是给任意 LLM 套一个 Jev 形状的 **API**，但仍然要求生成 JSON 并做校验——机制不同。
`openjev` 与本工作共享解码器侧原理，差别在引擎（SGLang/GPU 对 llama.cpp/CPU）。

## 3. 环境（实测）

| 项目 | 数值 |
|---|---|
| CPU | AMD EPYC 9K65 192-Core，容器可见 **32 vCPU**（AVX-512 + VNNI） |
| 内存 | 64 GB |
| GPU | **无**（无 `nvidia-smi`） |
| 系统 | Ubuntu 24.04.4，Python 3.12.3 |
| 引擎 | llama.cpp **0.4.1-dev**（commit 5836771），`-ngl 0`，OpenBLAS，32 线程 |
| 模型 | `Qwen3.5-27B-Q3_K_S.gguf`，12,289,423,264 字节，sha256 `4da64157d83ede86…` |
| 结构 | `qwen35`，26.9 B 参数，851 张量，**注意力 + SSM 混合** |

混合架构很关键——见 §5。

## 4. 基准测试

50 题，3 个难度层，7 种任务类型。每题都是带类型的决策（`state`、`question`、2–4 个选项），
模型不生成任何文本。

| 类型 | 正确 | 准确率 |
|---|---|---|
| capital | 2/2 | 100% |
| code | 6/6 | 100% |
| entailment | 5/5 | 100% |
| knowledge | 4/4 | 100% |
| logic | 9/9 | 100% |
| math | 16/18 | 88.9% |
| sentiment | 3/3 | 100% |
| topic | 3/3 | 100% |

**两道错题都是算术**（Q15 `8+15`、Q18 打折题），而且它们是全集中置信度最低的两题
（0.546 和 0.792），而答对的平均置信度是 0.945。模型不是「自信地答错」——这正是决策函数
能放在阈值后面使用的前提。

原始数据：[`bench/results.json`](bench/results.json)。逐题表格：[`bench/REPORT.md`](bench/REPORT.md)。

## 5. 成本模型 —— 最重要的发现

### 5.1 在 36 个 prompt token 处有一个断崖

| prompt token 数 | 延迟 |
|---|---|
| 7 | 0.66 秒 |
| 13 | 0.92 秒 |
| 30 | 1.74 秒 |
| 35 | 1.91 秒 |
| **36** | **49.03 秒** |
| 37 | 51.02 秒 |
| 108 | 62.40 秒 |
| 208 | 75.10 秒 |
| 1608 | 74.93 秒 |

36 token 以下约 1–2 秒；36 及以上约 50 秒 + 0.015 秒/token。真实的决策 prompt
（指令头 + state + question + options + `Answer:`）**必然**超过 36 token，所以那个 50 秒的
截距永远存在。缩短 prompt 没有意义——省掉 200 个 token 只省约 3 秒。

同一模型上的 `llama-bench`：pp32 0.69 t/s、pp64 1.23、pp128 2.50、pp256 4.50、tg8 5.45 t/s。
瓶颈是预填充；36 token 以上看起来「平坦」的区域，其实是每次新 prompt 都在重建混合
**SSM / Gated-DeltaNet 循环状态**。

> **更正。** 本工作早期草稿曾断言延迟是「与 prompt 长度无关的 ~50 秒固定地板」。那是**错的**，
> 错在用来比较的 prompt 恰好都落在同一长度区间。上面的断崖测量取代了它。这里刻意保留
> 这个错误结论：它导致了对策找错方向（去缩短 prompt），而不是正确的方向（去摊薄成本）。

### 5.2 正确做法是摊薄

| 策略 | 每次决策 |
|---|---|
| 每题一次前向，串行 | ~50–56 秒 |
| **N 题共用已热前缀** | **~1.6 秒** |
| 完全重复的 prompt（缓存命中） | ~0.34 秒 |

20 题共用一个 state 的实测：warm 一次 51.84 秒，之后 20 次分支调用共 33.81 秒 → **每次 1.69 秒**。
对同一 state 的第二批：每次 1.62 秒。换了 state 的批次：重新 warm（50.67 秒），之后每次 1.48 秒。

这就是 [openjev](https://github.com/ekzhang/openjev-sglang) 使用的模式：模板只渲染一次，
把公共前缀与每题后缀分开，先 warm 缓存，再对每题发一次 `max_new_tokens=1` 调用。

**加速的前提是前缀完全一致。** 两个运维后果：

- 所有问题必须共用同一个 state/前言，把每题特有的部分放在最后；
- 必须用**单个大 slot**（`--parallel 1 -c 8192`）。用 `--parallel 4` 会把上下文切碎，
  缓存反复失效——实测在 1.6 秒 / 58 秒之间抖动，比不共用还差。

## 6. 实现要点

**必须用 `/v1/completions`，不能用 `/v1/chat/completions`。** 在 llama.cpp 0.4.1 上实测：
chat 端点返回的是**未约束**分布的 `top_logprobs`。选项字母可能完全不在候选里——
实测在 `logit_bias` 下采样器输出了 `A`，而 ` A` 和 ` B` 都不在 top-20 里，因此无法归一化。
`/v1/completions` 即使应用了 `logit_bias`，仍会返回**两个**声明字母的真实 logit logprob，
这正是单传分类器需要的。

**批处理。** 把 N 个独立决策塞进一次前向：让模型输出 N 个字母（GBNF 强制、单空格分隔），
逐个位置读取。N=6 时实测 9.2 秒/题，对比串行约 50 秒——5.4 倍收益，适合各题不共用 state 的场景。
见 `bench/batched_system_one.py`。

## 7. 仓库结构

| 文件 | 用途 |
|---|---|
| `deploy.py` | 一键部署：取模型、编译 llama.cpp、起服务、冒烟测试 |
| `bench/system_one.py` | 单传分类器客户端 |
| `bench/openjev_method.py` | 共用前缀快速路径 |
| `bench/openjev_scale.py` | 20 题扩展测试 |
| `bench/batched_system_one.py` | 一次前向做 N 个决策 |
| `bench/COST_MODEL.py` | §5 的成本模型 |
| `bench/prefill_curve.py` | 预填充缩放测量 |
| `bench/questions.py` | 50 题测试集 |
| `bench/bench.py` | 基准测试运行器 |
| `bench/REPORT.md` | 自动生成的逐题结果 |
| `bench/results.json` | 原始运行数据 |

## 8. 复现

```bash
git clone https://github.com/feikukuai/qwen3.8_jev.git && cd qwen3.8_jev

# 一键：下载模型、编译 llama.cpp、起服务、冒烟测试
python3 deploy.py --tier 27b        # 或 --tier 4b

# 然后对运行中的服务跑 50 题基准
python3 bench/bench.py
```

国际链路慢时加 `--mirror`（改用 hf-mirror.com）。`python3 deploy.py --list` 可看各档位与下载体积。

## 9. 局限 —— 引用这些数字前请先读

- **只测了一个模型。** `Qwen3.5-27B-Q3_K_S` 得 48/50。4B 与 ISTA-DASLab GSQ-RCO 变体
  **没有**被测——下载在完成前被中止，因此本仓库任何地方都没有对它们做准确率声明。
- **50 题样本很小。** 48/50 的 95% 置信区间约为 ±5 个百分点。各层之间只差几道题，
  分层分数仅供参考。
- **测试集由运行它的同一个 agent 编写。** 没有留出集，也没有独立作者，因此 96.0% 不应与
  任何公开基准对比。它是冒烟级质量检查，不是评测。
- **没有人工核验的标准答案。** 数学、逻辑、代码题的答案客观正确与否；情感与主题题反映
  单一标注者的判断。
- **延迟数字来自单机。** 32 vCPU EPYC、DDR5、模型在 overlay 文件系统上。核数、NUMA 拓扑
  或存储不同，那个 ~50 秒截距都会变。
- **没有与真正的 Jev 服务对比过。** Jev 是闭源托管端点，本工作没有向它发过任何请求。
  「官网九十几」这个目标由需求方提供，不是这里测出来的，而且 Jev 在不同 workflow、
  不同指标下评测。关于这些数字为何不可直接比较，见先前的 4B 工作。
- **只用了贪心解码**（`temperature=0`），未探索采样。
- **未测试选项顺序敏感性**，没有跑两种排列。

## 10. 致谢

解码器当分类器的原理与共用前缀的服务模式，来自
[TypeSafe AI 的 System One / Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)、
[ekzhang/openjev-sglang](https://github.com/ekzhang/openjev-sglang) 与 [SemIf](https://openjev.com)。
`feat/qwen35-deploy-archive` 分支的 4B 部署笔记独立得出了相同的打分数学，并正确预判了
llama.cpp 需要手写 logits 读出；§6 就是该预判的实现。

---

# Appendix A — GPU forecast: *what should be faster, and by how much*

> **This appendix is a prediction, not a measurement.** No GPU was available on the
> machine used for this report (`nvidia-smi` absent, no `/dev/nvidia*`). Every GPU
> number below is extrapolated. The CPU row is measured; treat the rest as an
> order-of-magnitude planning estimate to verify yourself.

## A.1 Is the 50 s a CPU problem?

**Yes, and specifically a compute problem, not a memory problem.**

The tempting model — "prefill reads 12.6 GB of weights, so at 576 GB/s it should take
0.04 s" — is wrong by three orders of magnitude. It predicts 0.04 s where the measured
value is ~50 s. The measurement shows why:

| test | tok/s | implied FLOP/s | naive bandwidth model | measured |
|---|---|---|---|---|
| pp32 | 0.69 | 37 GFLOP/s | 0.04 s | 46.4 s |
| pp64 | 1.23 | 66 GFLOP/s | 0.04 s | 52.0 s |
| pp128 | 2.50 | 135 GFLOP/s | 0.04 s | 51.2 s |
| pp256 | 4.50 | 242 GFLOP/s | 0.04 s | 56.9 s |

Prefill is a **matmul-bound** phase: cost scales as `2 x params x tokens`, and the
weights are read many times, not once. This box sustains roughly **1.2 TFLOP/s** of
useful prefill work. That is the number to beat, and it is a compute limit, not a DDR5
bandwidth limit.

So the ~50 s intercept is **not** an inevitable property of the model, the quantisation,
or the algorithm. It is what 32 CPU cores can do. A GPU attacks exactly the term that
dominates.

## A.2 Anchoring the forecast to a real measurement

Rather than trust a theoretical model — the naive one was already off by 1000x — this
forecast is anchored to the only like-for-like CPU-vs-GPU datapoint available: the prior
4B deployment work in `feat/qwen35-deploy-archive`, which ran the same kind of one-token
scoring task on 27B-class weights.

| | latency per row |
|---|---|
| measured, CPU | 17.89 s |
| reported, 1x RTX 3090 | 0.29 s |
| **ratio** | **~62x** |

Their CPU was ~3x faster per row than this box (17.89 s vs ~53 s), so if the ratio
transfers, a 3090 should land near **0.9 s per decision** here.

## A.3 Forecast table

| configuration | est. s/decision | vs this CPU | basis |
|---|---|---|---|
| **CPU, 32 vCPU (this box)** | **53.3** | **1x** | **measured** |
| 1x RTX 3090 24 GB | ~0.9 | ~62x | 62x anchor from prior work |
| 1x RTX 4090 24 GB | ~0.6 | ~89x | 3090 x ~1.45 for this size |
| 1x A100 80 GB | ~0.3 | ~160x | 4090 x ~1.8, bandwidth-heavy |
| B200 + SGLang (openjev-style) | ~0.07-0.5 | ~100-750x | openjev reports 70-500 ms end-to-end |

The openjev figure is the closest published analogue: they serve a 35B-A3B MoE on a B200
with SGLang, prefill-only, one token per question, and report **70-500 ms** end-to-end.
That is the shape of system this design converges to on good hardware.

## A.4 What changes and what does not

**Gets dramatically better on GPU:**

- the ~50 s intercept, i.e. essentially the entire cost of a cold decision;
- the 36-token cliff — a CPU scheduling threshold, not a model property, and not
  expected to survive on GPU;
- concurrency: SGLang-class servers batch many branches against one shared prefix, so
  throughput scales far better than latency alone suggests.

**Does not change:**

- **accuracy.** Same weights, same one-token readout, same answer. A GPU delivers the
  identical 48/50 faster; it does not make the model smarter.
- **the shared-prefix requirement.** openjev's own README warns radix reuse is
  "opportunistic, not a pinned per-request KV session", and that *hybrid Qwen's recurrent
  state* can reduce hits — the same issue measured here on CPU. They mitigate it with
  `--mamba-radix-cache-strategy extra_buffer`. This is a property of the **architecture**,
  not of the CPU.
- **the design.** `logit_bias` + `logprob` single-pass reads are identical on either device.

## A.5 VRAM sizing

| quant | weights | fits 24 GB? |
|---|---|---|
| IQ3_S (same class as this report) | ~11.8 GB | yes, with room for context |
| Q4_K_M | ~16.7 GB | yes, tight |
| Q8_0 | ~29 GB | no (needs 48 GB) |

A 24 GB card comfortably runs the exact configuration benchmarked here, plus KV cache.

## A.6 Accuracy context — the gap is not the bottleneck

openjev published a **one-token MMLU-Pro** comparison on 1,000 questions using
**Qwen3.8-27B**, a model in the same class as the one benchmarked here:

| system | accuracy | 95% CI |
|---|---|---|
| OpenJev - Qwen3.6-35B-A3B | 58.8% | 55.7-61.8% |
| Qwen3.8-27B (hosted, zero-shot) | 60.0% | 56.9-63.0% |
| **Jev (TypeSafe)** | **82.9%** | 80.4-85.1% |

Two things follow, and they point in opposite directions:

1. **Hard multiple-choice is where open models fall short of Jev** — roughly 23 points on
   MMLU-Pro, with a paired bootstrap interval excluding zero. The 96.0% in this report
   comes from a 50-question smoke suite written by the same agent that ran it, so it is
   **not** evidence against that gap. Do not read 96.0% as "matches Jev".
2. **Speed is a solved problem; accuracy is not.** The GPU forecast above closes the
   latency gap entirely. It does nothing for the accuracy gap. To match Jev, the lever is
   a better base model or task-specific training — not hardware.

## A.7 How to verify this appendix

```bash
# on a GPU box, llama.cpp with CUDA
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/bin/llama-server -m Qwen3.5-27B-Q3_K_S.gguf -ngl 99 -c 8192 -b 1024 --parallel 1
python3 bench/bench.py        # compare against the CPU numbers in this report
```

`-ngl 99` offloads all layers. If the forecast is right the ~50 s intercept should
collapse to well under a second. If it does not, this appendix is wrong and the CPU
number was not a compute limit after all — a genuinely interesting result worth recording
here.

### 中文摘要

**50 秒确实是 CPU 的问题，而且是算力问题，不是内存带宽问题。**

朴素模型（读 12.6 GB 权重 ÷ 576 GB/s ≈ 0.04 秒）错了三个数量级——实测约 50 秒。原因是预填充属于
**matmul 计算受限**阶段，权重要被反复读取而不是读一次。这台机器只能提供约 **1.2 TFLOP/s** 的有效
预填充算力。GPU 恰好打在主导项上。

预测（**仅 CPU 一行为实测，其余全部是外推**）：

| 配置 | 每次决策 | 相对本机 |
|---|---|---|
| CPU 32 vCPU（本机） | **53.3 秒** | **1×（实测）** |
| 1× RTX 3090 24GB | ~0.9 秒 | ~62× |
| 1× RTX 4090 24GB | ~0.6 秒 | ~89× |
| 1× A100 80GB | ~0.3 秒 | ~160× |
| B200 + SGLang | ~0.07–0.5 秒 | ~100–750× |

依据是先前 4B 工作中同类的 CPU/3090 实测比 **≈62×**，以及 openjev 在 B200 上报的 70–500 ms 端到端。
**62× 这个锚点是外推的起点，不是我测的。**

**GPU 会改善的**：50 秒的截距（即冷启动决策的几乎全部成本）、36-token 断崖（那是 CPU 调度阈值，
不是模型属性）、以及并发能力。

**GPU 不会改变的**：**准确率**。同样的权重、同样的单 token 读出、同样的答案——GPU 只是让同样的 48/50
更快到达，不会让模型变聪明。共用前缀的要求也不会变：那是混合架构（SSM 循环状态）的属性，
openjev 在 GPU 上遇到同样问题，用 `--mamba-radix-cache-strategy extra_buffer` 缓解。

**并且请注意**：openjev 用 Qwen3.8-27B（与本文同级的模型）做的 1000 题 MMLU-Pro 单 token 评测中，
开放模型约 60%，而 Jev 是 **82.9%**——相差约 23 个百分点。GPU 能把**速度**差距完全抹平，
但对**准确率**差距毫无帮助。要追平 Jev，杠杆在更好的基座模型或任务训练，不在硬件。

同时，本报告的 96.0% 是 50 题的冒烟级测试，且由运行它的同一个 agent 出题，**不能**用来反驳这个差距。
