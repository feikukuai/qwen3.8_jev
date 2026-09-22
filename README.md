# qwen3.8_jev

**English** | [中文](README.zh.md)

Turn a Qwen 27B GGUF into a **System One / JEV-style decision function** by constraining
the decoder on the output side only — `grammar` + `logit_bias` + `logprob` — with a
single forward pass and **no second model** in the path.

> `logit_bias` `logprob` `qwen3.8`

**Result on pure CPU (no GPU): 48/50 = 96.0%**, ECE 0.060 — [full technical report](TECHNICAL_REPORT.md) · [verification page](https://feikukuai.github.io/qwen3.8_jev/verify/)

---

## Quick start — one command

```bash
git clone https://github.com/feikukuai/qwen3.8_jev.git && cd qwen3.8_jev
python3 deploy.py                 # downloads model, builds llama.cpp, serves, smoke-tests
```

That is the whole setup. `deploy.py` fetches the model, builds llama.cpp with native
CPU flags (plus OpenBLAS if present), starts the server with the correct flags, and runs
a smoke decision to prove it works.

```bash
python3 deploy.py --list          # show tiers and sizes
python3 deploy.py --tier 4b       # small tier (2.7 GB, runs on a laptop)
python3 deploy.py --tier 27b-gsq  # ISTA-DASLab GSQ-RCO IQ3_S
python3 deploy.py --mirror        # use hf-mirror.com for slow links
```

Then benchmark it:

```bash
python3 bench/bench.py            # the 50-question suite
```

## The idea in one paragraph

A chat LLM wastes its decoder on text you then have to parse. A System One model takes
a *state* plus a *typed question* and returns a **decision with calibrated
probabilities** in a single pass. You do not need to retrain anything:

* **grammar (GBNF)** — make only option letters reachable;
* **logit_bias** — hard-mask every token that is not a declared option letter;
* **logprob** — renormalise the first-token distribution over those letters.

That distribution *is* the answer, with a confidence attached. The model never writes
text, so it cannot hallucinate prose or go off the rails, and there is no JSON to parse.

## Models

| tier | model | size | download |
|---|---|---|---|
| `27b` | [Qwen3.5-27B-Q3_K_S](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF) | 12.29 GB | [link](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/resolve/main/Qwen3.5-27B-Q3_K_S.gguf) · [mirror](https://hf-mirror.com/unsloth/Qwen3.5-27B-GGUF/resolve/main/Qwen3.5-27B-Q3_K_S.gguf) |
| `27b-gsq` | [Qwen3.8-27B-GSQ-RCO-IQ3_S](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) | 11.77 GB | [link](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/resolve/main/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf) · [mirror](https://hf-mirror.com/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/resolve/main/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf) |
| `4b` | [Qwen3.5-4B-Q4_K_M](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) | 2.74 GB | [link](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf) · [mirror](https://hf-mirror.com/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf) |

`deploy.py` downloads these for you. **Only the `27b` tier has been benchmarked** (48/50);
the other tiers are provided as deployment options, not as accuracy claims — see
[Limitations](TECHNICAL_REPORT.md#9-limitations--read-before-quoting-these-numbers).

## Measured performance

| | value |
|---|---|
| Accuracy | **48/50 = 96.0%** (T1 93.3%, T2 95.0%, T3 100%) |
| ECE | 0.060 |
| Latency, one decision per prefill | ~50–56 s |
| Latency, shared-prefix branches | **~1.6 s** |
| Cache hit | ~0.34 s |

Hardware: AMD EPYC 9K65, 32 vCPU, 64 GB RAM, **no GPU**; llama.cpp 0.4.1;
`Qwen3.5-27B-Q3_K_S`.

### The cost cliff — read this before benchmarking

| prompt tokens | latency |
|---|---|
| 35 | 1.91 s |
| **36** | **49.03 s** |
| 208 | 75.10 s |

There is a sharp discontinuity at **36 prompt tokens**. A realistic decision prompt is
always above it, so the ~50 s intercept always applies — and shortening the prompt is
futile (200 tokens ≈ 3 s). **Amortise instead:** share one warmed prefix across many
questions and every decision after the first costs ~1.6 s.

Two operational requirements:

1. an **exact** shared prefix (same state, per-question part last);
2. a **single large slot** — `--parallel 1 -c 8192`. With `--parallel 4` the context is
   sharded and the cache thrashes (observed alternating 1.6 s / 58 s).

## Files

| file | purpose |
|---|---|
| [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) | full bilingual report: results, cost model, limitations |
| [`deploy.py`](deploy.py) | one-click deploy |
| [`bench/system_one.py`](bench/system_one.py) | single-pass classifier client |
| [`bench/openjev_method.py`](bench/openjev_method.py) | shared-prefix fast path |
| [`bench/batched_system_one.py`](bench/batched_system_one.py) | N decisions in one prefill |
| [`bench/COST_MODEL.py`](bench/COST_MODEL.py) | the cost model |
| [`bench/questions.py`](bench/questions.py) | the 50-question suite |
| [`bench/bench.py`](bench/bench.py) | benchmark runner |
| [`bench/results.json`](bench/results.json) | raw run data |
| [`verify/index.html`](https://feikukuai.github.io/qwen3.8_jev/verify/) | static verification page |

## The gotcha that mattered most

Use `/v1/completions`, **not** `/v1/chat/completions`. The chat endpoint returns
`top_logprobs` of the **unconstrained** distribution, so the option letters can be
missing from the returned candidates entirely and cannot be renormalised (observed:
under `logit_bias` the sampler emitted `A` while ` A` and ` B` were absent from top-20).
`/v1/completions` returns the raw-logit logprobs for **both** declared letters even with
`logit_bias` applied.

## Related work

[TypeSafe AI's Jev / System One](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
is the concept. [ekzhang/openjev-sglang](https://github.com/ekzhang/openjev-sglang) and
[SemIf](https://openjev.com) implement the same decoder-side principle on GPU with
`token_ids_logprob`. [smanx/llm2jev](https://github.com/smanx/llm2jev) takes a *different*
route: it wraps an LLM behind the Jev **API contract** and asks for JSON text — protocol
emulation, not decoder-side classification.

## Licence and honesty note

The 50-question suite was written by the same agent that ran it — no held-out split, no
independent authorship. Treat 96.0% as a smoke-level quality check, **not** a benchmark
comparable to published numbers. Full limitations in
[§9 of the report](TECHNICAL_REPORT.md#9-limitations--read-before-quoting-these-numbers).

## Will a GPU be faster?

**Yes — and the 50 s is genuinely a CPU problem.** This is a *prediction*: the machine used
here has no GPU. See [Appendix A](TECHNICAL_REPORT.md#appendix-a--gpu-forecast-what-should-be-faster-and-by-how-much).

| configuration | est. s/decision | vs this CPU |
|---|---|---|
| **CPU, 32 vCPU (this box)** | **53.3** | **1x (measured)** |
| 1x RTX 3090 24 GB | ~0.9 | ~62x |
| 1x RTX 4090 24 GB | ~0.6 | ~89x |
| 1x A100 80 GB | ~0.3 | ~160x |
| B200 + SGLang | ~0.07–0.5 | ~100–750x |

The ~50 s is **compute-bound, not bandwidth-bound**: a naive "read 12.6 GB of weights once"
model predicts 0.04 s, off by 1000x. Prefill re-reads the weights, so this box sustains only
~1.2 TFLOP/s of useful work. That is what a GPU attacks.

**What a GPU does not change: accuracy.** Same weights, same one-token readout, same answer.
A GPU delivers the identical 48/50 faster — it does not make the model smarter.

For scale: openjev's one-token MMLU-Pro on 1,000 questions puts Qwen3.8-27B around 60% and
**Jev at 82.9%**. Speed is a solved problem; the accuracy gap is not closed by hardware.
