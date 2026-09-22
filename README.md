# qwen3.8_jev

Turn a **Qwen 27B GGUF into a System One / JEV-style decision model** by constraining the
decoder on the *output side* only — grammar + logit_bias + logprob readout — with a
**single forward pass** and **no second model** in the path.

> `logit_bias` `logprob` `qwen3.8`

---

## The idea in one paragraph

A chat LLM wastes its decoder on text you then have to parse. A System One model instead
takes a *state* plus a *typed question* and returns a **decision with calibrated
probabilities** in a single pass. You do not need to retrain anything to get most of this
behaviour: declare the options as `A. ... B. ... C. ...`, then

* **grammar (GBNF)** — make only option letters reachable,
* **logit_bias** — hard-mask every token that is not a declared option letter,
* **logprob** — renormalise the first-token distribution over those letters.

That distribution *is* the answer, with a confidence attached. The model never writes
text, so it cannot hallucinate prose or go off the rails.

## Result on this box (pure CPU, no GPU)

| | value |
|---|---|
| CPU | AMD EPYC 9K65 (AVX-512 + VNNI), 32 vCPU / 64 GB, **no GPU** |
| Engine | llama.cpp 0.4.1-dev, `-ngl 0`, OpenBLAS, 32 threads |
| Model | `Qwen3.5-27B-Q3_K_S.gguf` (12.29 GB, 26.9 B params, hybrid attention + SSM) |

**50-question benchmark — 100% (50/50)** across sentiment / topic / math / logic /
entailment / code / world-knowledge, tiers 1-3, with calibrated confidence
(wrong answers were not confident; see `bench/REPORT.md`).

Speed measured with `llama-bench`:

| test | t/s |
|---|---|
| pp32 | 0.69 |
| pp64 | 1.23 |
| pp128 | 2.50 |
| pp256 | 4.50 |
| tg8 | 5.45 |

### The important finding: latency is a ~fixed floor

| prompt length | decision latency |
|---|---|
| 36 tok | 52.2 s |
| 126 tok | 53.2 s |
| 426 tok | 58.7 s |
| 1026 tok | 67.9 s |

Cost is **almost independent of prompt length** (~50 s fixed + ~0.017 s/token). It is
dominated by rebuilding the hybrid **SSM / Gated-DeltaNet recurrent state** on every fresh
prompt — which is also why the KV prefix cache does *not* help across different states.

**Consequence:** one decision per pass is unusable interactively. Two ways to fix it:

| strategy | per decision |
|---|---|
| sequential, 1 decision per prefill | **~50 s** |
| **grammar-batched, N=6 per prefill** | **~9.2 s** |
| exact-repeat (KV cache hit) | **~0.35 s** |

Batching N independent decisions into one constrained pass buys a **5.4x speedup**, because
the fixed recurrent-state cost is paid once and amortised over all N.

## Files

| file | purpose |
|---|---|
| `bench/system_one.py` | single-pass classifier client (logit_bias + grammar + logprob) |
| `bench/batched_system_one.py` | N decisions in one prefill — the fast path |
| `bench/questions.py` | the 50-question suite (3 tiers, 7 task types) |
| `bench/bench.py` | benchmark runner, writes `results.json` |
| `bench/REPORT.md` | full results, latency analysis, limitations |

## Gotcha that mattered most

`/v1/chat/completions` returns `top_logprobs` of the **unconstrained** distribution, so the
option letters may be absent from the returned candidates and **cannot be renormalised**
(measured: ` A` / ` B` missing entirely from top-20 while the sampler still emitted `A`).

`/v1/completions` returns the raw-logit logprobs for **both** declared letters even with
`logit_bias` applied. Decisions must go through the completions endpoint.

## Reproduce

```bash
# model
aria2c -x 16 -s 16 -o Qwen3.5-27B-Q3_K_S.gguf \
  https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/resolve/main/Qwen3.5-27B-Q3_K_S.gguf

# engine (OpenBLAS required for the BLAS backend)
apt-get install -y cmake libopenblas-dev
git clone --depth 1 https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_NATIVE=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 32

# serve
./build/bin/llama-server -m Qwen3.5-27B-Q3_K_S.gguf --host 127.0.0.1 --port 8080 \
  -t 32 -c 4096 -b 1024 -ub 1024 --parallel 4 -ngl 0 --no-warmup

# benchmark
python3 bench/bench.py
```

## Related prior art

[TypeSafe AI's Jev / System One](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
is the concept this reproduces on an open base model. Independent from-scratch
reproductions worth reading:
[chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision](https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-GGUF)
(LoRA trained, calibrated, 1 token per decision) and
[smanx/llm2jev](https://github.com/smanx/llm2jev) — a *different* approach: it wraps an LLM
behind the Jev **API contract**, asking for JSON text and parsing it. That is protocol
emulation, not decoder-side classification.
