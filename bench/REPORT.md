# Pure-CPU System One / JEV-style benchmark - Qwen3.5-27B-Q3_K_S

Engine llama.cpp 0.4.1-dev, `-ngl 0`, OpenBLAS, 32 threads, AMD EPYC 9K65 (AVX-512 + VNNI). No GPU. All decisions are one prefill, one read position.

## Headline

| metric | value |
|---|---|
| **Accuracy** | **48/50 = 96.0%** |
| Tier 1 | 14/15 = 93.3% |
| Tier 2 | 19/20 = 95.0% |
| Tier 3 | 15/15 = 100.0% |
| Latency, mean | 56.25 s |
| Latency, cold prefill | 56.25 s (50 items) |
| Wall time | 2812 s |
| Calibration error (ECE) | 0.060 |

## Accuracy by task type

| type | correct | accuracy |
|---|---|---|
| capital | 2/2 | 100.0% |
| code | 6/6 | 100.0% |
| entailment | 5/5 | 100.0% |
| knowledge | 4/4 | 100.0% |
| logic | 9/9 | 100.0% |
| math | 16/18 | 88.9% |
| sentiment | 3/3 | 100.0% |
| topic | 3/3 | 100.0% |

## Per-question results

| # | tier | type | predicted | gold | ok | confidence | latency |
|---|---|---|---|---|---|---|---|
| 1 | 1 | sentiment | positive | positive | OK | 0.988 | 52.66 s |
| 2 | 1 | topic | Business | Business | OK | 0.975 | 53.45 s |
| 3 | 1 | math | 43 | 43 | OK | 0.676 | 52.31 s |
| 4 | 1 | math | 63 | 63 | OK | 0.996 | 52.19 s |
| 5 | 1 | capital | Paris | Paris | OK | 0.998 | 53.85 s |
| 6 | 1 | sentiment | negative | negative | OK | 0.993 | 52.31 s |
| 7 | 1 | entailment | yes | yes | OK | 0.969 | 53.65 s |
| 8 | 1 | topic | Sports | Sports | OK | 0.997 | 56.57 s |
| 9 | 1 | math | 63 | 63 | OK | 0.799 | 59.63 s |
| 10 | 1 | sentiment | positive | positive | OK | 0.989 | 53.47 s |
| 11 | 1 | capital | Tokyo | Tokyo | OK | 0.989 | 52.99 s |
| 12 | 1 | math | 12 | 12 | OK | 0.966 | 52.44 s |
| 13 | 1 | entailment | no | no | OK | 0.988 | 52.97 s |
| 14 | 1 | topic | Science/Technology | Science/Technology | OK | 0.998 | 52.21 s |
| 15 | 1 | math | 21 | 23 | XX | 0.546 | 52.88 s |
| 16 | 2 | math | 322 | 322 | OK | 0.995 | 52.73 s |
| 17 | 2 | logic | yes | yes | OK | 0.982 | 51.69 s |
| 18 | 2 | math | 32 | 30 | XX | 0.792 | 54.72 s |
| 19 | 2 | entailment | no | no | OK | 0.985 | 52.55 s |
| 20 | 2 | code | [2, 3] | [2, 3] | OK | 0.986 | 53.92 s |
| 21 | 2 | logic | no | no | OK | 0.986 | 53.71 s |
| 22 | 2 | math | 36 | 36 | OK | 0.798 | 56.32 s |
| 23 | 2 | code | 5 | 5 | OK | 0.986 | 54.85 s |
| 24 | 2 | entailment | no | no | OK | 0.984 | 58.74 s |
| 25 | 2 | math | 150 | 150 | OK | 0.983 | 56.32 s |
| 26 | 2 | logic | no | no | OK | 0.841 | 52.01 s |
| 27 | 2 | code | 8 | 8 | OK | 0.986 | 52.48 s |
| 28 | 2 | math | 50 | 50 | OK | 0.378 | 59.17 s |
| 29 | 2 | logic | no | no | OK | 0.990 | 50.91 s |
| 30 | 2 | code | 0 | 0 | OK | 0.979 | 53.29 s |
| 31 | 2 | math | 25 percent | 25 percent | OK | 0.996 | 52.74 s |
| 32 | 2 | entailment | yes | yes | OK | 0.932 | 53.96 s |
| 33 | 2 | math | 7 | 7 | OK | 0.860 | 53.66 s |
| 34 | 2 | logic | yes | yes | OK | 0.980 | 58.84 s |
| 35 | 2 | code | [1, 2, 3] | [1, 2, 3] | OK | 0.978 | 68.86 s |
| 36 | 3 | math | 0.05 | 0.05 | OK | 0.860 | 53.75 s |
| 37 | 3 | logic | 9 | 9 | OK | 0.990 | 52.32 s |
| 38 | 3 | math | 5 | 5 | OK | 0.991 | 53.13 s |
| 39 | 3 | knowledge | Au | Au | OK | 0.997 | 51.51 s |
| 40 | 3 | math | 47 | 47 | OK | 0.983 | 71.63 s |
| 41 | 3 | knowledge | First human on the Moon | First human on the Moon | OK | 0.997 | 76.30 s |
| 42 | 3 | logic | apples only | apples only | OK | 0.994 | 52.94 s |
| 43 | 3 | math | 40 | 40 | OK | 0.920 | 95.34 s |
| 44 | 3 | knowledge | Jupiter | Jupiter | OK | 0.998 | 64.50 s |
| 45 | 3 | logic | Some A are C | Some A are C | OK | 0.996 | 56.48 s |
| 46 | 3 | code | True | True | OK | 0.966 | 56.15 s |
| 47 | 3 | math | 6400 | 6400 | OK | 0.944 | 58.84 s |
| 48 | 3 | knowledge | William Shakespeare | William Shakespeare | OK | 0.997 | 54.23 s |
| 49 | 3 | logic | yes | yes | OK | 0.965 | 54.53 s |
| 50 | 3 | math | 42 | 42 | OK | 0.994 | 55.58 s |

## Misclassifications

- **#15** (math, tier 1): predicted `21`, gold `23`, confidence 0.546
- **#18** (math, tier 2): predicted `32`, gold `30`, confidence 0.792

## Latency model: a near-fixed floor

Measured separately; cost is almost independent of prompt length:

| prompt | latency |
|---|---|
| 36 tok | 52.2 s |
| 126 tok | 53.2 s |
| 426 tok | 58.7 s |
| 1026 tok | 67.9 s |

`llama-bench`: pp32 0.69 t/s, pp64 1.23, pp128 2.50, pp256 4.50, tg8 5.45 t/s.

The floor comes from rebuilding the hybrid SSM / Gated-DeltaNet recurrent state on every fresh prompt, which is also why the KV prefix cache only helps on an exact-repeat prompt and not across different states.

| strategy | per decision |
|---|---|
| sequential, 1 decision per prefill | ~50 s |
| grammar-batched, N=6 per prefill | ~9.2 s |
| exact-repeat (KV cache hit) | ~0.35 s |

## Notes

- Decisions go through `/v1/completions`, not `/v1/chat/completions`: the chat endpoint returns unconstrained `top_logprobs`, so the option letters can be missing from the candidates and cannot be renormalised.
- `logit_bias` masks all non-letter tokens; a matching GBNF grammar is applied for the batched path.
- Confidence is the renormalised probability of the argmax option.
