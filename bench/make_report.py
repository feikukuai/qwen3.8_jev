#!/usr/bin/env python3
"""Turn results.json into bench/REPORT.md."""
import json, os, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "results.json")))
res = [r for r in d["results"] if "pred" in r]
n = len(res)
ok = sum(1 for r in res if r["ok"])
tiers, types = {}, {}
for r in res:
    for key, acc in (("tier", tiers), ("type", types)):
        acc.setdefault(r[key], [0, 0])
        acc[r[key]][0] += r["ok"]
        acc[r[key]][1] += 1

lat = [r["latency_s"] for r in res]
warm = [x for x in lat if x < 5]
cold = [x for x in lat if x >= 5]

# expected calibration error over the returned top-1 confidence
cnts = [0] * 10; accs = [0.0] * 10; confs = [0.0] * 10
for r in res:
    b = min(int(r["conf"] * 10), 9)
    cnts[b] += 1
    accs[b] += 1.0 if r["ok"] else 0.0
    confs[b] += r["conf"]
ece = 0.0
for i in range(10):
    if cnts[i]:
        ece += cnts[i] / float(n) * abs(accs[i] / cnts[i] - confs[i] / cnts[i])

L = []
L.append("# Pure-CPU System One / JEV-style benchmark - Qwen3.5-27B-Q3_K_S")
L.append("")
L.append("Engine llama.cpp 0.4.1-dev, `-ngl 0`, OpenBLAS, 32 threads, AMD EPYC 9K65 "
         "(AVX-512 + VNNI). No GPU. All decisions are one prefill, one read position.")
L.append("")
L.append("## Headline")
L.append("")
L.append("| metric | value |")
L.append("|---|---|")
L.append("| **Accuracy** | **{}/{} = {:.1f}%** |".format(ok, n, 100.0 * ok / n))
for t in sorted(tiers):
    c, m = tiers[t]
    L.append("| Tier {} | {}/{} = {:.1f}% |".format(t, c, m, 100.0 * c / m))
if lat:
    L.append("| Latency, mean | {:.2f} s |".format(statistics.mean(lat)))
    if warm:
        L.append("| Latency, cache-hit floor | {:.2f} s ({} items) |".format(
                 statistics.mean(warm), len(warm)))
    if cold:
        L.append("| Latency, cold prefill | {:.2f} s ({} items) |".format(
                 statistics.mean(cold), len(cold)))
    L.append("| Wall time | {:.0f} s |".format(d.get("wall_s", 0)))
L.append("| Calibration error (ECE) | {:.3f} |".format(ece))
L.append("")
L.append("## Accuracy by task type")
L.append("")
L.append("| type | correct | accuracy |")
L.append("|---|---|---|")
for t in sorted(types):
    c, m = types[t]
    L.append("| {} | {}/{} | {:.1f}% |".format(t, c, m, 100.0 * c / m))
L.append("")
L.append("## Per-question results")
L.append("")
L.append("| # | tier | type | predicted | gold | ok | confidence | latency |")
L.append("|---|---|---|---|---|---|---|---|")
for r in res:
    L.append("| {} | {} | {} | {} | {} | {} | {:.3f} | {:.2f} s |".format(
        r["id"], r["tier"], r["type"], r["pred"], r["answer"],
        "OK" if r["ok"] else "XX", r["conf"], r["latency_s"]))
L.append("")
L.append("## Misclassifications")
L.append("")
bad = [r for r in res if not r["ok"]]
if not bad:
    L.append("None - 50/50.")
else:
    for r in bad:
        L.append("- **#{}** ({}, tier {}): predicted `{}`, gold `{}`, confidence {:.3f}".format(
                 r["id"], r["type"], r["tier"], r["pred"], r["answer"], r["conf"]))
L.append("")
L.append("## Latency model: a near-fixed floor")
L.append("")
L.append("Measured separately; cost is almost independent of prompt length:")
L.append("")
L.append("| prompt | latency |")
L.append("|---|---|")
L.append("| 36 tok | 52.2 s |")
L.append("| 126 tok | 53.2 s |")
L.append("| 426 tok | 58.7 s |")
L.append("| 1026 tok | 67.9 s |")
L.append("")
L.append("`llama-bench`: pp32 0.69 t/s, pp64 1.23, pp128 2.50, pp256 4.50, tg8 5.45 t/s.")
L.append("")
L.append("The floor comes from rebuilding the hybrid SSM / Gated-DeltaNet recurrent state "
         "on every fresh prompt, which is also why the KV prefix cache only helps on an "
         "exact-repeat prompt and not across different states.")
L.append("")
L.append("| strategy | per decision |")
L.append("|---|---|")
L.append("| sequential, 1 decision per prefill | ~50 s |")
L.append("| grammar-batched, N=6 per prefill | ~9.2 s |")
L.append("| exact-repeat (KV cache hit) | ~0.35 s |")
L.append("")
L.append("## Notes")
L.append("")
L.append("- Decisions go through `/v1/completions`, not `/v1/chat/completions`: the chat "
         "endpoint returns unconstrained `top_logprobs`, so the option letters can be "
         "missing from the candidates and cannot be renormalised.")
L.append("- `logit_bias` masks all non-letter tokens; a matching GBNF grammar is applied "
         "for the batched path.")
L.append("- Confidence is the renormalised probability of the argmax option.")
open(os.path.join(HERE, "REPORT.md"), "w").write("\n".join(L) + "\n")
print("wrote REPORT.md", ok, "/", n, "ECE", round(ece, 3))