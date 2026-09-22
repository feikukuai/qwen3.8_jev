#!/usr/bin/env python3
"""
COST MODEL for a single-pass decision on this box (llama.cpp, CPU, Qwen3.5-27B-Q3_K_S).

Summary of everything measured:

  prompt tokens    latency
  -------------    --------
    7-14           ~0.9 s
    30-35          ~1.8 s
    36-37          ~50 s      <-- sharp discontinuity
    ~100           ~50 s
    ~250           ~52 s
    ~1600          ~75 s

Two regimes:
  * t < 36 tokens : ~1-2 s   (short-prompt path)
  * t >= 36 tokens: ~50 s + ~0.015 s/token

The ~50 s intercept dominates for any realistic decision prompt, because a
decision prompt (instruction header + state + question + options + "Answer:")
is essentially never under 36 tokens.

So the actionable conclusion is NOT "shorten the prompt" (saving 200 tokens
costs only 3 s). It is: amortise the fixed cost.

  * one decision per prefill, sequential    : ~50-56 s per decision
  * N decisions sharing a warm common prefix: ~1.5-1.6 s per decision after one 52 s warm
  * exact-repeat prompt (cache hit)         : ~0.34 s

The shared-prefix pattern is what openjev/SemIf does (render the template once,
warm the radix cache with the common prefix, then one branch call per question).
On this CPU box that turns 50 s/decision into ~1.6 s/decision for every decision
after the first -- a ~30x improvement, and this is what the earlier 4B notes and
openjev both rely on.

Critically, the win requires the prefix to be REUSED. Cache only matches on an
exact prefix, so:
  - all questions must share the SAME state/preamble
  - the per-question part must come LAST
  - use a single large slot (--parallel 1 -c 8192); --parallel 4 shards the
    context into small slots and thrashes the cache (observed: alternating
    1.6 s / 58 s)
"""
import json, time, urllib.request

def post(payload, timeout=1800):
    req = urllib.request.Request("http://127.0.0.1:8080/v1/completions",
        json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def measure(prompt):
    t0 = time.time()
    post({"model":"local","prompt":prompt,"max_tokens":1,"temperature":0,"logprobs":2})
    return time.time() - t0

if __name__ == "__main__":
    import random, string
    print("=== demonstrate the two regimes on this box ===")
    for n in (10, 30, 50, 100, 400, 1200):
        filler = ("context " * n)
        p = "Z" + "".join(random.choices(string.ascii_lowercase, k=6)) + " " + filler + "Answer:"
        print("  ~%5d tok -> %7.2f s" % (n * 2, measure(p)))
