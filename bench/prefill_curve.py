#!/usr/bin/env python3
"""
Measure prefill throughput vs prompt length to get the real cost model.

Earlier I claimed latency was a near-fixed ~50 s floor. That was WRONG: the
"fixed" look came from comparing prompts that all happened to be in the same
length band. The truth is close to linear in prompt tokens, with a large slope.
"""
import json, time, urllib.request, random, string

def post(payload, timeout=3600):
    req = urllib.request.Request("http://127.0.0.1:8080/v1/completions",
        json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

uniq = lambda: "".join(random.choices(string.ascii_lowercase, k=8))

print("words | approx tok | latency s | tok/s")
rows = []
for words in (10, 25, 50, 100, 200, 400, 800):
    prompt = "Zz" + uniq() + " " + ("context " * words) + "\nAnswer with one letter:"
    toks = len(prompt) // 4
    t0 = time.time()
    post({"model": "local", "prompt": prompt, "max_tokens": 1,
          "temperature": 0, "logprobs": 2})
    dt = time.time() - t0
    rows.append((toks, dt))
    print("%5d | %10d | %9.2f | %6.2f" % (words, toks, dt, toks / dt))

# linear fit
n = len(rows)
sx = sum(r[0] for r in rows); sy = sum(r[1] for r in rows)
sxx = sum(r[0]*r[0] for r in rows); sxy = sum(r[0]*r[1] for r in rows)
slope = (n*sxy - sx*sy) / (n*sxx - sx*sx)
inter = (sy - slope*sx) / n
print()
print("fit: latency ~= %.4f * tokens + %.1f  (s)" % (slope, inter))
print("=> %.1f ms per prompt token, %.1f tok/s prefill" % (slope*1000, 1/slope))
