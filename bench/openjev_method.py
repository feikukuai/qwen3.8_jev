#!/usr/bin/env python3
"""
openjev method vs single-shot: the shared-prefix warm + branch pattern.

openjev (ekzhang/openjev-sglang) renders the chat template ONCE, splits a COMMON
PREFIX from per-question suffixes, warms the cache with the prefix at
max_new_tokens=1, then issues one branch call per question -- again
max_new_tokens=1 -- reading the label logprobs and renormalising with a stable
softmax. Options render as "A: description" (colon, matching their examples).

Why it is much faster: the expensive per-call cost is rebuilding the hybrid
Qwen recurrent state for a COLD prompt. Once the common prefix is resident, each
branch only pays for its own short suffix.

This is the corrected, reproducible version of the openjev pattern on llama.cpp.
"""
import json, math, time, urllib.request

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def softmax(xs):
    peak = max(xs)
    w = [math.exp(x - peak) for x in xs]
    s = math.fsum(w)
    return [x / s for x in w]


class OpenJevClient:
    def __init__(self, url="http://127.0.0.1:8080", model="local", timeout=1800):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _post(self, path, payload):
        req = urllib.request.Request(self.url + path, json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    @staticmethod
    def common_prefix(state, preamble=None):
        p = preamble or ("You are a decision function. Read the state, then answer each "
                         "question by choosing exactly one option.\n\n")
        return p + f"[State]\n{state}\n\n"

    @staticmethod
    def branch_suffix(question, options):
        opts = "\n".join(f"{LETTERS[i]}: {o}" for i, o in enumerate(options))
        return f"[Question]\n{question}\n\n[Options]\n{opts}\n\nAnswer:\n"

    def warm(self, prefix):
        t0 = time.time()
        self._post("/v1/completions", {"model": self.model, "prompt": prefix,
                                       "max_tokens": 1, "temperature": 0, "logprobs": 2})
        return time.time() - t0

    def branch(self, prompt, n_opts):
        t0 = time.time()
        out = self._post("/v1/completions", {"model": self.model, "prompt": prompt,
                                             "max_tokens": 1, "temperature": 0,
                                             "logprobs": 20})
        dt = time.time() - t0
        ch = out["choices"][0]
        top = {t["token"]: t["logprob"]
               for t in ch["logprobs"]["content"][0]["top_logprobs"]}
        vals, found = [], 0
        for i in range(n_opts):
            L = LETTERS[i]
            v = [x for tok, x in top.items() if tok.strip() == L]
            vals.append(max(v) if v else None)
            found += bool(v)
        if found < 2:
            return None, dt, ch["text"], found
        probs = softmax(vals)
        return probs, dt, ch["text"], found

    def decide_batch(self, state, questions, warm=True):
        """questions: [(question, options)]. Returns list of decisions + timings."""
        prefix = self.common_prefix(state)
        t_warm = self.warm(prefix) if warm else 0.0
        out = []
        t0 = time.time()
        for q, opts in questions:
            prompt = prefix + self.branch_suffix(q, opts)
            probs, dt, tok, found = self.branch(prompt, len(opts))
            if probs is None:
                out.append({"choice": None, "confidence": 0.0, "latency_s": dt})
            else:
                best = max(range(len(opts)), key=lambda i: probs[i])
                out.append({
                    "choice": opts[best],
                    "confidence": probs[best],
                    "distribution": list(zip(opts, probs)),
                    "latency_s": dt,
                    "argmax_token": tok,
                })
        wall = time.time() - t0
        for r in out:
            r["batch_wall_s"] = wall
            r["amortised_s"] = (wall + t_warm) / len(questions)
            r["warm_s"] = t_warm
        return out


if __name__ == "__main__":
    c = OpenJevClient()
    state = ("The plot is thin, but the two leads are so charming that I left the "
             "cinema smiling. The soundtrack was forgettable.")
    qs = [
        ("What is the sentiment of this review?", ["negative", "positive"]),
        ("Is this review about a film?", ["no", "yes"]),
        ("Would a typical reader recommend it?", ["no", "yes"]),
        ("Is the tone mostly positive?", ["no", "yes"]),
        ("Does the review mention music?", ["no", "yes"]),
        ("Is the reviewer overall satisfied?", ["no", "yes"]),
    ]
    res = c.decide_batch(state, qs)
    for r in res:
        print(f"  {r['choice']:6s} p={r['confidence']:.3f}  {r['latency_s']*1000:7.0f} ms")
    print(f"  warm {res[0]['warm_s']:.2f}s | batch {res[0]['batch_wall_s']:.2f}s "
          f"-> {res[0]['amortised_s']:.2f}s per question")
