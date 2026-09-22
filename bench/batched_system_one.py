#!/usr/bin/env python3
"""
Batched System One: N independent decisions in ONE prefill ("单传" at scale).

Measured on this box (EPYC 9K65, 32 threads, Qwen3.5-27B-Q3_K_S, pure CPU):

    sequential, 1 decision per prefill : ~50 s / decision
    batched,   N decisions per prefill : ~9.2 s / decision  (N=6, grammar-constrained)
    exact-repeat (KV cache hit)        : ~0.35 s / decision

The ~50 s is a near-FIXED per-call cost, not proportional to prompt length
(36 tok -> 52 s, 1026 tok -> 68 s). It is dominated by the hybrid SSM/Gated-DeltaNet
recurrent state that must be rebuilt on every fresh prompt, so the win comes from
amortising ONE prefill across many decisions.

Constraints on the output side (decoder as classifier):
  * grammar   -> GBNF admitting exactly N letters, single-space separated
  * logit_bias-> masks every token that is not a declared option letter
  * logprob   -> per-position letter distribution IS the decision
"""
import json, math, time, urllib.request

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _letter_bias(max_opts, strength=100.0):
    b = {}
    for i in range(max_opts):
        L = LETTERS[i]
        b[f" {L}"] = strength
        b[L] = strength
    return b


def grammar_for(n_items, n_letters):
    """Exactly n_items letters, single-space separated, then stop."""
    letters = " ".join(f'" {L}"' for L in LETTERS[:n_letters])
    if n_items == 1:
        seq = "letter"
    else:
        seq = "letter " + " ".join(['" " letter'] * (n_items - 1))
    return f'root ::= {seq}\nletter ::= {letters}\n'


def build_batch_prompt(items):
    """items: list of (state, question, [options])."""
    blocks = []
    for i, (st, q, opts) in enumerate(items):
        lines = "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(opts))
        blocks.append(f"[Item {i}]\nState: {st}\nQuestion: {q}\nOptions:\n{lines}")
    return ("You are a decision function. For each item output ONLY its option letter.\n\n"
            + "\n\n".join(blocks)
            + "\n\nAnswer with only the letters, one per item:\n")


class BatchedSystemOne:
    def __init__(self, url="http://127.0.0.1:8080", model="local", timeout=1800):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _post(self, path, payload):
        req = urllib.request.Request(self.url + path, json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def decide_many(self, items, temperature=0.0):
        """items: [(state, question, options)] -> list of decision dicts."""
        n = len(items)
        maxopt = max(len(o) for _, _, o in items)
        prompt = build_batch_prompt(items)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 2 * n + 2,
            "temperature": temperature,
            "logprobs": 20,
            "logit_bias": _letter_bias(maxopt),
            "grammar": grammar_for(n, maxopt),
        }
        t0 = time.time()
        out = self._post("/v1/completions", payload)
        dt = time.time() - t0

        ch = out["choices"][0]
        positions = ch["logprobs"]["content"]

        results = []
        # letter positions are the even-indexed emitted tokens (letter, space, letter, ...)
        li = 0
        for k, (st, q, opts) in enumerate(items):
            # find the k-th position whose token is a letter
            dist = None
            while li < len(positions):
                lp = positions[li]
                top = {t["token"]: t["logprob"] for t in lp["top_logprobs"]}
                found = {}
                for j in range(len(opts)):
                    L = LETTERS[j]
                    v = [x for tok, x in top.items() if tok.strip() == L]
                    if v:
                        found[L] = max(v)
                li += 1
                if len(found) >= 2:
                    vals = list(found.values())
                    m = max(vals)
                    ex = [math.exp(v - m) for v in vals]
                    s = sum(ex)
                    probs = {L: e / s for L, e in zip(found.keys(), ex)}
                    dist = probs
                    break
            if dist is None:
                results.append({"choice": None, "confidence": 0.0,
                                "distribution": [], "item": k})
                continue
            ranked = sorted(((opts[LETTERS.index(L)], p) for L, p in dist.items()),
                            key=lambda t: -t[1])
            results.append({
                "choice": ranked[0][0],
                "confidence": ranked[0][1],
                "distribution": ranked,
                "item": k,
            })
        for r in results:
            r["batch_latency_s"] = dt
            r["amortised_s"] = dt / n
        return results


if __name__ == "__main__":
    b = BatchedSystemOne()
    items = [
        ("The plot is thin, but the two leads are so charming that I left smiling.", "Sentiment?", ["negative", "positive"]),
        ("The food was cold and the waiter was rude.", "Sentiment?", ["negative", "positive"]),
        ("Shares of the chipmaker jumped 8% after it raised guidance.", "Section?", ["World", "Sports", "Business", "Science/Technology"]),
        ("The team scored a touchdown in the final seconds.", "Section?", ["World", "Sports", "Business", "Science/Technology"]),
        ("What is 9 * 7?", "Result?", ["56", "63", "72", "81"]),
        ("What is 100 - 37?", "Result?", ["53", "63", "67", "73"]),
    ]
    for r in b.decide_many(items):
        print(f"  item{r['item']}: {r['choice']:20s} p={r['confidence']:.3f}  "
              f"batch={r['batch_latency_s']:.1f}s amortised={r['amortised_s']:.1f}s")
