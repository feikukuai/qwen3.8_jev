#!/usr/bin/env python3
"""
System One / JEV-style output-side harness for llama.cpp on CPU.

The decoder is used as a CLASSIFIER, never as a generator:
  * logit_bias  -> hard mask: only declared option-letter tokens are samplable
  * grammar     -> the same constraint as GBNF (belt and braces)
  * logprob     -> the single first-token distribution over the letters IS the decision

Why /v1/completions and not /v1/chat/completions
------------------------------------------------
Measured on llama.cpp 0.4.1: the chat endpoint returns top_logprobs of the
UNCONSTRAINED distribution, so the option letters can be absent from the returned
candidates and cannot be renormalised. /v1/completions returns the raw logprobs
for BOTH declared letters even with logit_bias applied -- exactly what a
single-pass classifier needs.

Single pass ("单传"): one prefill, one read position, no generation loop, and no
second model anywhere in the path ("JEV只做输出端分类").
"""
import json, math, os, time, urllib.request

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
HEADER = ("You are a decision function. Read the state, then answer the question "
          "by choosing exactly one option.\n\n")
MAX_OPTIONS = 20


def build_prompt(state, question, options):
    lines = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return (f"{HEADER}[State]\n{state}\n\n[Question]\n{question}\n\n"
            f"[Options]\n{lines}\n\nAnswer:")


def build_grammar(n_options):
    """GBNF: after 'Answer:' the only legal continuation is one option letter."""
    opts = " | ".join(f'" {LETTERS[i]}"' for i in range(n_options))
    return f'root ::= letter\nletter ::= {opts}\n'


def letter_bias(n_options, strength=100.0):
    """logit_bias keyed by literal token text; covers both ' A' and 'A'."""
    b = {}
    for i in range(n_options):
        L = LETTERS[i]
        b[f" {L}"] = strength
        b[L] = strength
    return b


class SystemOne:
    """Single-pass decision client against a llama.cpp OpenAI-compatible server."""

    def __init__(self, url, model="local", n_probs=20, timeout=1200):
        self.url = url.rstrip("/")
        self.model = model
        self.n_probs = n_probs
        self.timeout = timeout
        self._warm = set()

    def _post(self, path, payload):
        req = urllib.request.Request(
            self.url + path, json.dumps(payload).encode(),
            {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def decide(self, state, question, options, temperature=0.0, use_grammar=False):
        """One prefill -> calibrated distribution over the declared options."""
        n = len(options)
        assert 2 <= n <= MAX_OPTIONS, "option count out of range"
        prompt = build_prompt(state, question, options)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": temperature,
            "logprobs": self.n_probs,
            "logit_bias": letter_bias(n),
        }
        if use_grammar:
            payload["grammar"] = build_grammar(n)

        t0 = time.time()
        out = self._post("/v1/completions", payload)
        dt = time.time() - t0

        ch = out["choices"][0]
        content = ch["logprobs"]["content"][0]
        top = {t["token"]: t["logprob"] for t in content["top_logprobs"]}

        floor = min(top.values()) - 5.0
        logits = []
        for i in range(n):
            L = LETTERS[i]
            v = [lp for tok, lp in top.items() if tok.strip() == L]
            logits.append(max(v) if v else floor)

        z = max(logits)
        exps = [math.exp(x - z) for x in logits]
        s = sum(exps)
        probs = [e / s for e in exps]
        ranked = sorted(zip(options, probs), key=lambda t: -t[1])
        return {
            "choice": ranked[0][0],
            "confidence": ranked[0][1],
            "distribution": ranked,
            "latency_s": dt,
            "argmax_token": ch["text"],
            "letters_found": sum(1 for i in range(n)
                                 if any(tok.strip() == LETTERS[i] for tok in top)),
        }


if __name__ == "__main__":
    base = SystemOne(os.environ.get("SO_URL", "http://127.0.0.1:8080"),
                     os.environ.get("SO_MODEL", "local"))
    r = base.decide("The plot is thin, but the two leads are so charming that I left smiling.",
                    "What is the sentiment of this review?", ["negative", "positive"])
    print(json.dumps(r, indent=1, default=str))
