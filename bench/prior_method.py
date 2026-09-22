#!/usr/bin/env python3
"""
Prior-work method (SemIf/OpenJev style) adapted to llama.cpp + 27B.

Differences from my earlier harness, taken from the 4B deploy notes:
  * options are passed as a JSON payload {evidence, criterion, options:[{letter,description}]}
  * the system prompt explicitly says "Respond with only its uppercase letter"
  * thinking is disabled at the template level (enable_thinking=False)
  * probabilities come from the option-letter LOGITS only (not top_logprobs)

That last point is the big one: reading raw logits at the FIRST position for a
fixed set of token ids is exactly what openjev does with token_ids_logprob.
llama.cpp can do this via logit_bias + reading the returned top_logprobs, which
is what my system_one.py does. Here I test the JSON-payload prompt shape.
"""
import json, math, time, urllib.request

LETTERS = "ABCDEFGHIJKLMNOP"

def post(payload, timeout=1800):
    req = urllib.request.Request("http://127.0.0.1:8080/v1/completions",
        json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def build_prompt(state, question, options):
    payload = {"evidence": state, "criterion": question,
               "options": [{"letter": LETTERS[i], "description": o}
                           for i, o in enumerate(options)]}
    system = ("Apply the supplied criterion to the supplied evidence. "
              "Choose exactly one listed option. Respond with only its "
              "uppercase letter, with no explanation or reasoning.")
    return system + "\n\n" + json.dumps(payload, ensure_ascii=False) + "\n\nAnswer:"

def softmax(vals):
    m = max(vals); ex = [math.exp(v - m) for v in vals]; s = sum(ex)
    return [v / s for v in ex]

def decide(state, question, options):
    prompt = build_prompt(state, question, options)
    t0 = time.time()
    out = post({"model": "local", "prompt": prompt, "max_tokens": 1,
                "temperature": 0, "logprobs": 20,
                "logit_bias": {f" {L}": 100.0 for L in LETTERS[:len(options)]} |
                              {L: 100.0 for L in LETTERS[:len(options)]}})
    dt = time.time() - t0
    ch = out["choices"][0]
    top = {t["token"]: t["logprob"] for t in ch["logprobs"]["content"][0]["top_logprobs"]}
    vals = []
    for i in range(len(options)):
        L = LETTERS[i]
        v = [x for tok, x in top.items() if tok.strip() == L]
        vals.append(max(v) if v else -100.0)
    probs = softmax(vals)
    best = max(range(len(probs)), key=probs.__getitem__)
    return {"choice": options[best], "confidence": probs[best],
            "latency_s": dt, "dist": dict(zip(options, [round(p, 4) for p in probs]))}

if __name__ == "__main__":
    # the exact same 6 decisions used for my method, for a like-for-like number
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
    tot = 0.0
    for q, opts in qs:
        r = decide(state, q, opts)
        tot += r["latency_s"]
        print("  %-8s p=%.3f  %6.0f ms  %s" % (r["choice"], r["confidence"],
                                                r["latency_s"]*1000, r["dist"]))
    print("  TOTAL %.1f s -> %.2f s per question (sequential, cold each time)" % (tot, tot/len(qs)))
