#!/usr/bin/env python3
"""
Local test: does INPUT SIZE affect decision latency?

Builds one syntactically identical decision (state + question + 10 options) and
scales ONLY the state from ~50 chars up to ~10,000 Chinese characters (万字),
then measures a single-pass one-token readout for each size.

Each prompt is unique (nonce in the state) so the prefix cache cannot mask the
cost. One call per size, sequential, with the server otherwise idle.
"""
import json, math, time, urllib.request, random, string, sys

URL = "http://127.0.0.1:8080"
LETTERS = "ABCDEFGHIJKLMNOP"

def post(path, payload, timeout=3600):
    req = urllib.request.Request(URL + path, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def ntok(text):
    try:
        return len(post("/tokenize", {"content": text}).get("tokens", []))
    except Exception:
        return -1

# 10 options, as requested
OPTIONS = [
    "账户锁定需要人工解锁",
    "密码重置邮件未送达",
    "用户输错了密码",
    "账户被风控临时冻结",
    "需要二次验证",
    "邮箱地址拼写错误",
    "系统故障导致登录失败",
    "账户已过期需要续费",
    "需要联系客服处理",
    "无法确定具体原因",
]

# A realistic Chinese state, padded with filler to reach the target size.
SENTENCE = ("客户反馈称密码重置成功但登录仍然提示账户已锁定，"
            "已经申请过两封解锁邮件但均未收到，多次尝试后账户状态仍无变化。")

def make_state(target_chars, nonce):
    head = "[工单编号 %s] " % nonce
    body = ""
    i = 0
    while len(head) + len(body) < target_chars:
        body += SENTENCE
        i += 1
    body = body[:max(0, target_chars - len(head))]
    return head + body

def build_prompt(state):
    opts = "\n".join("%s. %s" % (LETTERS[i], o) for i, o in enumerate(OPTIONS))
    return ("你是一个决策函数。阅读状态，然后从选项中选择恰好一个答案。\n\n"
            "[状态]\n%s\n\n[问题]\n这个工单最可能的根本原因是什么？\n\n"
            "[选项]\n%s\n\n答案：" % (state, opts))

def softmax(xs):
    m = max(xs); ex = [math.exp(x - m) for x in xs]; s = sum(ex)
    return [v / s for v in ex]

bias = {}
for L in LETTERS[:len(OPTIONS)]:
    bias[" " + L] = 100.0
    bias[L] = 100.0

print("server: %s" % URL)
print("options: %d" % len(OPTIONS))
print()
print("%8s | %7s | %9s | %9s | %8s" % ("chars", "tokens", "latency_s", "tok/s", "answer"))
print("-" * 62)

targets = [50, 200, 500, 1000, 2000, 4000, 6000, 8000, 10000]
rows = []
for t in targets:
    nonce = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    state = make_state(t, nonce)
    prompt = build_prompt(state)
    tk = ntok(prompt)
    t0 = time.time()
    try:
        out = post("/v1/completions", {
            "model": "local", "prompt": prompt, "max_tokens": 1,
            "temperature": 0, "logprobs": 20, "logit_bias": bias})
        dt = time.time() - t0
        ch = out["choices"][0]
        top = {x["token"]: x["logprob"]
               for x in ch["logprobs"]["content"][0]["top_logprobs"]}
        vals = []
        for i in range(len(OPTIONS)):
            L = LETTERS[i]
            v = [x for tok, x in top.items() if tok.strip() == L]
            vals.append(max(v) if v else -100.0)
        probs = softmax(vals)
        best = max(range(len(probs)), key=probs.__getitem__)
        ans = OPTIONS[best][:12]
        tps = tk / dt if dt else 0
    except Exception as e:
        dt = time.time() - t0
        ans = "ERROR: %s" % str(e)[:30]
        tps = 0
    rows.append((len(state), tk, dt, tps, ans))
    print("%8d | %7d | %9.2f | %9.3f | %s" % (len(state), tk, dt, tps, ans), flush=True)

print()
print("=== fit latency vs tokens ===")
pts = [(r[1], r[2]) for r in rows if r[1] > 0]
if len(pts) >= 3:
    n = len(pts)
    sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
    sxx = sum(p[0]**2 for p in pts); sxy = sum(p[0]*p[1] for p in pts)
    denom = n*sxx - sx*sx
    if denom:
        slope = (n*sxy - sx*sy)/denom
        inter = (sy - slope*sx)/n
        print("latency ~= %.6f * tokens + %.1f  (s)" % (slope, inter))
        print("=> %.2f ms per token, intercept %.1f s" % (slope*1000, inter))
json.dump([{"chars": r[0], "tokens": r[1], "latency_s": r[2], "tok_per_s": r[3], "answer": r[4]} for r in rows],
          open("lettersize_results.json", "w"), ensure_ascii=False, indent=1)
print("wrote lettersize_results.json")
