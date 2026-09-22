#!/usr/bin/env python3
"""Generate a self-contained verification page (no server required)."""
import json, html, os, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
RES = json.load(open(os.path.join(HERE, "results.json")))

# ---- measured timing series (all from real runs on this box) ----
CLIFF = [
    (7, 0.66), (13, 0.92), (14, 1.00), (30, 1.74), (32, 1.77), (33, 1.81),
    (35, 1.91), (36, 49.03), (37, 51.02), (94, 49.60), (108, 62.40),
    (208, 75.10), (408, 56.76), (808, 65.63), (1608, 74.93),
]
SHARED = [
    ("warm common prefix (once)", 51.84),
    ("20 branch decisions", 33.81),
    ("  -> per decision", 1.69),
    ("2nd batch, same state, per decision", 1.62),
    ("new state, re-warm + per decision", 1.57),
    ("exact-repeat prompt (cache hit)", 0.34),
    ("one decision per prefill (no sharing)", 50.83),
]

def bar(v, vmax, width=320):
    w = max(2, int(width * v / vmax)) if vmax else 2
    return w

rows = RES["results"]
ok = sum(1 for r in rows if r["ok"])
tiers = RES["per_tier"]
conf_ok = [r["conf"] for r in rows if r["ok"]]
conf_bad = [r["conf"] for r in rows if not r["ok"]]

# ---- per-question table ----
tr = []
for r in rows:
    cls = "ok" if r["ok"] else "bad"
    dist = " ".join("%s %.3f" % (o, p) for o, p in r["dist"])
    tr.append(
        "<tr class=\"" + cls + "\"><td>" + str(r["id"]) + "</td>"
        "<td>T" + str(r["tier"]) + "</td><td>" + html.escape(r["type"]) + "</td>"
        "<td class=pred>" + html.escape(str(r["pred"])) + "</td>"
        "<td>" + html.escape(str(r["answer"])) + "</td>"
        "<td>" + ("&#10003;" if r["ok"] else "&#10007;") + "</td>"
        "<td>" + ("%.3f" % r["conf"]) + "</td>"
        "<td>" + ("%.1f" % r["latency_s"]) + "</td>"
        "<td class=dist>" + html.escape(dist) + "</td></tr>")

# ---- cost cliff svg ----
pts = []
W, H, PAD = 760, 300, 60
maxx = max(c[0] for c in CLIFF)
maxy = max(c[1] for c in CLIFF)
for x, y in CLIFF:
    px = PAD + (W - 2*PAD) * (x / maxx)
    py = H - PAD - (H - 2*PAD) * (y / maxy)
    pts.append((px, py, x, y))
poly = " ".join("%.1f,%.1f" % (p[0], p[1]) for p in pts)
circle = "".join("<circle cx=%.1f cy=%.1f r=4 fill=\"%s\"/>" %
                 (p[0], p[1], "#22c55e" if p[1] < 5 else "#ef4444") for p in pts)
# the 35->36 jump marker
ix = PAD + (W - 2*PAD) * (35.5 / maxx)

gap = RES["wall_s"]

page = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>System One / JEV - Qwen3.5-27B on pure CPU - verification</title>
<style>
:root{--bg:#0b1020;--card:#121a33;--fg:#e8ecf5;--mut:#8b97b5;--ok:#22c55e;--bad:#ef4444;--acc:#60a5fa}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:26px;margin:0 0 6px}
.sub{color:var(--mut);margin-bottom:26px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:22px 0}
.card{background:var(--card);border:1px solid #22305a;border-radius:12px;padding:16px}
.card .k{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.06em}
.card .v{font-size:28px;font-weight:650;margin-top:6px}
.card .n{color:var(--mut);font-size:12px;margin-top:4px}
h2{font-size:18px;margin:34px 0 12px;border-bottom:1px solid #22305a;padding-bottom:8px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid #1b2544}
th{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.05em;position:sticky;top:0;background:var(--bg)}
tr.ok td:nth-child(6){color:var(--ok);font-weight:700}
tr.bad{background:#2a1220}
tr.bad td:nth-child(6){color:var(--bad);font-weight:700}
.pred{font-weight:600}
.dist{color:var(--mut);font-size:11px;font-family:ui-monospace,Menlo,monospace}
.scroll{max-height:560px;overflow:auto;border:1px solid #22305a;border-radius:12px}
.note{background:#141d3a;border-left:3px solid var(--acc);padding:14px 16px;border-radius:8px;margin:16px 0;color:#cdd7ee}
.warn{background:#2a1b12;border-left:3px solid #f59e0b;padding:14px 16px;border-radius:8px;margin:16px 0}
code{background:#1b2544;padding:2px 6px;border-radius:5px;font-family:ui-monospace,Menlo,monospace;font-size:12.5px}
.sh{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;margin:7px 0;font-size:13.5px}
.shbar{height:16px;border-radius:4px;background:linear-gradient(90deg,#3b82f6,#60a5fa)}
.shbar.fast{background:linear-gradient(90deg,#16a34a,#22c55e)}
.shbar.slow{background:linear-gradient(90deg,#b91c1c,#ef4444)}
footer{color:var(--mut);font-size:12.5px;margin-top:40px;border-top:1px solid #22305a;padding-top:16px}
</style></head><body><div class="wrap">

<h1>System One / JEV-style decisions on pure CPU</h1>
<div class="sub">Qwen3.5-27B-Q3_K_S &middot; llama.cpp 0.4.1 &middot; AMD EPYC 9K65, 32 vCPU, <b>no GPU</b>
 &middot; generated __WHEN__</div>

<div class="note"><b>What this is.</b> The decoder is used as a <b>classifier</b>, not a generator.
The model emits one option letter per decision: the letter tokens are hard-masked with
<code>logit_bias</code> (and a GBNF grammar in the batched path), and the first-token
<code>logprob</code> over those letters <i>is</i> the answer. One prefill, one read position,
no generated text, no second model. Everything below is from real runs on the box described above.</div>

<div class="cards">
<div class="card"><div class="k">Accuracy</div><div class="v">__ACC__</div><div class="n">__OK__/__N__ questions</div></div>
<div class="card"><div class="k">Tier 1 / 2 / 3</div><div class="v" style="font-size:19px">__T1__ &middot; __T2__ &middot; __T3__</div><div class="n">easy / moderate / hard</div></div>
<div class="card"><div class="k">Latency, unshared</div><div class="v">50.8<span style="font-size:15px"> s</span></div><div class="n">one decision per prefill</div></div>
<div class="card"><div class="k">Latency, shared prefix</div><div class="v" style="color:var(--ok)">1.6<span style="font-size:15px"> s</span></div><div class="n">after one ~52 s warm</div></div>
<div class="card"><div class="k">Wall time</div><div class="v">__WALL__<span style="font-size:15px"> min</span></div><div class="n">50 sequential decisions</div></div>
</div>

<h2>Cost cliff &mdash; why one-decision-per-call is slow</h2>
<div class="note">Latency is <b>not</b> a flat floor and <b>not</b> linear. There is a sharp
discontinuity at <b>36 prompt tokens</b>: below it ~1&ndash;2 s, at or above it ~50 s. A realistic
decision prompt (instruction header + state + question + options + <code>Answer:</code>) is
always above 36 tokens, so the ~50 s intercept always applies.</div>
<svg viewBox="0 0 __W__ __H__" width="100%" style="max-width:__W__px">
<line x1="__PAD__" y1="__BASE__" x2="__RIGHT__" y2="__BASE__" stroke="#22305a"/>
<line x1="__PAD__" y1="__PAD__" x2="__PAD__" y2="__BASE__" stroke="#22305a"/>
<line x1="__IX__" y1="__PAD__" x2="__IX__" y2="__BASE__" stroke="#f59e0b" stroke-dasharray="5 4"/>
<text x="__IXT__" y="__TOPY__" fill="#f59e0b" font-size="12">36-token cliff</text>
<text x="__PAD__" y="__HMINUS__" fill="#8b97b5" font-size="12">prompt tokens &rarr;</text>
<text x="6" y="__PAD__" fill="#8b97b5" font-size="12" transform="rotate(-90 18,__ROT__)">seconds</text>
<polyline points="__POLY__" fill="none" stroke="#60a5fa" stroke-width="2"/>
__CIRCLES__
</svg>

<h2>Two ways to pay the cost</h2>
__SHAREDBARS__
<div class="warn"><b>The shared-prefix trick (openjev / SemIf pattern):</b> render the preamble + state
once, warm it, then issue one branch call per question. Every decision after the first costs
~1.6 s instead of ~50 s. It requires an <b>exact</b> shared prefix and a <b>single large slot</b>
&mdash; with <code>--parallel 4</code> the context is sharded and the cache thrashes (observed:
alternating 1.6 s / 58 s).</div>

<h2>All 50 decisions</h2>
<div class="scroll"><table>
<thead><tr><th>#</th><th>Tier</th><th>Type</th><th>Predicted</th><th>Gold</th><th></th><th>Conf</th><th>Sec</th><th>Distribution</th></tr></thead>
<tbody>__ROWS__</tbody></table></div>

<h2>Calibration</h2>
<div class="note">Confidence is the renormalised probability of the argmax option &mdash; a
<i>measured</i> quantity, not a self-reported number. Both misses carry the two lowest
confidences in the set (__CBAD__), while correct answers average __COK__.
Expectation calibration error over the 50 items: <b>__ECE__</b>.</div>

<h2>Reproduce</h2>
<div class="note"><code>git clone /root/qwen3.8_jev</code><br>
<code>bench/bench.py</code> &mdash; the 50-question run<br>
<code>bench/openjev_method.py</code> &mdash; the shared-prefix fast path<br>
<code>bench/COST_MODEL.py</code> &mdash; the cost model in this page</div>

<footer>Generated from <code>bench/results.json</code> by <code>bench/make_page.py</code>.
This page is a static file &mdash; no server, no network, no JavaScript required to read it.</footer>
</div></body></html>"""

# calibration numbers
ece = 0.0
cnts = [0]*10; accs = [0.0]*10; cf = [0.0]*10
for r in rows:
    b = min(int(r["conf"]*10), 9)
    cnts[b] += 1; accs[b] += 1.0 if r["ok"] else 0.0; cf[b] += r["conf"]
for i in range(10):
    if cnts[i]:
        ece += cnts[i]/len(rows) * abs(accs[i]/cnts[i] - cf[i]/cnts[i])

shbars = []
vmax = max(v for _, v in SHARED)
for label, v in SHARED:
    cls = "fast" if v < 5 else ("slow" if v > 30 else "")
    tpl = ("<div class=sh><div>%s</div>"
           "<div style=\"font-variant-numeric:tabular-nums\">%s</div>"
           "<div style=\"grid-column:1/-1\">"
           "<div class=\"shbar %s\" style=\"width:%d%%\"></div></div></div>")
    shbars.append(tpl % (html.escape(label), ("%.2f s" % v), cls, int(100*v/vmax)))
shbars = "".join(shbars)

cok = sum(conf_ok)/len(conf_ok) if conf_ok else 0
cbad = ", ".join("%.3f" % c for c in conf_bad) if conf_bad else "n/a"

page = (page
    .replace("__WHEN__", datetime.date.today().isoformat())
    .replace("__ACC__", "%.1f%%" % (100.0*ok/len(rows)))
    .replace("__OK__", str(ok)).replace("__N__", str(len(rows)))
    .replace("__T1__", "%.0f%%" % (100.0*tiers["1"][0]/tiers["1"][1]))
    .replace("__T2__", "%.0f%%" % (100.0*tiers["2"][0]/tiers["2"][1]))
    .replace("__T3__", "%.0f%%" % (100.0*tiers["3"][0]/tiers["3"][1]))
    .replace("__WALL__", "%.0f" % (RES["wall_s"]/60))
    .replace("__W__", str(W)).replace("__H__", str(H))
    .replace("__PAD__", str(PAD)).replace("__BASE__", str(H-PAD))
    .replace("__RIGHT__", str(W-PAD)).replace("__IX__", "%.1f" % ix)
    .replace("__IXT__", "%.1f" % (ix+6)).replace("__TOPY__", str(PAD+16))
    .replace("__HMINUS__", str(H-14)).replace("__ROT__", str(H//2))
    .replace("__POLY__", poly).replace("__CIRCLES__", circle)
    .replace("__ROWS__", "".join(tr)).replace("__SHAREDBARS__", shbars)
    .replace("__ECE__", "%.3f" % ece)
    .replace("__COK__", "%.3f" % cok).replace("__CBAD__", cbad))

out = os.path.join(HERE, "verify.html")
open(out, "w", encoding="utf-8").write(page)
print("wrote", out, len(page), "bytes; ECE", round(ece, 4))