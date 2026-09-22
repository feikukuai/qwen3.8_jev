#!/usr/bin/env python3
"""Run the 50-question System-One benchmark against the local server."""
import json, os, sys, time, statistics
from questions import load
from system_one import SystemOne

URL = os.environ.get("SO_URL", "http://127.0.0.1:8080")
MODEL = os.environ.get("SO_MODEL", "local")

def main():
    base = SystemOne(URL, MODEL, n_probs=20, timeout=600)
    items = load()
    results, lat = [], []
    correct = 0
    per_tier = {1: [0, 0], 2: [0, 0], 3: [0, 0]}
    t_start = time.time()

    for (qid, tier, ttype, state, question, options, answer) in items:
        try:
            r = base.decide(state, question, options)
        except Exception as e:
            print(f"  [{qid:02d}] ERROR {e}")
            results.append({"id": qid, "tier": tier, "error": str(e)})
            per_tier[tier][1] += 1
            continue
        ok = (r["choice"] == answer)
        correct += ok
        per_tier[tier][0] += ok
        per_tier[tier][1] += 1
        lat.append(r["latency_s"])
        results.append({
            "id": qid, "tier": tier, "type": ttype,
            "pred": r["choice"], "answer": answer, "ok": bool(ok),
            "conf": round(r["confidence"], 4), "latency_s": round(r["latency_s"], 4),
            "dist": [(o, round(p, 4)) for o, p in r["distribution"]],
        })
        mark = "OK " if ok else "XX "
        print(f"  [{qid:02d}] {mark} T{tier} {ttype:14s} pred={r['choice'][:28]:28s} "
              f"gold={answer[:28]:28s} conf={r['confidence']:.3f} {r['latency_s']*1000:7.1f}ms")

    total = len(items)
    wall = time.time() - t_start
    print("\n" + "="*78)
    print(f"ACCURACY      : {correct}/{total} = {correct/total*100:.1f}%")
    for t in (1, 2, 3):
        c, n = per_tier[t]
        if n: print(f"  Tier {t}      : {c}/{n} = {c/n*100:.1f}%")
    if lat:
        lat_sorted = sorted(lat)
        print(f"LATENCY (s)   : mean {statistics.mean(lat):.3f} | "
              f"median {statistics.median(lat):.3f} | "
              f"min {min(lat):.3f} | max {max(lat):.3f}")
        print(f"THROUGHPUT    : {1/statistics.mean(lat):.2f} decisions/sec (serial)")
        print(f"WALL TIME     : {wall:.1f}s total")
    # calibration
    confs = [r["conf"] for r in results if "conf" in r]
    if confs:
        print(f"MEAN CONF     : {statistics.mean(confs):.3f}  (accuracy {correct/total:.3f})")
    with open("results.json", "w") as f:
        json.dump({"accuracy": correct/total, "n": total, "results": results,
                   "per_tier": {str(k): v for k, v in per_tier.items()},
                   "latency_mean_s": statistics.mean(lat) if lat else None,
                   "wall_s": wall}, f, indent=1)
    print("wrote results.json")

if __name__ == "__main__":
    main()
