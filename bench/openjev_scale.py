#!/usr/bin/env python3
"""Scale test: openjev shared-prefix method at 20 questions."""
import time
from openjev_method import OpenJevClient

c = OpenJevClient()
state = ("A customer says a password reset succeeded, but every login attempt still "
         "returns account locked. Two unlock emails were requested and neither arrived.")

print("=== batch 1: 20 questions, ONE warm then 20 branches ===")
qs = [(f"Is aspect {i} a problem?", ["no", "yes"]) for i in range(20)]
res = c.decide_batch(state, qs)
lat = [r["latency_s"] for r in res]
warm = res[0]["warm_s"]
print("  warm           : %.2f s (paid once)" % warm)
print("  branches total : %.2f s for %d -> %.2f s each" % (sum(lat), len(lat), sum(lat)/len(lat)))
print("  min / max      : %.2f s / %.2f s" % (min(lat), max(lat)))
print("  TOTAL          : %.1f s -> %.2f s per question" % (warm + sum(lat), res[0]["amortised_s"]))

print()
print("=== batch 2: 20 more questions, same state (prefix already hot) ===")
qs2 = [(f"Second pass aspect {i}?", ["no", "yes"]) for i in range(20)]
t0 = time.time()
res2 = c.decide_batch(state, qs2, warm=False)
lat2 = [r["latency_s"] for r in res2]
print("  branches total : %.2f s -> %.2f s each" % (sum(lat2), sum(lat2)/len(lat2)))

print()
print("=== batch 3: DIFFERENT state (prefix must be rebuilt) ===")
state3 = "The build failed on a missing dependency and CI is red."
qs3 = [(f"Third pass aspect {i}?", ["no", "yes"]) for i in range(20)]
res3 = c.decide_batch(state3, qs3)
lat3 = [r["latency_s"] for r in res3]
print("  warm           : %.2f s" % res3[0]["warm_s"])
print("  branches total : %.2f s -> %.2f s each" % (sum(lat3), sum(lat3)/len(lat3)))
