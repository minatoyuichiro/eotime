import json, sys, numpy as np
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2])); worst = 0
for k in a:
    x = np.ravel(np.array(a[k] if not isinstance(a[k], dict) else list(a[k].values()), float))
    y = np.ravel(np.array(b[k] if not isinstance(b[k], dict) else list(b[k].values()), float))
    d = np.abs(x - y); rel = np.max(np.where(np.abs(x) > 1e-12, d / np.maximum(np.abs(x), 1e-300), d))
    worst = max(worst, rel)
    if rel > 1e-9: print(f"DIFF {k:35s} {rel:.1e} {x} {y}")
print(f"{len(a)} quantities compared, worst relative difference {worst:.1e}")
