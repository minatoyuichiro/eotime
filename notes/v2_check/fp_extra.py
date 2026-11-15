import sys, os, json, itertools
sys.path.insert(0, os.getcwd())
import numpy as np
V = sys.argv[1]
from eo_sim import TimeNoise
import ftqc, vqe_h2
out = {}
st = ftqc.repcode_round().run(TimeNoise(eps=1e-2))
p = ftqc.physical_outcome_probs(st).reshape([2] * 5)
if V == 'old': p = p.transpose(4, 3, 2, 1, 0)        # mirror slots to compare
out['repcode P'] = p.ravel().tolist()
out['repcode metric'] = ftqc.repcode_metric(TimeNoise(eps=1e-2))
th = json.load(open('../results/vqe_h2.json'))['theta_star']; E0 = -1.857275030202381
for seed in range(2):
    rng = np.random.default_rng(100 + seed); d = [float(rng.normal(0, 3e-3)) for i in range(20)]
    pe = (lambda N: {(i, i + 1): d[i] for i in range(20)}) if V == 'old' else (lambda N: {(N - 2 - i, N - 1 - i): d[i] for i in range(N - 1)})
    out[f'bond h2 {seed}'] = vqe_h2.energy(th, {'pair_eps': pe(6)})
    out[f'bond grover {seed}'] = ftqc.grover_metric(TimeNoise(pair_eps=pe(9)))
    out[f'bond qpe {seed}'] = ftqc.qpe_metric(0.375, TimeNoise(pair_eps=pe(12)))
print(json.dumps(out))
