"""Quasi-static Zeeman-gradient (hyperfine) dephasing/leakage and the J trade-off.

Model: each spin i has a static detuning h_i ~ N(0, sigma_h) in units of J; during every pulse of duration t
(units of t_SWAP) the state evolves under pi * sum_i h_i S_z^i * t in addition to the exchange.  For a singlet
this gives T2* = t_SWAP / (pi * sigma_h) (Gaussian decay of the singlet-T0 coherence at the 1/e point).
Experiments:
  A. sigma_h scan (40 draws) for H2 VQE, Grover, QPE(3/8): error vs T2*/t_SWAP.
  B. J trade-off: physical grid 0.5 ns, T2* = 3.5 us (Weinstein 2023), J/h from 10 MHz to 1 GHz:
     t_SWAP = h/2J; grid/t_SWAP and sigma_h = t_SWAP/(pi T2*) both follow from J.  Grover and H2 vs J.
"""
import os as _os, json, time, sys
RES = _os.environ.get('EOTIME_RESULTS', '../results') + '/'
import numpy as np
from eo_sim import TimeNoise
import vqe_h2, ftqc

N_DRAWS = 40
h2 = json.load(open(RES + 'vqe_h2.json')); th0 = h2['theta_star']; E0 = h2['E0']
out = {'measured_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'n_draws': N_DRAWS}


def run(noise_kw, draws):
    dE, gr, qp, lk = [], [], [], []
    for s in range(draws):
        kw = dict(noise_kw, seed=s)
        dE.append(vqe_h2.energy(th0, kw) - E0)
        g = ftqc.grover_metric(TimeNoise(**kw)); gr.append(g['p_marked']); lk.append(g['leak'])
        qp.append(ftqc.qpe_metric(0.375, TimeNoise(**kw))['p_correct'])
    return {'h2_dE': float(np.mean(dE)), 'h2_se': float(np.std(dE) / np.sqrt(draws)), 'grover_p': float(np.mean(gr)),
            'grover_leak': float(np.mean(lk)), 'qpe_p': float(np.mean(qp))}


rows = []
for sh in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
    r = run({'sigma_h': sh}, N_DRAWS); r['sigma_h'] = sh; r['T2star_over_tswap'] = 1 / (np.pi * sh); rows.append(r)
    print(f"sigma_h={sh:.0e} (T2*/tSWAP={1/(np.pi*sh):.0f}): H2 dE={r['h2_dE']:.3e}  Grover p={r['grover_p']:.3f} leak={r['grover_leak']:.2e}  QPE p={r['qpe_p']:.3f}", flush=True)
out['sigma_h_scan'] = rows

rows = []
T2 = 3.5e-6; grid_phys = 0.5e-9
for JMHz in [10, 20, 50, 100, 200, 500, 1000]:
    tsw = 1 / (2 * JMHz * 1e6)
    sh = tsw / (np.pi * T2); grid = grid_phys / tsw
    r = run({'sigma_h': sh, 'grid': grid}, N_DRAWS); r.update({'J_MHz': JMHz, 't_swap_ns': tsw * 1e9, 'sigma_h': sh, 'grid': grid}); rows.append(r)
    print(f"J/h={JMHz} MHz: tSWAP={tsw*1e9:.1f} ns grid={grid:.3f} sigma_h={sh:.1e}: H2 dE={r['h2_dE']:.3e}  Grover p={r['grover_p']:.3f}  QPE p={r['qpe_p']:.3f}", flush=True)
out['J_tradeoff'] = rows

rows = []
for JMHz in [10, 20, 50, 100, 200, 500, 1000]:
    tsw = 1 / (2 * JMHz * 1e6); sh = tsw / (np.pi * T2)
    r = run({'sigma_h': sh}, N_DRAWS); r.update({'J_MHz': JMHz, 't_swap_ns': tsw * 1e9, 'sigma_h': sh}); rows.append(r)
    print(f"[amplitude control, no grid] J/h={JMHz} MHz: sigma_h={sh:.1e}: H2 dE={r['h2_dE']:.3e}  Grover p={r['grover_p']:.3f}  QPE p={r['qpe_p']:.3f}", flush=True)
out['J_tradeoff_nogrid'] = rows
json.dump(out, open(RES + 'decoherence.json', 'w'), indent=1)
print('done')
