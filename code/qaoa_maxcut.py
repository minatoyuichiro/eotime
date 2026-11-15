"""QAOA MaxCut on EO qubits (linear chain of encoded qubits) with pulse-time errors.

Graphs: ring C4 (needs routing for the closing edge), path P6 (no routing), prism (6 nodes, 9 edges, routing).
Routing: logical qubits live on a linear chain; a non-adjacent edge is served by moving one endpoint with
logical SWAPs (9 physical SWAP pulses each) and leaving it there (mapping tracked).
Metrics: approximation ratio r = <C>/C_max (C measured physically from <Z_i Z_j>), and P(optimal cut)
from the code-space amplitudes (leakage counts as failure).
Experiments: fixed noiseless-optimal angles under systematic eps/delta (both signs); re-optimised angles
under systematic error; per-pulse jitter averaged over samples.
"""
import os as _os
RES = _os.environ.get('EOTIME_RESULTS', '../results') + '/'
import json, time, sys, itertools
import numpy as np
from scipy.optimize import minimize
from circuit import Circuit, expect_pauli, ideal_expect_pauli
from eo_sim import TimeNoise

LEVELS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
N_SAMPLES = 40

# vertex labels mirrored with respect to v1 (vertex v here = v1 vertex n-1-v); edge order and orientation kept
GRAPHS = {
    'C4': (4, [(3, 2), (2, 1), (1, 0), (3, 0)]),
    'P6': (6, [(5, 4), (4, 3), (3, 2), (2, 1), (1, 0)]),
    'prism6': (6, [(5, 4), (4, 3), (5, 3), (2, 1), (1, 0), (2, 0), (5, 2), (4, 1), (3, 0)]),
}


def cut_values(n, edges):
    vals = np.zeros(2 ** n)
    for b in range(2 ** n):
        bits = [(b >> (n - 1 - k)) & 1 for k in range(n)]
        vals[b] = sum(bits[i] != bits[j] for i, j in edges)
    return vals


def build(n, edges, gammas, betas):
    """Returns (Circuit, mapping logical->physical slot at the end)."""
    c = Circuit(n)
    for k in range(n):
        c.h(k)
    pos = list(range(n))            # pos[logical] = physical slot
    for g, b in zip(gammas, betas):
        for (i, j) in edges:
            pi, pj = pos[i], pos[j]
            # move j's qubit next to i's
            while abs(pos[i] - pos[j]) > 1:
                step = 1 if pos[j] < pos[i] else -1
                a = pos[j]; bslot = a + step
                c.swap(a, bslot)
                # update mapping: whoever sits at bslot moves to a
                other = pos.index(bslot); pos[other] = a; pos[j] = bslot
            c.zz(pos[i], pos[j], g)
        for k in range(n):
            c.rx(pos[k], 2 * b)
    return c, pos


def ideal_cost(n, edges, params, p):
    c, pos = build(n, edges, params[:p], params[p:])
    tot = 0.0
    for (i, j) in edges:
        s = ['I'] * n; s[pos[i]] = 'Z'; s[pos[j]] = 'Z'
        tot += (1 - ideal_expect_pauli(c, ''.join(s))) / 2
    return tot


def eo_eval(n, edges, params, p, noise, cmax, vals):
    c, pos = build(n, edges, params[:p], params[p:])
    st = c.run(None if noise is None else TimeNoise(**noise))
    zz = st.expect_all_zz([(pos[i], pos[j]) for i, j in edges])     # all cost terms are Z-type: one run
    tot = sum((1 - z) / 2 for z in zz)
    amp, leak = st.logical_amplitudes()
    # un-permute amplitudes: physical slot -> logical
    A = np.abs(amp.reshape([2] * n)) ** 2
    A = np.transpose(A, pos).reshape(-1)        # axis s (slot) -> logical index
    popt = float(A[vals == cmax].sum())
    return tot / cmax, popt, leak, len(st.pulse_log)


def optimise_ideal(n, edges, p, seeds=20):
    best = None
    rng = np.random.default_rng(0)
    for s in range(seeds):
        x0 = rng.uniform(0, np.pi, 2 * p)
        r = minimize(lambda x: -ideal_cost(n, edges, x, p), x0, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-9, 'maxiter': 4000})
        if best is None or r.fun < best.fun:
            best = r
    return best.x, -best.fun


def main(graph_names=None, ps=(1, 2)):
    out = {'levels': LEVELS, 'n_samples': N_SAMPLES, 'measured_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'runs': {}}
    for gname in (graph_names or GRAPHS):
        n, edges = GRAPHS[gname]
        vals = cut_values(n, edges); cmax = vals.max()
        for p in ps:
            x, cval = optimise_ideal(n, edges, p)
            r0, popt0, leak0, npulse = eo_eval(n, edges, x, p, None, cmax, vals)
            key = f'{gname}_p{p}'
            print(f"== {key}: ideal r={cval/cmax:.4f}  EO ideal r={r0:.4f} Popt={popt0:.4f} leak={leak0:.1e} pulses={npulse}", flush=True)
            rec = {'n': n, 'edges': edges, 'cmax': float(cmax), 'params': x.tolist(), 'ideal_ratio': float(cval / cmax),
                   'ideal_popt': popt0, 'pulses': npulse}
            for kind in ['eps', 'delta']:
                for sign in [+1, -1]:
                    k2 = f'{kind}{"+" if sign > 0 else "-"}'
                    rows = []
                    for lv in LEVELS:
                        nz = {kind: sign * lv}
                        r, po, lk, _ = eo_eval(n, edges, x, p, nz, cmax, vals)
                        row = {'ratio': r, 'popt': po, 'leak': lk}
                        if lv >= 1e-3:   # re-optimise only where it matters (cost)
                            rr = minimize(lambda y: -eo_eval(n, edges, y, p, nz, cmax, vals)[0], x, method='Nelder-Mead',
                                          options={'xatol': 1e-3, 'fatol': 1e-5, 'maxiter': 120})
                            rres = eo_eval(n, edges, rr.x, p, nz, cmax, vals)
                            row['reopt_ratio'] = rres[0]; row['reopt_popt'] = rres[1]; row['reopt_params'] = rr.x.tolist()
                        rows.append(row)
                        print(f"  {k2} {lv:.0e}: r={r:.4f} Popt={po:.4f} leak={lk:.2e}" + (f" | reopt r={row['reopt_ratio']:.4f} Popt={row['reopt_popt']:.4f}" if 'reopt_ratio' in row else ''), flush=True)
                    rec[k2] = rows
            for kind in ['sigma_add', 'sigma_mult']:
                rows = []
                for lv in LEVELS:
                    rs, pos_, lks = [], [], []
                    for s in range(N_SAMPLES):
                        r, po, lk, _ = eo_eval(n, edges, x, p, {kind: lv, 'seed': s}, cmax, vals)
                        rs.append(r); pos_.append(po); lks.append(lk)
                    rows.append({'ratio': float(np.mean(rs)), 'ratio_se': float(np.std(rs) / np.sqrt(N_SAMPLES)),
                                 'popt': float(np.mean(pos_)), 'leak': float(np.mean(lks))})
                    print(f"  {kind} {lv:.0e}: r={rows[-1]['ratio']:.4f}+-{rows[-1]['ratio_se']:.4f} Popt={rows[-1]['popt']:.4f} leak={rows[-1]['leak']:.2e}", flush=True)
                rec[kind] = rows
            out['runs'][key] = rec
            json.dump(out, open(RES + f'qaoa_{"_".join(graph_names or ["all"])}_p{"".join(map(str, ps))}.json', 'w'), indent=1)


if __name__ == '__main__':
    gs = sys.argv[1].split(',') if len(sys.argv) > 1 else None
    ps = tuple(int(v) for v in sys.argv[2].split(',')) if len(sys.argv) > 2 else (1, 2)
    main(gs, ps)
