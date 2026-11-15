"""Run in eotime/code of either version: prints JSON of metrics; version given as argv[1] ('old'|'new')."""
import sys, json, os
sys.path.insert(0, os.getcwd())
import numpy as np
V = sys.argv[1]; which = sys.argv[2].split(',')
from eo_sim import TimeNoise
out = {}
NZ = [('eps', {'eps': 1e-2}), ('epsm', {'eps': -3e-3}), ('delta', {'delta': 1e-2}), ('sadd', {'sigma_add': 1e-2, 'seed': 3}),
      ('smul', {'sigma_mult': 3e-3, 'seed': 5}), ('grid', {'grid': 0.05}), ('fp', {'fp_bits': 6}), ('sh', {'sigma_h': 1e-3, 'seed': 2})]
if 'gates' in which:
    import gate_errors as G
    for name, (nq, ap, U) in G.GATES.items():
        for k, nz in NZ[:5] + [('fp', {'fp_bits': 6})]:
            out[f'gate {name} {k}'] = G.metrics(G.block(nq, ap, nz)[0], U)
    for seed in range(2):
        rng = np.random.default_rng(seed); d = [float(rng.normal(0, 1e-2)) for i in range(5)]
        pe = {(i, i + 1): d[i] for i in range(5)} if V == 'old' else {(4 - i, 5 - i): d[i] for i in range(5)}
        for name in ['CNOT(28p)', 'SWAP(9p)']:
            nq, ap, U = G.GATES[name]; out[f'gate {name} bond{seed}'] = G.metrics(G.block(nq, ap, {'pair_eps': pe})[0], U)
if 'h2' in which:
    import vqe_h2, json as J
    th = J.load(open('../results/vqe_h2.json'))['theta_star']
    for k, nz in NZ: out[f'h2 {k}'] = vqe_h2.energy(th, nz)
if 'lih' in which:
    import vqe_lih, json as J
    x = np.array(J.load(open('../results/vqe_lih.json'))['params'])
    terms, E0, _ = vqe_lih.hamiltonian(); groups = vqe_lih.group_terms(terms)
    for k, nz in NZ[:5]: out[f'lih {k}'] = vqe_lih.energy_eo(x, terms, groups, nz)
if 'qaoa' in which:
    import qaoa_maxcut as Q, json as J
    for f, key in [('qaoa_C4_p1', 'C4_p1'), ('qaoa_C4_p2', 'C4_p2'), ('qaoa_P6_p1', 'P6_p1'), ('qaoa_prism6_p1', 'prism6_p1'), ('qaoa_prism6_p2', 'prism6_p2')]:
        r = J.load(open(f'../results/{f}.json'))['runs'][key]; n, e = Q.GRAPHS[key.split('_')[0]]; p = int(key[-1])
        vals = Q.cut_values(n, e); cm = vals.max()
        for k, nz in [('ideal', None)] + NZ[:5]:
            out[f'qaoa {key} {k}'] = Q.eo_eval(n, e, np.array(r['params']), p, nz, cm, vals)[:3]
if 'ftqc' in which:
    import ftqc as F
    for t, f in F.TASKS.items():
        for k, nz in [('ideal', {})] + NZ:
            out[f'{t} {k}'] = f(TimeNoise(**nz))
if 'depth' in which:
    import depth_scaling as D
    s0 = np.array([1, 0, 1, 0], complex) / np.sqrt(2) if V == 'old' else np.array([1, 1, 0, 0], complex) / np.sqrt(2)
    for k in [1, 4]:
        out[f'depth cnot {k}'] = D.infid(D._cnot_chain(k), TimeNoise(eps=1e-3), s0)
        out[f'depth cnot jit {k}'] = D.infid(D._cnot_chain(k), TimeNoise(sigma_add=1e-3, seed=1), s0)
        out[f'depth swap {k}'] = D.infid(D._swap_chain(k), TimeNoise(eps=1e-3), s0)
print(json.dumps(out, default=lambda o: o if not isinstance(o, np.generic) else o.item()))
