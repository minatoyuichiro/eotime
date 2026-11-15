"""Additional checks requested by the code review (2026-10-07).

1. VQE error decomposition: measured energy vs energy of the normalised code-space component.
2. Leakage at re-optimised parameters (H2 from stored dtheta; QAOA from stored reopt_params).
3. QAOA C4 p=1: physical ratio vs conditional code-space ratio.
4. Logical SWAP / CNOT with bond-dependent static errors (pair_eps ~ N(0,s)).
5. QPE phi=0.3: circular mean under jitter (40 draws) vs systematic error.
6. Time quantisation: nominal durations rounded to a grid, no amplitude compensation.
7. Bond-dependent quasi-static error on H2 VQE, Grover, QPE (20 seeds) vs global eps.
Writes results/extra_checks.json.
"""
import os as _os
RES = _os.environ.get('EOTIME_RESULTS', '../results') + '/'
import json, time, itertools
import numpy as np
from eo_sim import EOState, TimeNoise, synth_single
from circuit import Circuit, expect_pauli
import vqe_h2, vqe_lih, qaoa_maxcut, ftqc

out = {'measured_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
I2 = np.eye(2); Z = np.diag([1, -1]); X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]])
PM = {'I': I2, 'Z': Z, 'X': X, 'Y': Y}


def pauli_matrix(p):
    m = PM[p[0]]
    for ch in p[1:]:
        m = np.kron(m, PM[ch])
    return m


def code_energy(amp, terms):
    """<H> of the normalised code-space component."""
    nrm = np.vdot(amp, amp).real
    psi = amp / np.sqrt(nrm)
    return sum(c * np.vdot(psi, pauli_matrix(p) @ psi).real for p, c in terms.items())


# ---------------- 1. VQE decomposition
h2 = json.load(open(RES + 'vqe_h2.json')); th0 = h2['theta_star']; E0 = h2['E0']
rows = []
for lv in [1e-3, 3e-3, 1e-2, 3e-2]:
    nz = {'eps': lv}
    e_meas = vqe_h2.energy(th0, nz)
    st = vqe_h2.ansatz(th0).run(TimeNoise(**nz)); amp, leak = st.logical_amplitudes()
    e_code = code_energy(amp, vqe_h2.TERMS)
    rows.append({'eps': lv, 'dE_measured': e_meas - E0, 'dE_codespace': e_code - E0, 'leak': leak})
    print(f"H2 eps={lv:.0e}: dE_meas={e_meas-E0:.3e} dE_code={e_code-E0:.3e} leak={leak:.2e}", flush=True)
out['vqe_h2_decomp'] = rows
terms, E0l, _ = vqe_lih.hamiltonian(); groups = vqe_lih.group_terms(terms)
lih = json.load(open(RES + 'vqe_lih.json')); x0 = np.array(lih['params']); Eref = lih['E_vqe_ideal']
rows = []
for lv in [1e-3, 3e-3, 1e-2, 3e-2]:
    nz = {'eps': lv}
    e_meas = vqe_lih.energy_eo(x0, terms, groups, nz)
    st = vqe_lih.ansatz(x0).run(TimeNoise(**nz)); amp, leak = st.logical_amplitudes()
    e_code = code_energy(amp, terms)
    rows.append({'eps': lv, 'dE_measured': e_meas - Eref, 'dE_codespace': e_code - Eref, 'leak': leak})
    print(f"LiH eps={lv:.0e}: dE_meas={e_meas-Eref:.3e} dE_code={e_code-Eref:.3e} leak={leak:.2e}", flush=True)
out['vqe_lih_decomp'] = rows

# ---------------- 2. leakage at re-optimised parameters
rows = []
for kind in ['eps+', 'delta+']:
    for lv, (dE_fixed, leak_fixed), (dE_re, dth) in zip(h2['levels'], h2[kind]['fixed'], h2[kind]['reopt']):
        if lv < 1e-3:
            continue
        nz = {kind[:-1]: lv}
        st = vqe_h2.ansatz(th0 + dth).run(TimeNoise(**nz)); _, leak_re = st.logical_amplitudes()
        rows.append({'kind': kind, 'level': lv, 'leak_fixed': leak_fixed, 'leak_reopt': leak_re, 'dE_fixed': dE_fixed, 'dE_reopt': dE_re})
        print(f"H2 reopt {kind} {lv:.0e}: leak {leak_fixed:.2e} -> {leak_re:.2e}; dE {dE_fixed:.2e} -> {dE_re:.2e}", flush=True)
out['h2_reopt_leak'] = rows
rows = []
import glob
for fjson in sorted(glob.glob(RES + 'qaoa_*.json')):
    q = json.load(open(fjson))
    for key, rec in q['runs'].items():
        n, edges = qaoa_maxcut.GRAPHS[key.rsplit('_', 1)[0]]      # stored edges use the v1 labels
        vals = qaoa_maxcut.cut_values(n, edges); cmax = vals.max(); p = int(key[-1])
        for kind in ['eps+', 'delta+']:
            for lv, r in zip(q['levels'], rec[kind]):
                if 'reopt_params' not in r or lv not in (3e-3, 1e-2, 3e-2):
                    continue
                nz = {kind[:-1]: lv}
                rr = qaoa_maxcut.eo_eval(n, edges, np.array(r['reopt_params']), p, nz, cmax, vals)
                rows.append({'run': key, 'kind': kind, 'level': lv, 'leak_fixed': r['leak'], 'leak_reopt': rr[2],
                             'popt_fixed': r['popt'], 'popt_reopt': rr[1]})
                print(f"QAOA {key} reopt {kind} {lv:.0e}: leak {r['leak']:.3f} -> {rr[2]:.3f}; Popt {r['popt']:.3f} -> {rr[1]:.3f}", flush=True)
out['qaoa_reopt_leak'] = rows

# ---------------- 3. QAOA C4 p=1: physical vs conditional ratio
q = json.load(open(RES + 'qaoa_C4_p1.json')); rec = q['runs']['C4_p1']
n, edges = qaoa_maxcut.GRAPHS['C4']; vals = qaoa_maxcut.cut_values(n, edges); cmax = vals.max()
x = np.array(rec['params']); rows = []
for lv in [0, 1e-3, 3e-3, 1e-2]:
    nz = None if lv == 0 else {'eps': lv}
    c, pos = qaoa_maxcut.build(n, edges, x[:1], x[1:])
    st = c.run(None if nz is None else TimeNoise(**nz))
    zz = st.expect_all_zz([(pos[i], pos[j]) for i, j in edges]); r_phys = sum((1 - z) / 2 for z in zz) / cmax
    amp, leak = st.logical_amplitudes()
    A = (np.abs(amp) ** 2).reshape([2] * n); A = np.transpose(A, pos).reshape(-1); A = A / A.sum()
    r_cond = float((A * vals).sum() / cmax)
    rows.append({'eps': lv, 'ratio_physical': r_phys, 'ratio_codespace_conditional': r_cond, 'leak': leak})
    print(f"C4 p1 eps={lv:.0e}: ratio phys={r_phys:.4f} cond={r_cond:.4f} leak={leak:.2e}", flush=True)
out['c4p1_ratio_decomp'] = rows

# ---------------- 4. bond-dependent static errors on SWAP and CNOT
from gate_errors import block, metrics, SWAP, CNOT
rows = []
for s_ in [1e-3, 1e-2]:
    for name, apply, U in [('SWAP', lambda st: st.swap_logical(0), SWAP), ('CNOT', lambda st: st.cnot(0), CNOT)]:
        infs, lks = [], []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            pe = {(4 - i, 5 - i): float(rng.normal(0, s_)) for i in range(5)}     # bond draws from the right end (as v1)
            A = block(2, apply, {'pair_eps': pe})[0]
            c_, l_ = metrics(A, U); infs.append(c_); lks.append(l_)
        rows.append({'gate': name, 'sigma_bond': s_, 'infid_mean': float(np.mean(infs)), 'leak_mean': float(np.mean(lks))})
        print(f"bond-dependent eps ~N(0,{s_:.0e}) {name}: infid={np.mean(infs):.2e} leak={np.mean(lks):.2e}", flush=True)
out['bond_dependent_gates'] = rows

# ---------------- 5. QPE phi=0.3 circular mean under jitter
def qpe_mean(noise):
    st = ftqc.qpe(0.3).run(noise); amp, leak = st.logical_amplitudes()
    p = ftqc.counting_probs(np.abs(amp) ** 2).reshape(-1)
    return float(np.angle(np.sum(p * np.exp(2j * np.pi * np.arange(8) / 8))) / (2 * np.pi) % 1)
rows = []
for s_ in [3e-3, 1e-2]:
    ms = [qpe_mean(TimeNoise(sigma_add=s_, seed=k)) for k in range(40)]
    rows.append({'sigma_add': s_, 'mean_phase': float(np.mean(ms)), 'se': float(np.std(ms) / np.sqrt(40)),
                 'sys_plus': qpe_mean(TimeNoise(eps=s_)), 'sys_minus': qpe_mean(TimeNoise(eps=-s_))})
    print(f"QPE phi=0.3 sigma={s_:.0e}: jitter mean phase {rows[-1]['mean_phase']:.4f}+-{rows[-1]['se']:.4f}; systematic +/-: {rows[-1]['sys_plus']:.4f} / {rows[-1]['sys_minus']:.4f} (ideal {qpe_mean(TimeNoise()):.4f})", flush=True)
out['qpe_jitter_mean'] = rows

# ---------------- 6. time quantisation
rows = []
for grid in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
    nz = {'grid': grid}
    dE = vqe_h2.energy(th0, nz) - E0
    gr = ftqc.grover_metric(TimeNoise(**nz)); qp = ftqc.qpe_metric(0.375, TimeNoise(**nz))
    rows.append({'grid': grid, 'h2_dE': dE, 'grover_p': gr['p_marked'], 'grover_leak': gr['leak'], 'qpe_p': qp['p_correct']})
    print(f"grid={grid}: H2 dE={dE:+.3e}  Grover p={gr['p_marked']:.3f} leak={gr['leak']:.2e}  QPE p={qp['p_correct']:.3f}", flush=True)
out['quantisation'] = rows

# ---------------- 7. bond-dependent quasi-static error on algorithms
rows = []
for s_ in [1e-3, 3e-3, 1e-2]:
    dEs, grs, qps, lks = [], [], [], []
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        draws = [float(rng.normal(0, s_)) for i in range(20)]
        pe = lambda N: {(N - 2 - i, N - 1 - i): draws[i] for i in range(N - 1)}   # bond draws from the right end (as v1)
        dEs.append(vqe_h2.energy(th0, {'pair_eps': pe(6)}) - E0)
        g_ = ftqc.grover_metric(TimeNoise(pair_eps=pe(9))); grs.append(g_['p_marked']); lks.append(g_['leak'])
        qps.append(ftqc.qpe_metric(0.375, TimeNoise(pair_eps=pe(12)))['p_correct'])
    rows.append({'sigma_bond': s_, 'h2_dE': float(np.mean(dEs)), 'grover_p': float(np.mean(grs)), 'grover_leak': float(np.mean(lks)), 'qpe_p': float(np.mean(qps))})
    print(f"bond-dependent quasi-static N(0,{s_:.0e}): H2 dE={np.mean(dEs):.3e}  Grover p={np.mean(grs):.3f}  QPE p={np.mean(qps):.3f}", flush=True)
out['bond_dependent_algorithms'] = rows

# ---------------- 8. repetition-code failure diagnostic (physical readout) at eps = 1e-2
from ftqc import repcode_round, physical_outcome_probs
st = repcode_round().run(TimeNoise(eps=1e-2))
p = physical_outcome_probs(st).reshape([2] * 5)          # d0 a0 d1 a1 d2
flip_map = {'00': None, '10': 0, '11': 1, '01': 2}
tot = 0.0; pat = {}
for idx in itertools.product((0, 1), repeat=5):
    d0, a0, d1, a1, d2 = idx; w = float(p[idx]); bits = [d0, d1, d2]; f = flip_map[f'{a0}{a1}']
    if f is not None: bits[f] ^= 1
    if sum(bits) >= 2:
        tot += w; pat[f'd={d0}{d1}{d2} s={a0}{a1}'] = w
top = max(pat, key=pat.get)
psi = st.psi
_d, _s = top[2:5], top[8:10]                       # e.g. 'd=010 s=10' -> slots d0 a0 d1 a1 d2
for k, bit in enumerate(int(ch) for ch in (_d[0], _s[0], _d[1], _s[1], _d[2])):
    a_, b_, c_ = st.spins(k); sw = psi[st._perm(a_, b_)]; psi = 0.5 * (psi - sw) if bit == 0 else 0.5 * (psi + sw)
w = float(np.vdot(psi, psi).real); cc = st.enc.conj().T @ psi; incode = float(np.vdot(cc, cc).real)
pa0 = float(p[:, 1, :, :, :].sum()); pd1 = float(p[:, :, 1, :, :].sum()); joint = float(p[:, 1, 1, :, :].sum())
out['repcode_diagnostic'] = {'eps': 1e-2, 'p_logical': tot, 'top_pattern': top, 'top_fraction': pat[top] / tot,
                             'top_pattern_leaked_fraction': 1 - incode / w, 'P_a0': pa0, 'P_d1': pd1, 'P_joint': joint, 'P_product': pa0 * pd1}
print('repcode diagnostic', out['repcode_diagnostic'], flush=True)
json.dump(out, open(RES + 'extra_checks.json', 'w'), indent=1)
print('done')
