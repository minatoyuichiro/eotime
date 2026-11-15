"""VQE for LiH (STO-3G, R=1.6 A) on EO qubits with pulse-time errors.

Active space: freeze Li 1s (orbital 0), keep 3 spatial orbitals -> 6 spin-orbitals, 2 electrons;
symmetry-conserving Bravyi-Kitaev (OpenFermion) removes 2 qubits -> 4 qubits.
Ansatz: hardware-efficient, Ry layers + CNOT ladder (3->2,2->1,1->0), L layers (+ final Ry) = 4(L+1) params.
Qubit labels are mirrored with respect to v1 (qubit q here = v1 qubit 3-q; parameter x[l, k] acts on qubit 3-k).
Energy measured physically (basis-change pulses + singlet/triplet readout), terms grouped by measurement basis.
Experiments as in vqe_h2.py: fixed optimal params under eps/delta (+/-), re-optimised params at a few levels,
and per-pulse jitter.
"""
import os as _os
RES = _os.environ.get('EOTIME_RESULTS', '../results') + '/'
import json, time, sys
import numpy as np
from scipy.optimize import minimize
from openfermion import MolecularData, get_fermion_operator, symmetry_conserving_bravyi_kitaev, QubitOperator
from openfermion.linalg import get_sparse_operator
from openfermionpyscf import run_pyscf
from circuit import Circuit, expect_pauli, ideal_expect_pauli
from eo_sim import TimeNoise

LEVELS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
REOPT_LEVELS = [3e-3, 1e-2, 3e-2]
N_SAMPLES = 40
CHEM_ACC = 1.6e-3
NQ = 4
L = 2


def hamiltonian():
    mol = MolecularData([('Li', (0, 0, 0)), ('H', (0, 0, 1.6))], 'sto-3g', 1, 0)
    mol = run_pyscf(mol, run_fci=True)
    ferm = get_fermion_operator(mol.get_molecular_hamiltonian(occupied_indices=[0], active_indices=[1, 2, 3]))
    qop = symmetry_conserving_bravyi_kitaev(ferm, 6, 2)
    terms = {}
    for ops, coef in qop.terms.items():
        s = ['I'] * NQ
        for q, p in ops:
            s[NQ - 1 - q] = p           # mirrored qubit labels
        terms[''.join(s)] = float(coef.real)
    E0 = float(np.linalg.eigvalsh(get_sparse_operator(qop, NQ).toarray())[0])
    return terms, E0, float(mol.fci_energy)


def ansatz(x):
    c = Circuit(NQ)
    x = np.asarray(x).reshape(L + 1, NQ)
    for l in range(L):
        for k in range(NQ):
            c.ry(NQ - 1 - k, x[l, k])
        for k in range(NQ - 1):
            c.cnot(NQ - 1 - k, NQ - 2 - k)
    for k in range(NQ):
        c.ry(NQ - 1 - k, x[L, k])
    return c


def group_terms(terms):
    groups = {}
    for p, coef in terms.items():
        if p == 'I' * NQ:
            continue
        setting = ''.join('Z' if ch == 'I' else ch for ch in p)
        groups.setdefault(setting, []).append((p, coef))
    return groups


def energy_eo(x, terms, groups, noise):
    """One EO run per measurement setting; all terms compatible with the setting from that run."""
    from circuit import H, rz
    c = ansatz(x); e = terms.get('I' * NQ, 0.0)
    for setting, lst in groups.items():
        c2 = Circuit(NQ); c2.ops = list(c.ops)
        for k, ch in enumerate(setting):
            if ch == 'X':
                c2.h(k)
            elif ch == 'Y':
                c2.u(k, H @ rz(-np.pi / 2))
        st = c2.run(None if noise is None else TimeNoise(**noise))
        for p, coef in lst:
            e += coef * st.expect_z_string([k for k, ch in enumerate(p) if ch != 'I'])
    return e


def energy_ideal(x, terms):
    c = ansatz(x)
    psi = c.ideal_state()
    e = 0.0
    Z = np.diag([1, -1]).astype(complex); X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]])
    M = {'I': np.eye(2, dtype=complex), 'X': X, 'Y': Y, 'Z': Z}
    for p, coef in terms.items():
        op = M[p[0]]
        for ch in p[1:]:
            op = np.kron(op, M[ch])
        e += coef * np.vdot(psi, op @ psi).real
    return e


def leakage_at(x, noise):
    st = ansatz(x).run(TimeNoise(**noise))
    return st.logical_amplitudes()[1], len(st.pulse_log)


def main():
    terms, E0, efci = hamiltonian()
    groups = group_terms(terms)
    print(f"LiH 4q: {len(terms)} Pauli terms, {len(groups)} measurement settings; exact E0={E0:.8f} (FCI full={efci:.6f})", flush=True)
    rng = np.random.default_rng(0); best = None
    for s in range(12):
        x0 = rng.uniform(-np.pi, np.pi, NQ * (L + 1))
        r = minimize(lambda x: energy_ideal(x, terms), x0, method='BFGS', options={'gtol': 1e-9})
        if best is None or r.fun < best.fun:
            best = r
    x0 = best.x
    e_eo = energy_eo(x0, terms, groups, None)
    leak0, npulse = leakage_at(x0, {})
    print(f"ideal HEA(L={L}) min={best.fun:.8f} (gap to exact {best.fun-E0:.2e})  EO ideal={e_eo:.8f}  pulses={npulse}", flush=True)
    out = {'E0': E0, 'E_vqe_ideal': float(best.fun), 'params': x0.tolist(), 'levels': LEVELS, 'chem_acc': CHEM_ACC,
           'n_terms': len(terms), 'n_settings': len(groups), 'pulses_ansatz': npulse,
           'measured_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    Eref = float(best.fun)
    for kind in ['eps', 'delta']:
        for sign in [+1, -1]:
            key = f'{kind}{"+" if sign > 0 else "-"}'
            rows = []
            for lv in LEVELS:
                nz = {kind: sign * lv}
                e = energy_eo(x0, terms, groups, nz); lk = leakage_at(x0, nz)[0]
                row = {'dE': e - Eref, 'leak': lk}
                if lv in REOPT_LEVELS and sign > 0:
                    rr = minimize(lambda y: energy_eo(y, terms, groups, nz), x0, method='Nelder-Mead',
                                  options={'xatol': 1e-3, 'fatol': 1e-6, 'maxiter': 150})
                    row['dE_reopt'] = rr.fun - Eref
                rows.append(row)
                print(f"  {key} {lv:.0e}: dE={row['dE']:+.3e} leak={lk:.2e}" + (f" | reopt dE={row['dE_reopt']:+.3e}" if 'dE_reopt' in row else ''), flush=True)
            out[key] = rows
            json.dump(out, open(RES + 'vqe_lih.json', 'w'), indent=1)
    for kind in ['sigma_add', 'sigma_mult']:
        rows = []
        for lv in LEVELS:
            es, ls = [], []
            for s in range(N_SAMPLES):
                nz = {kind: lv, 'seed': s}
                es.append(energy_eo(x0, terms, groups, nz) - Eref); ls.append(leakage_at(x0, nz)[0])
            rows.append({'dE': float(np.mean(es)), 'se': float(np.std(es) / np.sqrt(N_SAMPLES)), 'leak': float(np.mean(ls))})
            print(f"  {kind} {lv:.0e}: dE={rows[-1]['dE']:+.3e} +- {rows[-1]['se']:.1e} leak={rows[-1]['leak']:.2e}", flush=True)
        out[kind] = rows
        json.dump(out, open(RES + 'vqe_lih.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
