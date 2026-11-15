"""VQE for H2 (STO-3G, R=0.735 A, parity mapping -> 2 qubits) on EO qubits with pulse-time errors.

Hamiltonian (electronic part, Hartree): standard 2-qubit parity-mapped H2 used in Qiskit tutorials.
Ansatz: Ry(theta) q1 ; CNOT(1,0) ; X q0   ->  cos(theta/2)|10> + sin(theta/2)|01>  (exact ground state reachable)
(qubit labels mirrored with respect to v1 together with the |S>_12|up>_3 encoding; strings are 'q0 q1')
Energy = sum_P c_P <P>, each <P> measured physically (basis-change pulses + singlet/triplet readout of every pair).
Leakage out of the code space reads as triplet (Z=-1).

Experiments
  A. fixed noiseless-optimal theta*, systematic eps / delta scan (both signs): Delta E and leakage
  B. same but theta re-optimised under the systematic error (what a real VQE loop would do)
  C. per-pulse jitter (sigma_add, sigma_mult): mean Delta E over N_SAMPLES draws at theta*
Reference: chemical accuracy 1.6e-3 Ha.
"""
import os as _os
RES = _os.environ.get('EOTIME_RESULTS', '../results') + '/'
import json, time, sys
import numpy as np
from scipy.optimize import minimize_scalar
from circuit import Circuit, expect_pauli, ideal_expect_pauli
from eo_sim import TimeNoise

TERMS = {'II': -1.052373245772859, 'ZI': 0.39793742484318045, 'IZ': -0.39793742484318045,
         'ZZ': -0.01128010425623538, 'XX': 0.18093119978423156}
LEVELS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
N_SAMPLES = 100
CHEM_ACC = 1.6e-3


def exact_ground():
    I = np.eye(2); Z = np.diag([1, -1]); X = np.array([[0, 1], [1, 0]])
    M = {'I': I, 'Z': Z, 'X': X}
    Hm = sum(c * np.kron(M[p[0]], M[p[1]]) for p, c in TERMS.items())
    return np.linalg.eigvalsh(Hm)[0]


def ansatz(theta):
    return Circuit(2).ry(1, theta).cnot(1, 0).x(0)


def energy(theta, noise=None):
    c = ansatz(theta); e = TERMS['II']
    for p, coef in TERMS.items():
        if p == 'II':
            continue
        nz = None if noise is None else TimeNoise(**noise)
        e += coef * expect_pauli(c, p, nz)
    return e


def energy_ideal(theta):
    c = ansatz(theta)
    return TERMS['II'] + sum(coef * ideal_expect_pauli(c, p) for p, coef in TERMS.items() if p != 'II')


def leakage_at(theta, noise):
    st = ansatz(theta).run(TimeNoise(**noise))
    return st.logical_amplitudes()[1], len(st.pulse_log)


def main():
    E0 = exact_ground()
    res = minimize_scalar(energy_ideal, bounds=(-np.pi, np.pi), method='bounded', options={'xatol': 1e-10})
    th0 = res.x
    print(f"exact E0={E0:.8f}  ideal VQE min={res.fun:.8f} at theta={th0:.6f}  (EO ideal: {energy(th0):.8f})")
    out = {'E0': E0, 'theta_star': th0, 'levels': LEVELS, 'chem_acc': CHEM_ACC,
           'pulses_ansatz': leakage_at(th0, {})[1],
           'measured_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    # A/B systematic
    for kind in ['eps', 'delta']:
        for sign in [+1, -1]:
            key = f'{kind}{"+" if sign > 0 else "-"}'
            rowsA, rowsB = [], []
            for lv in LEVELS:
                nz = {kind: sign * lv}
                eA = energy(th0, nz)
                leak = leakage_at(th0, nz)[0]
                rB = minimize_scalar(lambda t: energy(t, nz), bounds=(th0 - 1.0, th0 + 1.0), method='bounded',
                                     options={'xatol': 1e-6})
                # "true" energy of the re-optimised state is what the noisy device reports; also record
                rowsA.append((eA - E0, leak)); rowsB.append((rB.fun - E0, rB.x - th0))
                print(f"  {key} {lv:.0e}: dE_fixed={eA-E0:+.3e} leak={leak:.2e} | reopt dE={rB.fun-E0:+.3e} dtheta={rB.x-th0:+.4f}", flush=True)
            out[key] = {'fixed': rowsA, 'reopt': rowsB}
    # C jitter
    for kind in ['sigma_add', 'sigma_mult']:
        rows = []
        for lv in LEVELS:
            es, ls = [], []
            for s in range(N_SAMPLES):
                nz = {kind: lv, 'seed': s}
                es.append(energy(th0, nz) - E0); ls.append(leakage_at(th0, nz)[0])
            rows.append((float(np.mean(es)), float(np.std(es) / np.sqrt(N_SAMPLES)), float(np.mean(ls))))
            print(f"  {kind} {lv:.0e}: dE={rows[-1][0]:+.3e} +- {rows[-1][1]:.1e} leak={rows[-1][2]:.2e}", flush=True)
        out[kind] = rows
    json.dump(out, open(RES + 'vqe_h2.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
