"""Gate-level sensitivity of EO pulse sequences to pulse-time errors.

For each gate and each error setting we compute the logical 2x2 / 4x4 block A of the
physical operation and report
  coherent infidelity   1 - |tr(U^dag A)|^2 / d^2      (includes leakage loss)
  leakage               1 - tr(A^dag A) / d            (average population lost from code space)
Systematic errors (eps multiplicative, delta additive) are deterministic.
Random per-pulse jitter (sigma) is averaged over N_SAMPLES draws of the full sequence.
"""
import os as _os
RES = _os.environ.get('EOTIME_RESULTS', '../results') + '/'
import json, time, sys
import numpy as np
from eo_sim import EOState, TimeNoise, CM_W, CM_G, _rot, synth_single

N_SAMPLES = 100
LEVELS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
CNOT = np.eye(4)[[0, 3, 2, 1]].astype(complex)      # control q1 -> target q0 (native orientation), basis |q0 q1>
SWAP = np.eye(4)[[0, 2, 1, 3]].astype(complex)

GATES = {
    'Rz(pi/4)': (1, lambda st: st.rz(0, np.pi / 4), _rot([0, 0, 1], np.pi / 4)),
    'X': (1, lambda st: st.u1(0, _rot([1, 0, 0], np.pi)), _rot([1, 0, 0], np.pi)),
    'H': (1, lambda st: st.u1(0, H), H),
    'CM(native,20p)': (2, lambda st: st.native_cm(0), CM_G),
    'CNOT(28p)': (2, lambda st: st.cnot(0), CNOT),
    'SWAP(9p)': (2, lambda st: st.swap_logical(0), SWAP),
}


def block(nq, apply, noise_kw):
    """Logical block of the operation.  A fresh TimeNoise(**noise_kw) is created for every basis column so
    that all columns see the SAME pulse-error realisation (same seed -> same draws); the block is then a
    projection of one physical unitary."""
    d = 2 ** nq
    A = np.zeros((d, d), complex)
    for b in range(d):
        st = EOState(nq, TimeNoise(**noise_kw)); st.set_logical(np.eye(d)[b]); apply(st)
        A[:, b] = st.logical_amplitudes()[0]
    return A, len(st.pulse_log), sum(t for _, _, t in st.pulse_log)


def metrics(A, U):
    d = U.shape[0]
    Un = U / np.lib.scimath.sqrt(np.linalg.det(U)) if d == 2 else U
    coh = 1 - abs(np.trace(Un.conj().T @ A)) ** 2 / d ** 2
    leak = 1 - np.trace(A.conj().T @ A).real / d
    return float(coh), float(leak)


def check_unitary():
    A = block(1, GATES['H'][1], {'sigma_add': 1e-2, 'seed': 0})[0]
    ev = np.linalg.eigvalsh(A.conj().T @ A)
    print('H block with jitter: eigenvalues of A^dag A =', ev, '(must be <= 1)')
    assert ev.max() <= 1 + 1e-9


def main():
    check_unitary()
    out = {'levels': LEVELS, 'n_samples': N_SAMPLES, 'gates': {}, 'measured_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    for name, (nq, apply, U) in GATES.items():
        A0, npulse, ttot = block(nq, apply, {})
        rec = {'n_pulses': npulse, 'total_time_swap_units': ttot, 'ideal': metrics(A0, U)}
        for kind in ['eps', 'delta']:
            for sign in [+1, -1]:
                key = f'{kind}{"+" if sign > 0 else "-"}'
                rec[key] = []
                for lv in LEVELS:
                    rec[key].append(metrics(block(nq, apply, {kind: sign * lv})[0], U))
        for kind in ['sigma_mult', 'sigma_add']:
            rec[kind] = []
            for lv in LEVELS:
                cs, ls = [], []
                for s in range(N_SAMPLES):
                    c, l = metrics(block(nq, apply, {kind: lv, 'seed': s})[0], U); cs.append(c); ls.append(l)
                rec[kind].append((float(np.mean(cs)), float(np.mean(ls)), float(np.std(cs) / np.sqrt(N_SAMPLES))))
        out['gates'][name] = rec
        print(f"{name:16s} pulses={npulse:2d} T={ttot:6.2f}  "
              f"eps+1e-2: inf={rec['eps+'][4][0]:.2e} leak={rec['eps+'][4][1]:.2e} | "
              f"delta+1e-2: inf={rec['delta+'][4][0]:.2e} leak={rec['delta+'][4][1]:.2e} | "
              f"sig_add 1e-2: inf={rec['sigma_add'][4][0]:.2e} leak={rec['sigma_add'][4][1]:.2e}", flush=True)
    json.dump(out, open(RES + 'gate_errors.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
