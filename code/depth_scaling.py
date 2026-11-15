"""How does the error accumulate with circuit depth?  (coherent N^2 vs incoherent N)

Circuit: k repetitions of (CNOT(1,0) CNOT(1,0)) = identity on 2 qubits, k = 1..24  -> 40 k + 8 pulses (the W corrections between CNOTs merge away).
Also k repetitions of the 9-SWAP logical swap applied twice (identity).  (H H pairs are merged away by the compiler.)
Metric: 1 - |<psi_ideal|psi>|^2 on |0+> (control q1 in |+>; state infidelity, includes leakage) and leakage, at
eps=1e-3 (systematic) and sigma_add=1e-3 (jitter, 100 samples).  Also QPE phase-shift vs eps.
"""
import os as _os
RES = _os.environ.get('EOTIME_RESULTS', '../results') + '/'
import json, time
import numpy as np
from circuit import Circuit
from eo_sim import TimeNoise
from ftqc import qpe, counting_probs

KS = [1, 2, 4, 8, 16, 24]
LV = 1e-3
NS = 100


def infid(c, noise, state0):
    st = c.run(noise, state0)
    amp, leak = st.logical_amplitudes()
    ideal = c.ideal_state(state0)
    return 1 - abs(np.vdot(ideal, amp)) ** 2, leak, len(st.pulse_log)


def main():
    out = {'ks': KS, 'level': LV, 'n_samples': NS, 'measured_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    plus0 = np.array([1, 1, 0, 0], complex) / np.sqrt(2)     # |0>_q0 |+>_q1
    for name, mk, s0 in [('cnot_pairs', _cnot_chain, plus0), ('swap_pairs', _swap_chain, plus0)]:
        rows = []
        for k in KS:
            c = mk(k)
            e_sys, l_sys, npulse = infid(c, TimeNoise(eps=LV), s0)
            js = [infid(c, TimeNoise(sigma_add=LV, seed=s), s0) for s in range(NS)]
            rows.append({'k': k, 'pulses': npulse, 'sys_infid': e_sys, 'sys_leak': l_sys,
                         'jit_infid': float(np.mean([j[0] for j in js])), 'jit_leak': float(np.mean([j[1] for j in js]))})
            print(f"{name} k={k:2d} pulses={npulse:4d}  eps=1e-3: infid={e_sys:.3e} leak={l_sys:.3e} | sigma=1e-3: infid={rows[-1]['jit_infid']:.3e} leak={rows[-1]['jit_leak']:.3e}", flush=True)
        out[name] = rows
    # QPE phase shift: mean estimated phase (circular) vs eps for phi=0.3
    rows = []
    for eps in [-3e-2, -1e-2, -3e-3, -1e-3, 0, 1e-3, 3e-3, 1e-2, 3e-2]:
        st = qpe(0.3).run(TimeNoise(eps=eps))
        amp, leak = st.logical_amplitudes()
        p = counting_probs(np.abs(amp) ** 2).reshape(-1)
        ph = np.angle(np.sum(p * np.exp(2j * np.pi * np.arange(8) / 8))) / (2 * np.pi) % 1
        rows.append({'eps': eps, 'mean_phase': float(ph), 'p': p.tolist(), 'leak': leak})
        print(f"QPE phi=0.3 eps={eps:+.0e}: mean phase={ph:.4f} (ideal 0.3)  P(010)={p[2]:.3f} P(011)={p[3]:.3f} leak={leak:.2e}", flush=True)
    out['qpe_shift'] = rows
    json.dump(out, open(RES + 'depth_scaling.json', 'w'), indent=1)


def _cnot_chain(k):
    c = Circuit(2)
    for _ in range(k):
        c.cnot(1, 0).cnot(1, 0)
    return c


def _swap_chain(k):
    c = Circuit(2)
    for _ in range(k):
        c.swap(0, 1).swap(0, 1)
    return c


if __name__ == '__main__':
    main()
