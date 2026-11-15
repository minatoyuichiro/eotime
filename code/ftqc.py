"""FTQC-flavoured circuits on EO qubits with pulse-time errors.

1. Grover search, 3 qubits, marked |101>, 2 iterations (ideal success 0.9453).  CCZ via Z-string exponentials.
2. QPE with 3 counting qubits on a phase gate P(2 pi phi), target |1> on qubit 0.  phi = 3/8 (exact) and 0.3 (inexact).
3. Bit-flip repetition code d=3: data slots 0,2,4 in |000>, ancillas 1,3; one syndrome round (4 CNOT),
(qubit labels mirrored with respect to v1, together with the |S>_12|up>_3 encoding)
   ideal syndrome decoding; logical flip probability and leakage.
Grover and QPE metrics from the code-space amplitudes (leaked population counts as failure); the repetition code
is read physically (singlet/triplet of every pair, leaked blocks read as 1) and also reported on the code space.
"""
import os as _os
RES = _os.environ.get('EOTIME_RESULTS', '../results') + '/'
import json, time, sys
import numpy as np
from circuit import Circuit, H
from eo_sim import TimeNoise

LEVELS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
N_SAMPLES = 40


def probs_logical(c, noise):
    st = c.run(noise)
    amp, leak = st.logical_amplitudes()
    return np.abs(amp) ** 2, leak, len(st.pulse_log)


# ---------------- Grover
def grover(marked='101', iters=2):
    n = 3
    c = Circuit(n)
    for k in range(n):
        c.h(k)
    for _ in range(iters):
        for k, b in enumerate(marked):      # oracle: phase flip on |marked>
            if b == '0':
                c.x(k)
        c.ccz(2, 1, 0)
        for k, b in enumerate(marked):
            if b == '0':
                c.x(k)
        for k in range(n):                  # diffusion
            c.h(k); c.x(k)
        c.ccz(2, 1, 0)
        for k in range(n):
            c.x(k); c.h(k)
    return c


def grover_metric(noise):
    p, leak, npulse = probs_logical(grover(), noise)
    return {'p_marked': float(p[0b101]), 'leak': leak, 'pulses': npulse}


# ---------------- QPE
def qpe(phi, m=3):
    """target qubit 0 prepared in |1>, counting qubits m, m-1, ..., 1 (qubit m most significant)."""
    q = lambda k: m - k                      # k-th counting qubit
    c = Circuit(m + 1)
    c.x(0)
    for k in range(m):
        c.h(q(k))
    for k in range(m):                       # controlled-U^(2^k) on counting qubit k (bit-reversed input to the swap-less QFT^dag)
        c.cp(q(k), 0, 2 * np.pi * phi * 2 ** k)
    # inverse QFT on the counting qubits without the bit-reversal swaps; output then has qubit m = MSB
    for j in range(m - 1, -1, -1):
        for k in range(m - 1, j, -1):
            c.cp(q(k), q(j), -np.pi / 2 ** (k - j))
        c.h(q(j))
    return c


def counting_probs(p, m=3):
    """Distribution of the m-bit estimate (index with the most significant bit first) from the 2^(m+1) probabilities."""
    p = p.reshape([2] * (m + 1)).sum(axis=0)           # trace out the target (qubit 0)
    return p.transpose(list(range(m - 1, -1, -1)))     # axes: qubit m (MSB), ..., qubit 1


def qpe_metric(phi, noise, m=3):
    p, leak, npulse = probs_logical(qpe(phi, m), noise)
    p = counting_probs(p, m)
    best = int(round(phi * 2 ** m)) % 2 ** m        # nearest m-bit estimate (qubit 0 = MSB)
    return {'p_correct': float(p.reshape(-1)[best]), 'leak': leak, 'pulses': npulse}


# ---------------- repetition code
def repcode_round():
    c = Circuit(5)            # d0 a0 d1 a1 d2
    c.cnot(4, 3).cnot(2, 3).cnot(2, 1).cnot(0, 1)
    return c


def physical_outcome_probs(st):
    """Joint probabilities of singlet(0)/triplet(1) readout on the pair of every encoded qubit.
    P_T = (1 + SWAP_pair)/2, P_S = (1 - SWAP_pair)/2 on the pair (3k,3k+1).  Leaked (S=3/2) blocks read as triplet."""
    n = st.n
    branches = [(st.psi, ())]
    for k in range(n):
        a, b, c = st.spins(k)
        perm = st._perm(a, b)
        new = []
        for psi, bits in branches:
            sw = psi[perm]
            new.append((0.5 * (psi - sw), bits + (0,)))
            new.append((0.5 * (psi + sw), bits + (1,)))
        branches = new
    p = np.zeros(2 ** n)
    for psi, bits in branches:
        p[int(''.join(map(str, bits)), 2)] = float(np.vdot(psi, psi).real)
    return p


def repcode_metric(noise):
    """Physical readout: every pair (data and ancilla) is read as singlet/triplet; the syndrome is
    decoded from the ancilla readout and the logical value from the majority of the data readout."""
    c = repcode_round()
    st = c.run(noise)
    amp, leak = st.logical_amplitudes(); npulse = len(st.pulse_log)
    p = physical_outcome_probs(st)
    P = p.reshape([2] * 5)
    p_log = 0.0; p_syn = {}
    for s0 in (0, 1):
        for s1 in (0, 1):
            sub = P[:, s0, :, s1, :]          # data bits d0 d1 d2
            w = float(sub.sum()); p_syn[f'{s0}{s1}'] = w
            flip = {'00': None, '10': 0, '11': 1, '01': 2}[f'{s0}{s1}']
            for b in range(8):
                bits = [(b >> 2) & 1, (b >> 1) & 1, b & 1]
                if flip is not None:
                    bits[flip] ^= 1
                if sum(bits) >= 2:            # majority says logical 1 -> logical error
                    p_log += float(sub[(b >> 2) & 1, (b >> 1) & 1, b & 1])
    # raw (no correction) logical error = majority of data without using syndrome
    p_raw = sum(float(P.reshape([2] * 5)[d0, :, d1, :, d2].sum()) for d0 in (0, 1) for d1 in (0, 1) for d2 in (0, 1) if d0 + d1 + d2 >= 2)
    # code-space-only variant (leakage discarded), kept for comparison with the earlier run
    pc = (np.abs(amp) ** 2).reshape([2] * 5)
    p_log_code = 0.0
    for s0 in (0, 1):
        for s1 in (0, 1):
            sub = pc[:, s0, :, s1, :]
            flip = {'00': None, '10': 0, '11': 1, '01': 2}[f'{s0}{s1}']
            for b in range(8):
                bits = [(b >> 2) & 1, (b >> 1) & 1, b & 1]
                if flip is not None:
                    bits[flip] ^= 1
                if sum(bits) >= 2:
                    p_log_code += float(sub[(b >> 2) & 1, (b >> 1) & 1, b & 1])
    return {'p_logical': p_log, 'p_logical_codespace': p_log_code, 'p_raw_majority': p_raw,
            'p_nontrivial_syndrome': 1 - p_syn['00'], 'leak': leak, 'pulses': npulse}


TASKS = {
    'grover3': lambda nz: grover_metric(nz),
    'qpe_phi0.375': lambda nz: qpe_metric(0.375, nz),
    'qpe_phi0.3': lambda nz: qpe_metric(0.3, nz),
    'repcode_d3': lambda nz: repcode_metric(nz),
}


def main(names=None):
    out = {'levels': LEVELS, 'n_samples': N_SAMPLES, 'measured_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'tasks': {}}
    for name in (names or TASKS):
        f = TASKS[name]
        ideal = f(TimeNoise())
        print(f"== {name}: ideal {ideal}", flush=True)
        rec = {'ideal': ideal}
        for kind in ['eps', 'delta']:
            for sign in [+1, -1]:
                key = f'{kind}{"+" if sign > 0 else "-"}'
                rec[key] = [f(TimeNoise(**{kind: sign * lv})) for lv in LEVELS]
                print(f"  {key}: " + ' '.join(f"{lv:.0e}:{list(r.values())[0]:.4f}/{r['leak']:.1e}" for lv, r in zip(LEVELS, rec[key])), flush=True)
        for kind in ['sigma_add', 'sigma_mult']:
            rows = []
            for lv in LEVELS:
                rs = [f(TimeNoise(**{kind: lv}, seed=s)) for s in range(N_SAMPLES)]
                agg = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
                agg['se'] = float(np.std([list(r.values())[0] for r in rs]) / np.sqrt(N_SAMPLES))
                rows.append(agg)
            rec[kind] = rows
            print(f"  {kind}: " + ' '.join(f"{lv:.0e}:{list(r.values())[0]:.4f}/{r['leak']:.1e}" for lv, r in zip(LEVELS, rows)), flush=True)
        out['tasks'][name] = rec
        json.dump(out, open(RES + f'ftqc_{"_".join(names or ["all"])}.json', 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1].split(',') if len(sys.argv) > 1 else None)
