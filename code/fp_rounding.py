"""Floating-point rounding of pulse durations: every nominal duration is rounded to a binary mantissa of b bits
(relative resolution 2^-(b-1), i.e. b = 8 is bfloat16-like, b = 11 is float16-like, b = 24 float32).
Deterministic; H2 VQE energy, Grover success, QPE success, plus the CNOT gate infidelity."""
import os as _os, json, time
RES = _os.environ.get('EOTIME_RESULTS', '../results') + '/'
import numpy as np
from eo_sim import TimeNoise
import vqe_h2, ftqc
from gate_errors import block, metrics, CNOT

h2 = json.load(open(RES + 'vqe_h2.json')); th0 = h2['theta_star']; E0 = h2['E0']
out = {'measured_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'rows': []}
for b in [4, 5, 6, 7, 8, 9, 10, 11, 12, 16]:
    nz = {'fp_bits': b}
    dE = vqe_h2.energy(th0, nz) - E0
    g = ftqc.grover_metric(TimeNoise(**nz)); q = ftqc.qpe_metric(0.375, TimeNoise(**nz))
    A = block(2, lambda st: st.cnot(0), nz)[0]; cinf, cleak = metrics(A, CNOT)
    row = {'bits': b, 'rel_step': 2.0 ** (-(b - 1)), 'h2_dE': dE, 'grover_p': g['p_marked'], 'grover_leak': g['leak'],
           'qpe_p': q['p_correct'], 'cnot_infid': cinf, 'cnot_leak': cleak}
    out['rows'].append(row)
    print(f"b={b:2d} (step {row['rel_step']:.1e}): H2 dE={dE:+.2e}  Grover {g['p_marked']:.3f} leak {g['leak']:.1e}  QPE {q['p_correct']:.3f}  CNOT infid {cinf:.1e} leak {cleak:.1e}", flush=True)
json.dump(out, open(RES + 'fp_rounding.json', 'w'), indent=1)
print('done')
