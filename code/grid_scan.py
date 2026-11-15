"""Duration-grid scan (Table 'grid') as a standalone script: every nominal duration rounded to a multiple of
the grid (units of t_SWAP), no other error.  H2 VQE energy, Grover success, QPE(3/8) success.  Deterministic."""
import os as _os, json, time
RES = _os.environ.get('EOTIME_RESULTS', '../results') + '/'
import numpy as np
from eo_sim import TimeNoise, SYNTH_MODE
import vqe_h2, ftqc
h2 = json.load(open(RES + 'vqe_h2.json')); th0 = h2['theta_star']; E0 = h2['E0']
out = {'measured_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'synth_mode': SYNTH_MODE, 'rows': []}
for grid in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
    nz = {'grid': grid}
    dE = vqe_h2.energy(th0, nz) - E0
    g = ftqc.grover_metric(TimeNoise(**nz)); q = ftqc.qpe_metric(0.375, TimeNoise(**nz))
    out['rows'].append({'grid': grid, 'h2_dE': dE, 'grover_p': g['p_marked'], 'grover_leak': g['leak'], 'qpe_p': q['p_correct']})
    print(f"grid={grid}: H2 dE={dE:+.3e}  Grover p={g['p_marked']:.3f} leak={g['leak']:.2e}  QPE p={q['p_correct']:.3f}", flush=True)
json.dump(out, open(RES + 'grid_scan.json', 'w'), indent=1); print('done')
