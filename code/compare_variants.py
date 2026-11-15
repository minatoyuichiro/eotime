"""Compare the main results with the variants: fixed single-qubit decomposition (results_fixed),
Chadwick CZ entangler (results_cz), Chadwick leakage-controlled CZ (results_lccz).
Writes results/variants.md and paper/tables/variants.tex."""
import os, json, numpy as np
import analyze
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
VARIANTS = [('main', 'results'), ('fixed 1q', 'results_fixed'), ('CZ (Chadwick)', 'results_cz'), ('LCCZ (Chadwick)', 'results_lccz')]
TASKS = ['VQE H2', 'VQE LiH', 'QAOA C4_p2', 'QAOA prism6_p2', 'grover3', 'qpe_phi0.375', 'repcode_d3']
NAMES = {'VQE H2': r'VQE H$_2$', 'VQE LiH': 'VQE LiH', 'QAOA C4_p2': r'QAOA $C_4$ $p{=}2$', 'QAOA prism6_p2': r'QAOA prism $p{=}2$',
         'grover3': 'Grover', 'qpe_phi0.375': r'QPE $3/8$', 'repcode_d3': 'Rep.\ code'}

rows_md = ['| 対象 | ' + ' | '.join(f'{v} (パルス / ε 閾値 / 1e-2 での漏れ)' for v, _ in VARIANTS) + ' |', '|---|' + '---|' * len(VARIANTS)]
rows_tex = []
gate_rows = []
for t in TASKS:
    cells_md, cells_tex = [], []
    for vname, vdir in VARIANTS:
        analyze.R = os.path.join(BASE, vdir)
        try:
            cs = analyze.curves()
        except Exception as ex:
            cs = []
        c = next((c for c in cs if c['task'] == t and c['kind'] == 'eps+'), None)
        if c is None:
            cells_md.append('—'); cells_tex.append('--- & --- & ---'); continue
        thr = analyze.threshold(c['levels'], c['err'], c['thr']); i = c['levels'].index(1e-2)
        cells_md.append(f"{c['pulses']} / {thr:.1e} / {c['leak'][i]:.1e}")
        milli = lambda x: f'{x*1e3:.0f}' if x*1e3 >= 10 else f'{x*1e3:.1f}'   # table in units of 1e-3
        thr_tex = milli(thr) if thr != np.inf else '$>100$'
        cells_tex.append(f"{c['pulses']} & {thr_tex} & {milli(c['leak'][i])}")
    rows_md.append(f'| {t} | ' + ' | '.join(cells_md) + ' |')
    rows_tex.append(f'{NAMES[t]} & ' + ' & '.join(cells_tex) + r' \\')
# gate-level CNOT comparison
for vname, vdir in VARIANTS:
    p = os.path.join(BASE, vdir, 'gate_errors.json')
    if not os.path.exists(p):
        continue
    g = json.load(open(p)); rec = g['gates']['CNOT(28p)']; i = g['levels'].index(1e-2)
    gate_rows.append(f"| CNOT [{vname}] | pulses {rec['n_pulses']} T={rec['total_time_swap_units']:.1f} | eps 1e-2: infid {rec['eps+'][i][0]:.2e} leak {rec['eps+'][i][1]:.2e} ({rec['eps+'][i][1]/rec['eps+'][i][0]:.0%}) | sigma_add 1e-2: infid {rec['sigma_add'][i][0]:.2e} leak {rec['sigma_add'][i][1]:.2e} |")
tex_gate = []
for vname, vdir in VARIANTS:
    p = os.path.join(BASE, vdir, 'gate_errors.json')
    if not os.path.exists(p):
        continue
    g = json.load(open(p)); rec = g['gates']['CNOT(28p)']; i = g['levels'].index(1e-2)
    def sci(x):
        m, e = f'{x:.1e}'.split('e'); return rf'${m}\times10^{{{int(e)}}}$'
    tex_gate.append(f"{vname} & {rec['n_pulses']} & {rec['total_time_swap_units']:.1f} & {sci(rec['eps+'][i][0])} & {sci(rec['eps+'][i][1])} & {rec['eps+'][i][1]/rec['eps+'][i][0]*100:.0f}\\% & {sci(rec['sigma_add'][i][0])} & {sci(rec['sigma_add'][i][1])} \\\\")
open(os.path.join(BASE, 'paper', 'tables', 'variants_cnot.tex'), 'w').write('\n'.join(tex_gate))
md = '# 変種の比較（自動生成: compare_variants.py）\n\n' + '\n'.join(rows_md) + '\n\n## CNOT のゲート単位\n\n' + '\n'.join(gate_rows) + '\n'
open(os.path.join(BASE, 'results', 'variants.md'), 'w').write(md); print(md)
open(os.path.join(BASE, 'paper', 'tables', 'variants.tex'), 'w').write('\n'.join(rows_tex))
