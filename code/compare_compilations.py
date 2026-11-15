"""Tables for the grid, floating-point rounding and J trade-off experiments as ranges over three compilations:
search-based single-qubit synthesis on machine 1 (results/), the same on a second machine (results/*_suneo.json), and
fixed single-qubit sequences (results_fixed/).  Writes paper/tables/grid_range.tex, fp_range.tex, deco_mid_range.tex."""
import os, json
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); T = os.path.join(B, 'paper', 'tables')
L = lambda p: json.load(open(os.path.join(B, p)))
def sci(x):
    if x == 0: return '0'
    m, e = f'{x:.1e}'.split('e'); e = int(e)
    return f'{m}' if e == 0 else rf'{m}\times10^{{{e}}}'
def rng(vals, fmt):
    lo, hi = min(vals), max(vals)
    return f'${fmt(lo)}$--$\\,{fmt(hi)}$' if fmt is sci else f'{fmt(lo)}--{fmt(hi)}'
f3 = lambda x: f'{x:.3f}'
def mha(x):   # Table 9 rows in mHa
    v = x*1e3
    return f'{v:.0f}' if v >= 10 else (f'{v:.1f}' if v >= 1 else f'{v:.2g}')
# grid
A = [L('results/grid_scan.json'), L('results/grid_scan_suneo.json'), L('results_fixed/grid_scan.json')]
rows = []
for i in range(len(A[0]['rows'])):
    r = [a['rows'][i] for a in A]
    rows.append(f"{r[0]['grid']} & {rng([x['h2_dE'] for x in r], sci)} & {rng([x['grover_p'] for x in r], f3)} & {rng([x['qpe_p'] for x in r], f3)} \\\\")
open(os.path.join(T, 'grid_range.tex'), 'w').write('\n'.join(rows))
# fp
A = [L('results/fp_rounding.json'), L('results/fp_rounding_suneo.json'), L('results_fixed/fp_rounding.json')]
rows = []
for i in range(len(A[0]['rows'])):
    r = [a['rows'][i] for a in A]
    if r[0]['bits'] not in (4, 5, 6, 7, 8, 11, 16): continue
    rows.append(f"{r[0]['bits']} & ${sci(r[0]['rel_step'])}$ & {rng([x['h2_dE'] for x in r], sci)} & {rng([x['grover_p'] for x in r], f3)} & {rng([x['qpe_p'] for x in r], f3)} & {rng([x['cnot_infid'] for x in r], sci)} \\\\")
open(os.path.join(T, 'fp_range.tex'), 'w').write('\n'.join(rows))
# deco middle (grid) and bottom (no grid)
A = [L('results/decoherence.json'), L('results/decoherence_suneo.json'), L('results_fixed/decoherence.json')]
for key, fn in [('J_tradeoff', 'deco_mid_range.tex'), ('J_tradeoff_nogrid', 'deco_bot_range.tex')]:
    rows = {'h2': [], 'gr': [], 'qp': []}
    for i in range(len(A[0][key])):
        r = [a[key][i] for a in A]
        rows['h2'].append(rng([x['h2_dE'] for x in r], mha)); rows['gr'].append(rng([x['grover_p'] for x in r], f3)); rows['qp'].append(rng([x['qpe_p'] for x in r], f3))
    open(os.path.join(T, fn), 'w').write(f"$\\Hz$ $\\Delta E$ [mHa] & " + ' & '.join(rows['h2']) + " \\\\\nGrover $P$ & " + ' & '.join(rows['gr']) + " \\\\\nQPE $P$ & " + ' & '.join(rows['qp']) + " \\\\")
print(open(os.path.join(T, 'grid_range.tex')).read()); print(open(os.path.join(T, 'fp_range.tex')).read()); print(open(os.path.join(T, 'deco_mid_range.tex')).read())
