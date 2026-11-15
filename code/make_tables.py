"""Generate paper/tables/*.tex from results/*.json (do not edit the .tex tables by hand)."""
import os, json, numpy as np
from analyze import curves, threshold, KINDS, load, R

T = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'paper', 'tables')
os.makedirs(T, exist_ok=True)
cs = curves()
NAMES = {'VQE H2': r'VQE H$_2$ (2q)', 'VQE LiH': r'VQE LiH (4q)',
         'QAOA C4_p1': r'QAOA $C_4$, $p=1$', 'QAOA C4_p2': r'QAOA $C_4$, $p=2$',
         'QAOA P6_p1': r'QAOA $P_6$, $p=1$', 'QAOA P6_p2': r'QAOA $P_6$, $p=2$',
         'QAOA prism6_p1': r'QAOA prism, $p=1$', 'QAOA prism6_p2': r'QAOA prism, $p=2$',
         'grover3': r'Grover (3q, 2 iter.)', 'qpe_phi0.375': r'QPE, $\phi=3/8$', 'qpe_phi0.3': r'QPE, $\phi=0.3$',
         'repcode_d3': r'Repetition code $d=3$'}
ORDER = ['VQE H2', 'VQE LiH', 'QAOA C4_p1', 'QAOA C4_p2', 'QAOA P6_p1', 'QAOA P6_p2', 'QAOA prism6_p1', 'QAOA prism6_p2',
         'grover3', 'qpe_phi0.375', 'qpe_phi0.3', 'repcode_d3']


def get(t, k):
    return next((c for c in cs if c['task'] == t and c['kind'] == k), None)


def sci(x, digits=1):
    if x == np.inf:
        return r'$>10^{-1}$'
    if x == 0:
        return '$0$'
    m, e = f'{x:.{digits}e}'.split('e')
    e = int(e)
    if e == 0:
        return f'${m}$'
    return rf'${m}\times10^{{{e}}}$'


def num(x):
    """signed number: plain if 0.01<=|x|<1000 else sci"""
    if x == 0:
        return '$0$'
    if 0.01 <= abs(x) < 1000:
        return f'${x:+.2g}$' if abs(x) < 10 else f'${x:+.0f}$'
    s = sci(abs(x))
    return s.replace('$', '$' + ('+' if x > 0 else '-'), 1)


# --- Table: gate-level errors at x = 1e-2
g = load('gate_errors.json')
rows = []
for name, rec in g['gates'].items():
    i = g['levels'].index(1e-2)
    rows.append(f"{name.replace('(native,20p)', ' (native)').replace('(28p)', '').replace('(9p)', '')} & {rec['n_pulses']} & {rec['total_time_swap_units']:.2f} & "
                f"{rec['eps+'][i][0]:.1e} & {rec['eps+'][i][1]:.1e} & {rec['delta+'][i][0]:.1e} & {rec['delta+'][i][1]:.1e} & "
                f"{rec['sigma_add'][i][0]:.1e} & {rec['sigma_add'][i][1]:.1e} \\\\")
# simpler formatting: rebuild cleanly
rows = []
for name, rec in g['gates'].items():
    i = g['levels'].index(1e-2)
    nm = {'Rz(pi/4)': r'$R_z(\pi/4)$', 'X': '$X$', 'H': '$H$', 'CM(native,20p)': r'native $C(\hat n\cdot\sigma)$',
          'CNOT(28p)': 'CNOT', 'SWAP(9p)': 'logical SWAP'}[name]
    rows.append(f"{nm} & {rec['n_pulses']} & {rec['total_time_swap_units']:.2f} & {sci(rec['eps+'][i][0])} & {sci(max(rec['eps+'][i][1], 1e-16))} & "
                f"{sci(rec['delta+'][i][0])} & {sci(max(rec['delta+'][i][1], 1e-16))} & {sci(rec['sigma_add'][i][0])} & {sci(max(rec['sigma_add'][i][1], 1e-16))} \\\\")
open(os.path.join(T, 'gates.tex'), 'w').write('\n'.join(rows))

# --- Table: thresholds
rows = []
for t in ORDER:
    c0 = get(t, 'eps+')
    if c0 is None:
        continue
    cells = [sci(threshold(get(t, k)['levels'], get(t, k)['err'], get(t, k)['thr'])) for k in ['eps+', 'eps-', 'delta+', 'sigma_add', 'sigma_mult']]
    rows.append(f"{NAMES[t]} & {c0['pulses']} & " + ' & '.join(cells) + r' \\')
open(os.path.join(T, 'thresholds.tex'), 'w').write('\n'.join(rows))

# --- Table: linear / quadratic decomposition at x = 1e-3
rows = []
for t in ORDER:
    c1, c2 = get(t, 'eps+'), get(t, 'eps-')
    if c1 is None:
        continue
    i = c1['levels'].index(1e-3); x = 1e-3
    ep, em = c1['err'][i], c2['err'][i]
    b = (ep - em) / (2 * x); a = (ep + em) / (2 * x * x); xc = abs(b) / a if a > 0 else float('inf')
    ideal = f"{c1['ideal_popt']:.3f}" if 'ideal_popt' in c1 else (f"{c1['ideal']:.4g}" if 'ideal' in c1 and t != 'repcode_d3' else ('0' if t == 'repcode_d3' else r'$E_0$'))
    rows.append(f"{NAMES[t]} & {ideal} & {num(b)} & {sci(a)} & {sci(xc) if xc > 1e-7 else '$<10^{-7}$'} \\\\")
open(os.path.join(T, 'linquad.tex'), 'w').write('\n'.join(rows))

# --- Table: values at x = 1e-2 (error / leakage / re-optimised)
rows = []
for t in ORDER:
    c = get(t, 'eps+')
    if c is None:
        continue
    i = c['levels'].index(1e-2)
    cells = []
    for k in ['eps+', 'delta+', 'sigma_add']:
        cc = get(t, k)
        if t.startswith('VQE') or t.startswith('repcode'):
            s = f"{sci(cc['err'][i])} / {sci(cc['leak'][i])}"
            if cc.get('reopt') is not None and not np.isnan(cc['reopt'][i]):
                s += f" / {sci(cc['reopt'][i])}"
        else:
            s = f"{cc['err'][i]:+.3f} / {cc['leak'][i]:.3f}"
            if cc.get('reopt') is not None and not np.isnan(cc['reopt'][i]):
                s += f" / {cc['reopt'][i]:+.3f}"
        cells.append(s)
    rows.append(f"{NAMES[t]} & " + ' & '.join(cells) + r' \\')
open(os.path.join(T, 'at1e-2.tex'), 'w').write('\n'.join(rows))

# --- Table: depth scaling
d = load('depth_scaling.json')
rows = []
for r1, r2 in zip(d['cnot_pairs'], d['swap_pairs']):
    rows.append(f"{r1['k']} & {r1['pulses']} & {sci(r1['sys_infid'])} & {sci(r1['jit_infid'])} & {r2['pulses']} & {sci(max(r2['sys_infid'], 1e-16))} & {sci(r2['jit_infid'])} \\\\")
open(os.path.join(T, 'depth.tex'), 'w').write('\n'.join(rows))

# --- QPE shift
line1 = ' & '.join(num(r['eps']) if r['eps'] != 0 else '$0$' for r in d['qpe_shift']) + r' \\'
line2 = r'$\bar\phi$ & ' + ' & '.join(f"{r['mean_phase']:.3f}" for r in d['qpe_shift']) + r' \\'
open(os.path.join(T, 'qpe_shift.tex'), 'w').write(line1 + '\n' + line2)

# --- floating-point rounding of durations
fp = load('fp_rounding.json')
if fp:
    rows = []
    for r in fp['rows']:
        if r['bits'] in (4, 5, 6, 7, 8, 11, 16):
            rows.append(f"{r['bits']} & {sci(r['rel_step'])} & {sci(r['h2_dE'])} & {r['grover_p']:.3f} & {r['qpe_p']:.3f} & {sci(max(r['cnot_infid'], 1e-12))} \\\\")
    open(os.path.join(T, 'fp_rounding.tex'), 'w').write('\n'.join(rows))

# --- macros with headline numbers
v = load('vqe_h2.json'); l = load('vqe_lih.json'); f = load('ftqc_all.json')
mac = []
def m(name, val):
    mac.append(f"\\newcommand{{\\{name}}}{{{val}}}")
m('HtwoThrEps', sci(threshold(v['levels'], [r[0] for r in v['eps+']['fixed']], v['chem_acc'])).strip('$'))
m('LiHThrEps', sci(threshold(l['levels'], [r['dE'] for r in l['eps+']], l['chem_acc'])).strip('$'))
m('GroverThrEps', sci(threshold(f['levels'], [f['tasks']['grover3']['ideal']['p_marked'] - r['p_marked'] for r in f['tasks']['grover3']['eps+']], 0.05)).strip('$'))
m('QPEThrEps', sci(threshold(f['levels'], [1 - r['p_correct'] for r in f['tasks']['qpe_phi0.375']['eps+']], 0.05)).strip('$'))
open(os.path.join(T, 'macros.tex'), 'w').write('\n'.join(mac) + '\n')
print('\n'.join(mac))
for fn in ['gates', 'thresholds', 'linquad', 'at1e-2', 'depth', 'qpe_shift']:
    print('==', fn); print(open(os.path.join(T, fn + '.tex')).read())
