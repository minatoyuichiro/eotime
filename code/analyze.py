"""Collect results/*.json into tables (results/summary.md) and figures (results/fig/*.png).

Per task we define an "error" that is 0 for the ideal device and grows with the timing error:
  gates      : coherent infidelity (and leakage separately)
  VQE        : Delta E [Ha]  (threshold: chemical accuracy 1.6e-3)
  QAOA       : ideal_ratio - ratio  (threshold 0.01)
  Grover/QPE : ideal_p - p          (threshold 0.05)
  repcode    : p_logical, leakage   (threshold 1e-3)
For each error kind we report the small-error power-law exponent (fit over the levels where error < 0.1 of
the ideal metric), the prefactor a in error = a * x^2, and the threshold level by log-log interpolation.
"""
import json, os, glob
import numpy as np

PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
KINDS = ['eps+', 'eps-', 'delta+', 'delta-', 'sigma_add', 'sigma_mult']
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def curves():
    """Returns list of dict(task, kind, levels, err, leak, pulses, thr)."""
    out = []
    g = load('gate_errors.json')
    if g:
        for name, rec in g['gates'].items():
            for k in KINDS:
                rows = rec[k]
                out.append(dict(task=f'gate:{name}', group='gate', kind=k, levels=g['levels'],
                                err=[r[0] for r in rows], leak=[r[1] for r in rows], pulses=rec['n_pulses'], thr=1e-3))
    v = load('vqe_h2.json')
    if v:
        for k in KINDS:
            rows = v[k]
            if k.startswith('sigma'):
                err = [r[0] for r in rows]; leak = [r[2] for r in rows]; reopt = None
            else:
                err = [r[0] for r in rows['fixed']]; leak = [r[1] for r in rows['fixed']]; reopt = [r[0] for r in rows['reopt']]
            out.append(dict(task='VQE H2', group='vqe', kind=k, levels=v['levels'], err=err, leak=leak, reopt=reopt,
                            pulses=v['pulses_ansatz'], thr=v['chem_acc']))
    v = load('vqe_lih.json')
    if v:
        for k in KINDS:
            if k not in v:
                continue
            rows = v[k]
            err = [r['dE'] for r in rows]; leak = [r['leak'] for r in rows]
            reopt = [r.get('dE_reopt', np.nan) for r in rows] if not k.startswith('sigma') else None
            out.append(dict(task='VQE LiH', group='vqe', kind=k, levels=v['levels'], err=err, leak=leak, reopt=reopt,
                            pulses=v['pulses_ansatz'], thr=v['chem_acc']))
    runs = {}
    for f in sorted(glob.glob(os.path.join(R, 'qaoa_*.json'))):
        q = json.load(open(f))
        for key, rec in q['runs'].items():
            if key not in runs or q['measured_at'] > runs[key][0]:
                runs[key] = (q['measured_at'], q['levels'], rec)
    for key, (_, levels_, rec) in runs.items():
        q = {'levels': levels_}
        if True:
            for k in KINDS:
                rows = rec[k]
                # main error: relative drop of P(optimal cut) (unbiased, from code-space amplitudes);
                # the measured approximation ratio can exceed the ideal because leaked qubits read as |1>.
                p0 = rec['ideal_popt']
                err = [(p0 - r['popt']) / p0 for r in rows]; leak = [r['leak'] for r in rows]
                reopt = [(p0 - r['reopt_popt']) / p0 if 'reopt_popt' in r else np.nan for r in rows] if not k.startswith('sigma') else None
                out.append(dict(task=f'QAOA {key}', group='qaoa', kind=k, levels=q['levels'], err=err, leak=leak, reopt=reopt,
                                pulses=rec['pulses'], thr=0.05, ideal=1.0,
                                ratio_drop=[rec['ideal_ratio'] - r['ratio'] for r in rows], ideal_ratio=rec['ideal_ratio'], ideal_popt=p0))
    f = load('ftqc_all.json')
    if f:
        for name, rec in f['tasks'].items():
            m = 'p_marked' if 'grover' in name else ('p_correct' if 'qpe' in name else 'p_logical')
            ideal = rec['ideal'][m]
            for k in KINDS:
                rows = rec[k]
                if m == 'p_logical':
                    err = [r[m] for r in rows]; thr = 1e-3
                else:
                    err = [ideal - r[m] for r in rows]; thr = 0.05
                out.append(dict(task=name, group='ftqc', kind=k, levels=f['levels'], err=err, leak=[r['leak'] for r in rows],
                                pulses=rec['ideal']['pulses'], thr=thr, ideal=ideal))
    return out


def fit_exponent(levels, err, ideal_scale=1.0):
    x = np.array(levels); y = np.array(err, float)
    m = (y > 1e-13) & (y < 0.1 * max(ideal_scale, 1e-9)) & (x <= 3e-3)
    if m.sum() < 3:
        return np.nan, np.nan
    p = np.polyfit(np.log10(x[m]), np.log10(y[m]), 1)
    a2 = np.median(y[m] / x[m] ** 2)
    return p[0], a2


def threshold(levels, err, thr):
    """Smallest scanned interval in which the signed error crosses thr upwards; log-log interpolation
    when both ends are positive, linear otherwise.  inf if never reached; x[0] if already above at x[0]."""
    x = np.array(levels); y = np.array(err, float)
    if y[0] >= thr:
        return x[0]
    for i in range(1, len(x)):
        if y[i] >= thr > y[i - 1]:
            if y[i - 1] > 0:
                lx = np.interp(np.log10(thr), np.log10([y[i - 1], y[i]]), np.log10([x[i - 1], x[i]]))
                return 10 ** lx
            return float(np.interp(thr, [y[i - 1], y[i]], [x[i - 1], x[i]]))
    return np.inf


def write_summary(cs):
    lines = ['# 結果まとめ（自動生成: analyze.py）', '',
             '誤差の定義: ゲート=不忠実度、VQE=ΔE[Ha]、QAOA=最適カット確率の相対低下（近似比は漏れの読み出しバイアスで理想値を上回ることがあるため副指標）、Grover/QPE=理想成功確率−成功確率、反復符号=論理誤り率。',
             'しきい値: VQE 1.6 mHa、QAOA P(opt) 5% 低下、Grover/QPE 0.05、反復符号 1e-3、ゲート 1e-3。', '',
             '## 誤差の種類ごとの「しきい値に達する時間誤差」（SWAP 時間 π/J を 1 とした単位）', '',
             '| 対象 | パルス数 | ' + ' | '.join(KINDS) + ' |', '|---|---|' + '---|' * len(KINDS)]
    tasks = []
    for c in cs:
        if c['task'] not in tasks:
            tasks.append(c['task'])
    for t in tasks:
        row = [t, str(next(c['pulses'] for c in cs if c['task'] == t))]
        for k in KINDS:
            c = next((c for c in cs if c['task'] == t and c['kind'] == k), None)
            if c is None:
                row.append('—'); continue
            th = threshold(c['levels'], c['err'], c['thr'])
            row.append('>1e-1' if th == np.inf else f'{th:.1e}')
        lines.append('| ' + ' | '.join(row) + ' |')
    lines += ['', '## 小さい誤差での冪（log-log の傾き、誤差 ≤ 3e-3 の範囲）と二乗係数 a（誤差 = a·x²）', '',
              '| 対象 | パルス数 | ' + ' | '.join(f'{k} 冪 / a' for k in KINDS) + ' |', '|---|---|' + '---|' * len(KINDS)]
    for t in tasks:
        row = [t, str(next(c['pulses'] for c in cs if c['task'] == t))]
        for k in KINDS:
            c = next((c for c in cs if c['task'] == t and c['kind'] == k), None)
            if c is None:
                row.append('—'); continue
            e, a = fit_exponent(c['levels'], c['err'], c.get('ideal', 1.0) if c['group'] != 'vqe' else 1.0)
            row.append('—' if np.isnan(e) else f'{e:.2f} / {a:.2g}')
        lines.append('| ' + ' | '.join(row) + ' |')
    lines += ['', '## 誤差 1e-2 のときの値（誤差 / 漏れ / 再最適化後の誤差）', '',
              '| 対象 | ' + ' | '.join(KINDS) + ' |', '|---|' + '---|' * len(KINDS)]
    for t in tasks:
        row = [t]
        for k in KINDS:
            c = next((c for c in cs if c['task'] == t and c['kind'] == k), None)
            if c is None:
                row.append('—'); continue
            i = c['levels'].index(1e-2)
            s = f"{c['err'][i]:.2e} / {c['leak'][i]:.1e}"
            if c.get('reopt') is not None and not np.isnan(c['reopt'][i]):
                s += f" / {c['reopt'][i]:.2e}"
            if 'ratio_drop' in c:
                s += f" (近似比低下 {c['ratio_drop'][i]:+.3f})"
            row.append(s)
        lines.append('| ' + ' | '.join(row) + ' |')
    lines += ['', '## 系統誤差の 1 次と 2 次の分離（ε=±1e-3 の対で評価）', '',
              '誤差 e(±x) から 1 次係数 b = (e(+x) − e(−x)) / 2x、2 次係数 a = (e(+x) + e(−x)) / 2x²。',
              '交差点 x_c = |b|/a より小さい誤差では 1 次（符号に依る）が、大きい誤差では 2 次が支配する。',
              '揺らぎは平均 0 なので 1 次項は消え、常に 2 次（σ 列の冪を参照）。', '',
              '| 対象 | 理想値（指標） | 乗法 ε: b / a / x_c | 加法 δ: b / a / x_c | 判定 |', '|---|---|---|---|---|']
    for t in tasks:
        row = [t]
        cp = next((c for c in cs if c['task'] == t and c['kind'] == 'eps+'), None)
        row.append('' if cp is None else (f"P(opt)={cp['ideal_popt']:.3f}" if 'ideal_popt' in cp else (f"{cp['ideal']:.4g}" if 'ideal' in cp else 'F=1 / E_exact')))
        verdict = ''
        for kp, km in [('eps+', 'eps-'), ('delta+', 'delta-')]:
            c1 = next((c for c in cs if c['task'] == t and c['kind'] == kp), None)
            c2 = next((c for c in cs if c['task'] == t and c['kind'] == km), None)
            if c1 is None or c2 is None:
                row.append('—'); continue
            i = c1['levels'].index(1e-3); x = 1e-3
            ep, em = c1['err'][i], c2['err'][i]
            b = (ep - em) / (2 * x); a = (ep + em) / (2 * x * x)
            xc = abs(b) / a if a > 0 else np.inf
            row.append(f'{b:+.2g} / {a:.2g} / {xc:.1e}')
            if kp == 'eps+':
                verdict = '2 次（1 次項 ≈ 0）' if xc < 1e-4 else ('1 次が 1e-3 まで効く' if xc > 1e-3 else '混合')
        row.append(verdict)
        lines.append('| ' + ' | '.join(row) + ' |')
    open(os.path.join(R, 'summary.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def figures(cs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(R, 'fig'), exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': 0.25, 'lines.linewidth': 1.6})
    kinds_plot = ['eps+', 'delta+', 'sigma_add', 'sigma_mult']
    labels = {'eps+': 'systematic ×(1+ε)', 'eps-': 'systematic ×(1−ε)', 'delta+': 'systematic +δ', 'delta-': 'systematic −δ',
              'sigma_add': 'jitter, additive σ', 'sigma_mult': 'jitter, multiplicative σ'}
    # Fig 1: per task, SIGNED error vs level (symlog y so that improvements (<0) are visible, no floor)
    YLAB = {'VQE H2': 'E_VQE − E_exact [Ha]', 'VQE LiH': 'E_VQE − E_exact [Ha]',
            'grover3': '0.9453 − P(marked)', 'qpe_phi0.375': '1 − P(correct 3-bit phase)',
            'qpe_phi0.3': '0.578 − P(nearest 3-bit phase)', 'repcode_d3': 'P(logical flip after correction)'}
    tasks = []
    for c in cs:
        if c['task'] not in tasks and c['group'] != 'gate':
            tasks.append(c['task'])
    ncol = 3; nrow = int(np.ceil(len(tasks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow), squeeze=False)
    for ax, t in zip(axes.ravel(), tasks):
        for i, k in enumerate(kinds_plot):
            c = next((c for c in cs if c['task'] == t and c['kind'] == k), None)
            if c is None:
                continue
            ax.plot(c['levels'], np.array(c['err'], float), '-o', ms=3.5, color=PAL[i], label=labels[k])
            if k == 'eps+':
                c2 = next((c for c in cs if c['task'] == t and c['kind'] == 'eps-'), None)
                if c2 is not None:
                    ax.plot(c2['levels'], np.array(c2['err'], float), '--s', ms=3, color=PAL[i], alpha=0.7, label=labels['eps-'])
        ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=1e-6, linscale=0.5)
        ax.axhline(0, color='#52514e', lw=0.6)
        ax.axhline(next(c['thr'] for c in cs if c['task'] == t), color='#52514e', lw=0.8, ls=':')
        ax.set_title(f"{t}  ({next(c['pulses'] for c in cs if c['task'] == t)} pulses)", fontsize=9)
        ax.set_xlabel('timing error x (units of SWAP time)')
        ylab = YLAB.get(t, 'P(opt)_ideal − P(opt), relative' if t.startswith('QAOA') else 'error')
        ax.set_ylabel(ylab, fontsize=8)
    for ax in axes.ravel()[len(tasks):]:
        ax.axis('off')
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle('Signed deviation from the ideal result vs pulse-timing error (y: symlog, negative = better than ideal; dotted: threshold)', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(R, 'fig', 'error_vs_timing.png'), dpi=160); plt.close(fig)
    # Fig 2: gate level
    g = [c for c in cs if c['group'] == 'gate']
    gnames = []
    for c in g:
        if c['task'] not in gnames:
            gnames.append(c['task'])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for i, name in enumerate(gnames):
        for ax, k in zip(axes, ['eps+', 'sigma_add']):
            c = next(c for c in g if c['task'] == name and c['kind'] == k)
            ax.loglog(c['levels'], np.maximum(np.array(c['err'], float), 1e-12), '-', color=PAL[i % 8], label=name.replace('gate:', ''))
            ax.loglog(c['levels'], np.maximum(np.array(c['leak'], float), 1e-12), ':', color=PAL[i % 8])
            ax.set_title(labels[k], fontsize=9); ax.set_xlabel('timing error'); ax.set_ylabel('infidelity (solid) / leakage (dotted)')
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(R, 'fig', 'gate_errors.png'), dpi=160); plt.close(fig)
    # Fig 3: prefactor a vs pulse count (eps+ and sigma_add)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for i, k in enumerate(['eps+', 'sigma_add']):
        xs, ys, names = [], [], []
        for c in cs:
            if c['kind'] != k or c['group'] == 'gate' or c['group'] == 'vqe':
                continue
            e, a = fit_exponent(c['levels'], c['err'], c.get('ideal', 1.0))
            if not np.isnan(a):
                xs.append(c['pulses']); ys.append(a); names.append(c['task'])
        ax.loglog(xs, ys, 'o', color=PAL[i], label=labels[k])
        for x, y, nm in zip(xs, ys, names):
            ax.annotate(nm, (x, y), fontsize=6, xytext=(3, 2), textcoords='offset points')
    ax.set_xlabel('pulses in circuit'); ax.set_ylabel('a  (error ≈ a·x²)'); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(R, 'fig', 'prefactor_vs_pulses.png'), dpi=160); plt.close(fig)


def depth_figure():
    d = load('depth_scaling.json')
    if not d:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, key, ttl in zip(axes, ['cnot_pairs', 'swap_pairs'], ['(CNOT CNOT)^k', '(SWAP_L SWAP_L)^k']):
        rows = d[key]; N = np.array([r['pulses'] for r in rows])
        ax.loglog(N, np.maximum([r['sys_infid'] for r in rows], 1e-15), '-o', ms=3.5, color=PAL[0], label='systematic ε=1e-3')
        ax.loglog(N, np.maximum([r['jit_infid'] for r in rows], 1e-15), '-o', ms=3.5, color=PAL[2], label='jitter σ=1e-3')
        if key == 'cnot_pairs':
            ax.loglog(N, rows[0]['sys_infid'] * (N / N[0]) ** 2, ':', color=PAL[0], lw=1, label='∝ N²')
            ax.loglog(N, rows[0]['jit_infid'] * (N / N[0]), ':', color=PAL[2], lw=1, label='∝ N')
        ax.set_title(ttl, fontsize=9); ax.set_xlabel('pulses N'); ax.set_ylabel('state infidelity')
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(R, 'fig', 'depth_scaling.png'), dpi=160); plt.close(fig)
    # QPE shift
    rows = d['qpe_shift']
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    ax.plot([r['eps'] for r in rows], [r['mean_phase'] for r in rows], '-o', ms=3.5, color=PAL[0])
    ax.axhline(0.3, color='#52514e', lw=0.8, ls=':'); ax.set_xlabel('systematic ε'); ax.set_ylabel('mean estimated phase (φ=0.3)')
    fig.tight_layout(); fig.savefig(os.path.join(R, 'fig', 'qpe_shift.png'), dpi=160); plt.close(fig)


if __name__ == '__main__':
    cs = curves()
    write_summary(cs)
    figures(cs)
    depth_figure()
