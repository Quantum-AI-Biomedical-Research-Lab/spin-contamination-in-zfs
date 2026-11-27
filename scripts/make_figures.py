"""Make the two manuscript figures (vector PDF) from results/*.json.

Usage (from the repository root): python scripts/make_figures.py [outdir]
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else 'figures'
os.makedirs(OUT, exist_ok=True)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root (scripts/ is one level down)
RES = os.path.join(HERE, 'results')

SYSTEMS = ['CH2', 'NH', 'NH_stretch', 'O2', 'TMM']
LABEL = {'CH2': r'CH$_2$', 'NH': 'NH', 'NH_stretch': r'NH ($2r_\mathrm{e}$)',
         'O2': r'O$_2$', 'TMM': 'TMM'}
POOLS = ['GSD', 'QEB', 'sfGSD']
COLOR = {'GSD': '#8c8c8c', 'QEB': '#cfcfcf', 'sfGSD': '#2a6fdb'}
CHEM = 1.6e-3

plt.rcParams.update({'font.size': 7, 'axes.labelsize': 7, 'legend.fontsize': 6.5,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'font.family': 'sans-serif'})

# ---------------------------------------------------------------- Figure 1 (main text)
err = {}
for s in SYSTEMS:
    o = json.load(open(os.path.join(RES, f'{s}.json')))
    Dr = o['D_CASCI']
    for p in POOLS:
        h = next(h for h in o['runs'][p]['history'] if h['dE'] < CHEM)
        err[s, p] = 100 * (h['D'] - Dr) / Dr

RANGE = {}
for p in POOLS:
    for sname in SYSTEMS:
        f = os.path.join(RES, f'spread_{sname}_{p}.json')
        if not os.path.exists(f):
            continue
        d = json.load(open(f)); vals = []
        for v in ('tie', 'gauge', 'noise0.16'):
            for run in d.get(v, []):
                Dx = run.get('D_exact', d['D_exact'])
                ic = next((n for n, q in enumerate(run['rows']) if q['dE'] < CHEM), None)
                if ic is not None:
                    vals.append(100 * (run['rows'][ic]['D'] - Dx) / Dx)
        if vals:
            RANGE[sname, p] = (min(vals), max(vals))
# naphthalene (20 qubits): standard runs in ROHF and natural orbitals; whiskers from the standard run and three repeats in each
NS = json.load(open(os.path.join(RES, 'naph_summary.json')))
SP = json.load(open(os.path.join(RES, 'spread_summary.json')))
NAPH = ['naph_rohf', 'naph_no']
LABEL.update({'naph_rohf': 'Naph.\n(ROHF)', 'naph_no': 'Naph.\n(NO)'})
for key, o in [('naph_rohf', 'rohf'), ('naph_no', 'no')]:
    for p in POOLS:
        v = NS[o][p].get('D_err_chem_pct')
        err[key, p] = v                                  # None: QEB does not reach chemical accuracy
for key in ('naph_rohf', 'naph_no'):
    for p in ('GSD', 'sfGSD'):
        vals = [r['err'] for r in SP.get(key, {}).get(p, {}).get('runs', []) if r.get('err') is not None]
        if len(vals) > 1:
            RANGE[key, p] = (min(vals), max(vals))
ALL = SYSTEMS + NAPH
fig, ax = plt.subplots(figsize=(3.3, 2.35))
x = np.arange(len(ALL))
w = 0.26
ax.axhspan(-1, 1, color='0.93', zorder=0, lw=0)
ax.axhline(0, color='0.3', lw=0.6, zorder=1)
for j, p in enumerate(POOLS):
    vals = [err[s, p] if err[s, p] is not None else 0.0 for s in ALL]
    bars = ax.bar(x + (j - 1) * w, vals, w, color=COLOR[p], edgecolor='0.45' if p == 'QEB' else COLOR[p],
                  lw=0.4, label=p + (' (spin-adapted)' if p == 'sfGSD' else ''), zorder=2)
    for xi, v, sname in zip(x + (j - 1) * w, vals, ALL):
        if err[sname, p] is None:
            ax.text(xi, 0.15, 'n.r.', rotation=90, ha='center', va='bottom', fontsize=5.2, color='0.35')
            continue
        if abs(v) < 0.005:                      # TMM: exact state, no label
            continue
        lo, hi = RANGE.get((sname, p), (v, v))
        ypos = max(v, hi) + 0.15 if v >= 0 else min(v, lo) - 0.15
        ax.text(xi, ypos, f'{v:+.2f}', rotation=90,
                ha='center', va='bottom' if v >= 0 else 'top', fontsize=5.2)
# whiskers: range over random tie-breaking, random orbital gauge and selection noise of 0.16 mEh (spread.py)
for (sname, p), (lo, hi) in RANGE.items():
    xc = x[ALL.index(sname)] + (POOLS.index(p) - 1) * w
    ax.plot([xc, xc], [lo, hi], color='k', lw=0.6, zorder=3)
    for yv in (lo, hi):
        ax.plot([xc - 0.05, xc + 0.05], [yv, yv], color='k', lw=0.6, zorder=3)
ax.axvline(len(SYSTEMS) - 0.5, color='0.6', lw=0.5, ls=':')
ax.text(x[ALL.index('TMM')], 0.25, 'all exact', ha='center', va='bottom', fontsize=5.5, color='0.35')
ax.set_xticks(x)
ax.set_xticklabels([LABEL[s] for s in ALL], fontsize=5.8)
ax.set_ylabel(r'Error in $D$ at chemical accuracy (%)')
ax.set_ylim(-3.6, 9.8)
ax.text(len(SYSTEMS) - 0.6, -1.15, r'$\pm$1%', ha='right', va='top', fontsize=5.5, color='0.4')
ax.legend(frameon=False, loc='upper right', ncol=1, handlelength=1.2)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, 'fig1_d_error.pdf'))
plt.close(fig)

# ---------------------------------------------------------------- contamination-scan figure (repository only, no longer in the paper; file fig2_contamination_scan.pdf)
M = json.load(open(os.path.join(RES, 'measure.json')))
fig, ax = plt.subplots(figsize=(3.3, 2.5))
markers = {'CH2': 'o', 'NH': 's', 'NH_stretch': 'D', 'O2': '^', 'TMM': 'v'}
shades = {'CH2': 1.0, 'NH': 0.8, 'NH_stretch': 0.6, 'O2': 0.45, 'TMM': 0.3}
for s in SYSTEMS:
    sc = np.array(M[s]['controlled_scan'])
    xs, ys = sc[:, 0], 100 * sc[:, 1]
    slope = np.polyfit(np.log10(xs[:5]), np.log10(ys[:5]), 1)[0]
    ax.loglog(xs, ys, marker=markers[s], ms=3, lw=1.0, color='#2a6fdb', alpha=shades[s],
              label=f'{LABEL[s]} (slope {slope:.3f})')
xr = np.array([4e-6, 4e-2])
ax.loglog(xr, 2000 * np.sqrt(xr) / 100, ls='--', color='0.45', lw=0.8)
ax.text(4e-3, 0.9 * 20 * np.sqrt(4e-3) * 0.5, r'slope $\frac{1}{2}$', fontsize=6, color='0.35',
        ha='left', va='top')
ax.set_xlabel(r'$\langle S^2\rangle - 2$')
ax.set_ylabel(r'$|\Delta D|/D$ (%)')
ax.legend(frameon=False, loc='upper left', handlelength=1.6)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, 'fig2_contamination_scan.pdf'))
plt.close(fig)
# ---------------------------------------------------------------- Figure (repository only) (error bars along trajectories; file fig5_error_bars.pdf)
panels = [('NH_stretch', 'GSD', r'NH ($2r_\mathrm{e}$), GSD, 12 qubits'),
          ('naph_no', 'sfGSD', r'Naphthalene (NO), sfGSD, 20 qubits'),
          ('naph_no', 'GSD', r'Naphthalene (NO), GSD, 20 qubits')]
panels = [p for p in panels if os.path.exists(os.path.join(RES, f'errbar_{p[0]}_{p[1]}.json'))]
fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 2.2))
for ax, (s, p, title) in zip(np.atleast_1d(axes), panels):
    rows = json.load(open(os.path.join(RES, f'errbar_{s}_{p}.json')))
    ref = json.load(open(os.path.join(RES, 'errbar_naph_no.json' if s.startswith('naph') else 'errbar_small.json')))
    Dex = abs(ref['D'] if s.startswith('naph') else ref[s]['D'])
    n = np.arange(len(rows))
    f = lambda k: 100 * np.maximum(np.array([r[k] for r in rows]), 1e-12) / Dex
    ax.semilogy(n, f('err'), color='k', lw=1.0, label='actual error')
    ax.semilogy(n, 100 * (np.array([r['b_contam'] + r['b_target'] for r in rows])) / Dex, color='0.55',
                ls='--', lw=0.9, label='rigorous bound')
    ax.semilogy(n, f('b_est'), color='#2a6fdb', lw=1.0, label='measurable estimate')
    ax.axhline(1, color='0.6', lw=0.6, ls=':')
    ax.set_title(title, fontsize=6.5, loc='left')
    ax.set_xlabel('ADAPT operators'); ax.set_ylim(1e-3, 1e4)
np.atleast_1d(axes)[0].set_ylabel(r'Error in $D$ (% of $|D|$)')
np.atleast_1d(axes)[1 if len(panels) > 1 else 0].legend(frameon=False, loc='lower right')
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, 'fig5_error_bars.pdf'))
plt.close(fig)
# ---------------------------------------------------------------- remedies: error in D vs CNOTs (file figS1_remedies.pdf; not shown in the manuscript)
if os.path.exists(os.path.join(RES, 'remedies.json')):
    REM = json.load(open(os.path.join(RES, 'remedies.json')))
    sysl = ['CH2', 'NH', 'NH_stretch', 'O2']
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.6), sharey='row')
    for j, s in enumerate(sysl):
        Dex = REM[s]['D_exact']
        for i, p in enumerate(['GSD', 'sfGSD']):
            ax = axes[i, j]
            A = json.load(open(os.path.join(RES, f'remedies_{s}_{p}_A.json')))['rows']
            B2 = json.load(open(os.path.join(RES, f'remedies_{s}_{p}_B2.json')))['rows']
            e = lambda rows, k: np.array([abs(r[k] - Dex) / abs(Dex) * 100 if r.get(k) is not None else np.nan
                                          for r in rows])
            c = lambda rows: np.array([r['cnot'] for r in rows])
            ax.semilogy(c(A), np.maximum(e(A, 'D'), 1e-4), color='k', lw=1.0, label='A: ADAPT')
            ax.semilogy(c(A), np.maximum(e(A, 'D_lanczos'), 1e-4), color='0.55', lw=0.9, ls='--',
                        label='C1: + Lanczos step')
            ax.semilogy(c(B2), np.maximum(e(B2, 'D'), 1e-4), color='#2a6fdb', lw=1.0, label='B2: D-aware')
            for rows, col in ((A, 'k'), (B2, '#2a6fdb')):
                k = next((n for n, r in enumerate(rows) if r['dE'] < CHEM), None)
                if k is not None:
                    ax.plot(rows[k]['cnot'], max(e(rows, 'D')[k], 1e-4), 'o', ms=3, color=col)
            ax.axhline(1, color='0.6', lw=0.6, ls=':')
            ax.set_xlim(0, 4000 if p == 'GSD' else 6000); ax.set_ylim(1e-3, 2e2)
            ax.set_xticks([0, 1000, 2000, 3000] if p == 'GSD' else [0, 2000, 4000])
            if i == 0:
                ax.set_title(LABEL[s], fontsize=7)
            if i == 1:
                ax.set_xlabel('CNOTs')
        axes[0, 0].set_ylabel('GSD: |error in $D$| (%)'); axes[1, 0].set_ylabel('sfGSD: |error in $D$| (%)')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc='upper center', ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, 'figS1_remedies.pdf'))
    plt.close(fig)

# ---------------------------------------------------------------- Figure (repository only) (spread under tie-breaking, gauge, noise; file fig4_spread.pdf)
if os.path.exists(os.path.join(RES, 'spread_summary.json')):
    SS = json.load(open(os.path.join(RES, 'spread_summary.json')))
    sysl = [s for s in ['CH2', 'NH', 'NH_stretch', 'O2'] if s in SS]
    has_naph = 'naph_no' in SS
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={'width_ratios': [len(sysl) * 3.6, 3.4 if has_naph else 0.01]})
    ax = axes[0]
    VSTYLE = [('tie', 'o', '0.55', 'random tie-breaking'), ('gauge', 's', '#e08a1e', 'random orbital gauge'),
              ('noise0.16', 'v', '#9cc0f0', r'selection noise $\delta E$ = 0.16 m$E_\mathrm{h}$'),
              ('noise0.5', '^', '#5b8fd9', r'0.5 m$E_\mathrm{h}$'), ('noise1.6', 'D', '#1f4fa8', r'1.6 m$E_\mathrm{h}$')]
    ax.axhspan(-1, 1, color='0.93', zorder=0, lw=0); ax.axhline(0, color='0.3', lw=0.6, zorder=1)
    xt, xl = [], []
    rng = np.random.default_rng(0)
    for i, s in enumerate(sysl):
        for j, p in enumerate(POOLS):
            x0 = i * 3.6 + j
            r = SS[s][p]
            for k, (v, m, c, lab) in enumerate(VSTYLE):
                if v not in r:
                    continue
                vals = []
                # recover signed errors from the per-run files
                d = json.load(open(os.path.join(RES, f'spread_{s}_{p}.json')))
                for run in d[v]:
                    Dex = run.get('D_exact', d['D_exact'])
                    ic = next((n for n, q in enumerate(run['rows']) if q['dE'] < CHEM), None)
                    if ic is not None:
                        vals.append(100 * (run['rows'][ic]['D'] - Dex) / Dex)
                xs = x0 + (k - 2) * 0.13 + rng.uniform(-0.03, 0.03, len(vals))
                ax.scatter(xs, vals, s=6, marker=m, color=c, lw=0, alpha=0.85, zorder=3,
                           label=lab if (i == 0 and j == 0) else None)
            ax.plot([x0 - 0.38, x0 + 0.38], [r['standard']['err']] * 2, color='k', lw=1.0, zorder=4,
                    label='standard run' if (i == 0 and j == 0) else None)
            xt.append(x0); xl.append(p)
        ax.text(i * 3.6 + 1, 1.02, LABEL[s], transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=7)
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=5.8)
    ax.text(-0.07, 1.09, '(a) 12 qubits', transform=ax.transAxes, ha='left', va='bottom', fontsize=7, fontweight='bold')
    ax.set_ylabel(r'Error in $D$ at chemical accuracy (%)')
    from matplotlib.lines import Line2D
    h6 = [Line2D([], [], ls='', marker=m, color=c, ms=3.5) for v, m, c, lab in VSTYLE] + [Line2D([], [], color='k', lw=1.0)]
    l6 = [lab for v, m, c, lab in VSTYLE] + ['standard run']
    if has_naph:
        ax2 = axes[1]
        ax2.axhspan(-1, 1, color='0.93', zorder=0, lw=0); ax2.axhline(0, color='0.3', lw=0.6, zorder=1)
        npool = {'GSD': 0, 'QEB': 1, 'sfGSD': 2}
        ax2.text(1, 0.0, 'not\nreached', ha='center', va='center', fontsize=5.5, color='0.4', zorder=5,
                 bbox=dict(facecolor='white', edgecolor='none', pad=0.5))
        for p in ['GSD', 'sfGSD']:
            j = npool[p]
            runs = SS['naph_no'][p]['runs']
            v = [q['err'] for q in runs if q.get('err') is not None]
            ax2.plot([j - 0.3, j + 0.3], [v[0]] * 2, color='k', lw=1.0, zorder=4)
            if len(v) > 1:
                ax2.scatter(j + rng.uniform(-0.08, 0.08, len(v) - 1), v[1:], s=10, marker='v', color='#9cc0f0',
                            lw=0, zorder=3)
        ax2.set_xticks([0, 1, 2]); ax2.set_xticklabels(['GSD', 'QEB', 'sfGSD'], fontsize=5.8); ax2.set_xlim(-0.6, 2.6)
        ax2.set_ylim(-1.2, 2.6)
        ax2.set_ylabel(r'Error in $D$ at chemical accuracy (%)')
        ax2.text(1, 1.02, 'Naphthalene (NO)', transform=ax2.get_xaxis_transform(), ha='center', va='bottom', fontsize=7)
        ax2.text(-0.25, 1.09, '(b) 20 qubits', transform=ax2.transAxes, ha='left', va='bottom', fontsize=7, fontweight='bold')
    else:
        axes[1].axis('off')
    fig.legend(h6, l6, frameon=False, fontsize=6, loc='upper center', ncol=6, handletextpad=0.2, columnspacing=1.0,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(OUT, 'fig4_spread.pdf'))
    plt.close(fig)
print('wrote', os.listdir(OUT))
