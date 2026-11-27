"""Figure 3 of the manuscript (file fig6_filter_guarantee.pdf): error in D after the Gaussian filter and its proven
bound (eqs 9 and 11) versus the filter length T(E1 - E0), from results/filter_guarantee.json.

Usage (from the repository root): python scripts/make_fig_guarantee.py [outdir]
"""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else 'figures'
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root (scripts/ is one level down)
d = json.load(open(os.path.join(HERE, 'results', 'filter_guarantee.json')))
nc = json.load(open(os.path.join(HERE, 'results', 'naph_certificate.json')))
TG = np.array(d['TGRID'])
FLOOR = 1e-3
COLOR = {'GSD': '#8c8c8c', 'sfGSD': '#2a6fdb', 'QEB': '#8c8c8c'}
plt.rcParams.update({'font.size': 7, 'axes.labelsize': 7, 'legend.fontsize': 6.5,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'font.family': 'sans-serif'})

def guides(ax):
    for y, lab in [(1, '1%'), (0.1, '0.1%')]:
        ax.axhline(y, color='0.75', lw=0.6, ls=':', zorder=0)
        ax.text(6.0, y / 1.15, lab, ha='right', va='top', fontsize=6, color='0.45')
    ax.axvline(3, color='0.85', lw=0.6, zorder=0)
    ax.set_xlim(-0.15, 6.15); ax.set_ylim(FLOOR * 0.7, 300); ax.set_yscale('log')
    ax.set_yticks([1e-3, 1e-2, 0.1, 1, 10, 1e2]); ax.set_yticklabels([r'$\leq10^{-3}$', '$10^{-2}$', '0.1', '1', '10', '$10^{2}$'])
    ax.set_xlabel(r'Filter length $\tau(E_1-E_0)$')

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), sharey=True, gridspec_kw={'wspace': 0.08})
ax = axes[0]
nn = d['naph_no']
for p in ['GSD', 'sfGSD']:
    a = nn[p]; c = np.array(a['curve_shift'])
    ax.plot(c[:, 0], np.maximum(c[:, 1], FLOOR), 'o-', ms=3, lw=1.0, color=COLOR[p], label=f'{p}: actual', zorder=3)
    lab = (lambda s: {'label': s}) if p == 'GSD' else (lambda s: {})
    ax.plot(TG, a['bound_curve_exact'], '--', lw=0.9, color='0.15', **lab('proven bound, exact $F$'))
    ax.plot(TG, [np.nan if v is None else v for v in nc[p]['cert_curve']], ':', lw=1.1, color='0.15',
            **lab('certificate, measured inputs'))
guides(ax)
ax.set_ylabel(r'Error in filtered $D$ (% of $|D|$)')
x_cert = json.load(open(os.path.join(HERE, 'results', 'certified_cost.json')))['x_cert']
ax.set_title(f'Naphthalene (NO, 20 qubits): 0.1% certified at $\\tau(E_1-E_0)$={x_cert:.2f}', loc='left', fontsize=7)
h, l = ax.get_legend_handles_labels(); order = [l.index(s) for s in ['GSD: actual', 'sfGSD: actual', 'proven bound, exact $F$', 'certificate, measured inputs']]
ax.legend([h[i] for i in order], [l[i] for i in order], frameon=False, loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=5.8, handlelength=2.2)

ax = axes[1]
a = d['NH_stretch']['QEB']
c1, c0 = np.array(a['curve_shift']), np.array(a['curve_noshift'])
ax.plot(c1[:, 0], np.maximum(c1[:, 1], FLOOR), '-', lw=1.2, color='#2a6fdb', label=r"actual, $\hat H'$ (spin-shifted)")
ax.plot(c0[:, 0], np.maximum(c0[:, 1], FLOOR), '-', lw=1.2, color='0.35', label=r'actual, $\hat H$ (no shift)')
ax.plot(TG, a['bound_curve_exact'], '--', lw=0.9, color='#2a6fdb', label='proven bound, exact $F$')
ax.plot(TG, [np.nan if v is None else v for v in a['cert_curve']], ':', lw=1.1, color='#2a6fdb', label='certificate, measured inputs')
guides(ax)
ax.set_title(r'NH ($2r_\mathrm{e}$), QEB: the spin shift removes the quintet error floor', loc='left', fontsize=7)
ax.legend(frameon=False, loc='lower left', fontsize=5.8, handlelength=2.2)
for a_, l in zip(axes, 'ab'):
    a_.text(-0.02, 1.06, l, transform=a_.transAxes, fontweight='bold', fontsize=8, ha='right', va='bottom')
fig.subplots_adjust(left=0.085, right=0.995, bottom=0.17, top=0.88)
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, 'fig6_filter_guarantee.pdf')); fig.savefig(os.path.join(OUT, 'fig6_filter_guarantee.png'), dpi=300)
