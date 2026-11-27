"""TOC graphic (toc_graphic.pdf, 3.25 x 1.75 in): the error split and the framework with a certified result.

Bars: error in D at chemical accuracy split into contamination (naive - projected) and target-state (projected)
terms; after the spin penalty; and after the Gaussian filter at the certified length for 0.1% (Table 2), the actual
error there. NH: QEB pool (results/propfilter_NH.json, results/filter_guarantee.json). Naphthalene (NO): sfGSD pool
(spin-pure; results/filter_guarantee.json, results/naph_certificate.json).
Usage (from the repository root): python scripts/make_toc.py [outdir]
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else '.'
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')   # repository root/results
nh = json.load(open(os.path.join(R, 'propfilter_NH.json')))['runs']
g = json.load(open(os.path.join(R, 'filter_guarantee.json'))); nc = json.load(open(os.path.join(R, 'naph_certificate.json')))
q, qm = nh['QEB'], nh['QEB_mu3']
# certified filter lengths (0.1%) and the actual error there
x_nh = g['NH']['QEB']['cert_T01']; TG = np.array(g['TGRID'])
e_nh = g['NH']['QEB']['curve_shift'][int(np.searchsorted(TG, x_nh))][1]
x_na = nc['sfGSD']['chosen']['T01']; c = [p for p in g['naph_no']['sfGSD']['curve_shift'] if p[0] >= x_na][0]; e_na = c[1]
nh_vals = [('VQE', q['D']['naive'] - q['D']['proj'], q['D']['proj'], q['D']['naive']),
           ('with\npenalty', qm['D']['naive'] - qm['D']['proj'], qm['D']['proj'], qm['D']['naive']),
           ('with\nfilter', 0.0, e_nh, e_nh)]
naph_D = 100 * (g['naph_no']['sfGSD']['curve_shift'][0][1]) / 100       # unfiltered error (%), spin-pure
naph_vals = [('VQE', 0.0, naph_D, naph_D), ('with\nfilter', 0.0, e_na, e_na)]
print('NH QEB', [(l, round(a, 3), round(b, 3), round(s, 4)) for l, a, b, s in nh_vals], 'certified T', round(x_nh, 2))
print('naph sfGSD', [(l, round(s, 4)) for l, a, b, s in naph_vals], 'certified T', round(x_na, 2))

plt.rcParams.update({'font.family': 'sans-serif', 'pdf.fonttype': 42, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'font.size': 6})
C_CON, C_TGT, C_OK = '#E08A1E', '#2B6CD4', '0.35'
fig = plt.figure(figsize=(3.25, 1.75))
ax = fig.add_axes([0.525, 0.27, 0.445, 0.53])
xs = [0, 1.1, 2.2, 3.6, 4.7]
for xi, (lab, cc, tt, s) in zip(xs, nh_vals + naph_vals):
    cc, tt = max(cc, 0), max(tt, 0)
    ax.bar(xi, cc, 0.72, color=C_CON, lw=0); ax.bar(xi, tt, 0.72, bottom=cc, color=C_TGT, lw=0)
    if s < 0.05:
        ax.plot([xi - 0.36, xi + 0.36], [0.035, 0.035], color=C_OK, lw=0.9, solid_capstyle='butt')
    ax.text(xi, max(cc + tt, 0.03) + 0.08, f'{s:.1f}%' if s >= 0.05 else ('<0.01%' if abs(s) < 0.005 else f'{abs(s):.2f}%'), ha='center', va='bottom', fontsize=5.0)
for xi in (2.2, 4.7):                                   # certified filter bars
    ax.text(xi, 0.62, 'certified\n' + r'$\leq$0.1%', ha='center', va='bottom', fontsize=4.8, color='0.25', linespacing=0.95)
ax.set_xticks(xs); ax.set_xticklabels([v[0] for v in nh_vals + naph_vals], fontsize=5.2, linespacing=0.95)
ax.set_ylim(0, 3.5); ax.set_yticks([0, 1, 2, 3]); ax.set_ylabel('Error in $D$ (%)', fontsize=6, labelpad=1.5)
ax.set_xlim(-0.5, 5.2)
for s_ in ['top', 'right']:
    ax.spines[s_].set_visible(False)
ax.tick_params(axis='x', length=0, pad=1.5); ax.tick_params(axis='y', labelsize=5.5, length=2, pad=1)
ax.text(1.1, 3.62, 'NH\n(spin-breaking)', ha='center', va='bottom', fontsize=5.5, linespacing=1.0)
ax.text(4.15, 3.62, 'Naphthalene\n(spin-adapted)', ha='center', va='bottom', fontsize=5.5, linespacing=1.0)
lx = fig.add_axes([0.495, 0.01, 0.50, 0.09]); lx.axis('off')
for i, (col, lab) in enumerate([(C_CON, 'spin contamination'), (C_TGT, 'target-state error')]):
    lx.add_patch(plt.Rectangle((0.0 + 0.54 * i, 0.2), 0.05, 0.6, color=col, transform=lx.transAxes, clip_on=False))
    lx.text(0.065 + 0.54 * i, 0.5, lab, va='center', fontsize=5.2, transform=lx.transAxes)
# left: schematic from Figure 1 - the spin-shifted filter keeps only the target (colors as in the bars)
C_T, C_FILT = '0.25', '#9CC98A'
fig.text(0.225, 0.975, 'VQE $\\rightarrow$ spin-shifted filter\n$\\rightarrow$ certified $D$', ha='center', va='top', fontsize=5.6, weight='bold', linespacing=1.0)
mx = fig.add_axes([0.01, 0.20, 0.43, 0.60]); mx.axis('off'); mx.set_xlim(0, 4.4); mx.set_ylim(-0.3, 2.15)
yy = np.linspace(-0.25, 1.95, 300); gw = np.exp(-(yy - 0.05) ** 2 * 2.6 ** 2 / 2)
L = 0.6
def col(x0, levels, head):
    mx.fill_betweenx(yy, x0, x0 + L * gw, color=C_FILT, alpha=0.6, lw=0, zorder=0)
    for yl, c, lab, kept in levels:
        mx.plot([x0, x0 + L], [yl, yl], color=c, lw=1.4, solid_capstyle='butt', zorder=2)
        mx.text(x0 + L + 0.08, yl + 0.05, lab, va='center', fontsize=4.6, color=c)
        mx.text(x0 + L + 0.08, yl - 0.15, 'kept' if kept else 'removed', va='center', fontsize=4.2,
                color='#3E7D2A' if kept else '0.45', weight='bold' if kept else 'normal')
    mx.text(x0 - 0.05, 2.13, head, ha='left', va='top', fontsize=5.0)
col(0.10, [(0.0, C_T, 'target', True), (0.42, C_CON, 'higher spin', True), (1.05, C_TGT, '$E_1$, spin $S$', False)], r'filter on $\hat H$')
x2 = 2.30
col(x2, [(0.0, C_T, 'target', True), (1.05, C_TGT, '$E_1$, spin $S$', False), (1.70, C_CON, 'higher spin', False)], r"filter on $\hat H'$")
mx.plot([x2, x2 + L], [0.42, 0.42], color=C_CON, lw=0.8, ls=(0, (2, 1.5)), alpha=0.8, zorder=1)
mx.annotate('', xy=(x2 + 0.18, 1.66), xytext=(x2 + 0.18, 0.45), arrowprops=dict(arrowstyle='->', color=C_CON, lw=0.7, ls='--', shrinkA=0, shrinkB=0), zorder=3)
mx.text(x2 + L + 0.08, 0.58, 'spin shift', fontsize=4.4, color=C_CON, va='center')
fig.text(0.225, 0.115, r'error in $D$ = spin contamination', ha='center', fontsize=5.0, color='0.2')
fig.text(0.225, 0.045, '+ target-state error', ha='center', fontsize=5.0, color='0.2')
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, 'toc_graphic.pdf')); fig.savefig(os.path.join(OUT, 'toc_graphic.png'), dpi=600)
r = fig.canvas.get_renderer()
T = [(t, t.get_window_extent(r)) for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip() and t.get_visible()]
ov = [(a.get_text(), b.get_text()) for i, (a, ba) in enumerate(T) for b, bb in T[i + 1:] if ba.overlaps(bb)]
out = [t.get_text() for t, b in T if not (fig.bbox.contains(b.x0, b.y0) and fig.bbox.contains(b.x1, b.y1))]
print('overlaps', ov, 'outside', out)
