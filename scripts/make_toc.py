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
nh_vals = [('ADAPT', q['D']['naive'] - q['D']['proj'], q['D']['proj'], q['D']['naive']),
           ('penalty', qm['D']['naive'] - qm['D']['proj'], qm['D']['proj'], qm['D']['naive']),
           ('filter', 0.0, e_nh, e_nh)]
naph_D = 100 * (g['naph_no']['sfGSD']['curve_shift'][0][1]) / 100       # unfiltered error (%), spin-pure
naph_vals = [('ADAPT', 0.0, naph_D, naph_D), ('filter', 0.0, e_na, e_na)]
print('NH QEB', [(l, round(a, 3), round(b, 3), round(s, 4)) for l, a, b, s in nh_vals], 'certified T', round(x_nh, 2))
print('naph sfGSD', [(l, round(s, 4)) for l, a, b, s in naph_vals], 'certified T', round(x_na, 2))

plt.rcParams.update({'font.family': 'sans-serif', 'pdf.fonttype': 42, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'font.size': 6})
C_CON, C_TGT, C_OK = '#E08A1E', '#2B6CD4', '0.35'
fig = plt.figure(figsize=(3.25, 1.75))
ax = fig.add_axes([0.50, 0.27, 0.465, 0.53])
xs = [0, 1, 2, 3.3, 4.3]
for xi, (lab, cc, tt, s) in zip(xs, nh_vals + naph_vals):
    cc, tt = max(cc, 0), max(tt, 0)
    ax.bar(xi, cc, 0.72, color=C_CON, lw=0); ax.bar(xi, tt, 0.72, bottom=cc, color=C_TGT, lw=0)
    if s < 0.05:
        ax.plot([xi - 0.36, xi + 0.36], [0.035, 0.035], color=C_OK, lw=0.9, solid_capstyle='butt')
    ax.text(xi, max(cc + tt, 0.03) + 0.08, f'{s:.1f}%' if s >= 0.05 else f'{abs(s):.2f}%', ha='center', va='bottom', fontsize=5.5)
for xi in (2, 4.3):                                   # certified filter bars
    ax.text(xi, 0.62, 'certified\n' + r'$\leq$0.1%', ha='center', va='bottom', fontsize=4.8, color='0.25', linespacing=0.95)
ax.set_xticks(xs); ax.set_xticklabels([v[0] for v in nh_vals + naph_vals], fontsize=5.5)
ax.set_ylim(0, 3.5); ax.set_yticks([0, 1, 2, 3]); ax.set_ylabel('Error in $D$ (%)', fontsize=6, labelpad=1.5)
ax.set_xlim(-0.5, 4.8)
for s_ in ['top', 'right']:
    ax.spines[s_].set_visible(False)
ax.tick_params(axis='x', length=0, pad=1.5); ax.tick_params(axis='y', labelsize=5.5, length=2, pad=1)
ax.text(1, 3.62, 'NH\n(spin-breaking)', ha='center', va='bottom', fontsize=5.5, linespacing=1.0)
ax.text(3.75, 3.62, 'Naphthalene\n(spin-adapted)', ha='center', va='bottom', fontsize=5.5, linespacing=1.0)
lx = fig.add_axes([0.50, 0.01, 0.465, 0.09]); lx.axis('off')
for i, (col, lab) in enumerate([(C_CON, 'contamination'), (C_TGT, 'target-state')]):
    lx.add_patch(plt.Rectangle((0.0 + 0.52 * i, 0.2), 0.055, 0.6, color=col, transform=lx.transAxes, clip_on=False))
    lx.text(0.075 + 0.52 * i, 0.5, lab, va='center', fontsize=5.5, transform=lx.transAxes)
# left: molecule, pipeline, split
mx = fig.add_axes([0.05, 0.41, 0.31, 0.55]); mx.set_aspect('equal'); mx.axis('off')
s3 = np.sqrt(3) / 2
for cx in (-s3, s3):
    ang = np.deg2rad(np.arange(90, 451, 60)); mx.plot(cx + np.cos(ang), np.sin(ang), color='0.15', lw=1.0, solid_capstyle='round')
    mx.add_patch(plt.Circle((cx, 0), 0.55, fill=False, color='0.15', lw=0.6))
for dx in (-0.25, 0.25):
    mx.annotate('', xy=(dx, 2.1), xytext=(dx, 1.25), arrowprops=dict(arrowstyle='-|>', color='0.15', lw=0.9, mutation_scale=6))
mx.text(0.55, 1.65, 'triplet', fontsize=5.5, va='center')
mx.set_xlim(-1.9, 1.9); mx.set_ylim(-1.15, 2.2)
fig.text(0.205, 0.285, r'VQE $\rightarrow$ spin-shifted filter', ha='center', fontsize=5.8, weight='bold')
fig.text(0.205, 0.20, r'$\rightarrow$ certified $D$', ha='center', fontsize=5.8, weight='bold')
fig.text(0.205, 0.095, r'$D_\mathrm{VQE}-D_\mathrm{exact}=$', ha='center', fontsize=5.2, color='0.2')
fig.text(0.205, 0.025, 'contamination + target-state', ha='center', fontsize=5.2, color='0.2')
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, 'toc_graphic.pdf')); fig.savefig(os.path.join(OUT, 'toc_graphic.png'), dpi=600)
r = fig.canvas.get_renderer()
T = [(t, t.get_window_extent(r)) for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip() and t.get_visible()]
ov = [(a.get_text(), b.get_text()) for i, (a, ba) in enumerate(T) for b, bb in T[i + 1:] if ba.overlaps(bb)]
out = [t.get_text() for t, b in T if not (fig.bbox.contains(b.x0, b.y0) and fig.bbox.contains(b.x1, b.y1))]
print('overlaps', ov, 'outside', out)
