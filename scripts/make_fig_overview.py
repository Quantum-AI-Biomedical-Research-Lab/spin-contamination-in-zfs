"""Figure 1 (fig0_overview.pdf): schematic overview of the error split, the spin-shifted filter and the workflow.
Purely schematic; it reads no data. Usage (from the repository root): python scripts/make_fig_overview.py [outdir]
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else '.'
plt.rcParams.update({'font.family': 'sans-serif', 'pdf.fonttype': 42, 'font.size': 7, 'axes.linewidth': 0.6})
C_T, C_R, C_Q = '0.25', '#2B6CD4', '#E08A1E'          # exact state, other spin-S states, higher-spin states
F1, F2, F3 = 7, 6.5, 6                                # title/label, annotation, small
fig = plt.figure(figsize=(7.0, 2.55))

def letter(x, y, t):
    fig.text(x, y, t, fontsize=9, weight='bold', va='top', ha='left')

# ---------------- (a) composition of a VQE state
ax = fig.add_axes([0.03, 0.08, 0.20, 0.74]); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
letter(0.005, 0.97, 'a'); fig.text(0.03, 0.97, 'Two kinds of error', fontsize=F1, va='top')
segs = [(0.62, C_T, r'exact state $|T\rangle$', ''), (0.19, C_R, r'other spin-$S$ states $|R\rangle$', 'target-state term'),
        (0.19, C_Q, r'higher spin $|Q\rangle$', 'contamination term')]
y = 0.04
for h, c, lab, term in segs:
    ax.add_patch(plt.Rectangle((0.02, y), 0.16, h - 0.01, color=c, lw=0))
    ax.text(0.24, y + h / 2 + (0.035 if term else 0), lab, va='center', fontsize=F2)
    if term:
        ax.text(0.24, y + h / 2 - 0.045, r'$\rightarrow$ ' + term, va='center', fontsize=F2, color=c, weight='bold')
    y += h
ax.text(0.10, 1.0, r'$|\psi\rangle$', ha='center', va='bottom', fontsize=F1)
ax.text(0.02, -0.04, 'weights not to scale', fontsize=F3, color='0.4', va='top')

# ---------------- (b) what the filter keeps, without and with the spin shift
C_KEEP, C_DROP, C_FILT = '#3E7D2A', '0.45', '#9CC98A'
bx = fig.add_axes([0.255, 0.05, 0.40, 0.80]); bx.set_xlim(-0.15, 4.75); bx.set_ylim(-0.56, 2.3); bx.axis('off')
letter(0.255, 0.97, 'b'); fig.text(0.28, 0.97, 'The spin shift moves wrong-spin states out of the filter', fontsize=F1, va='top')
yc, tau = 0.06, 2.6                                   # filter center E(psi) slightly above E0; filter width (schematic)
yy = np.linspace(-0.3, 2.05, 400); gw = np.exp(-(yy - yc) ** 2 * tau ** 2 / 2)
def column(x0, levels, head, sub):
    bx.fill_betweenx(yy, x0, x0 + 0.75 * gw, color=C_FILT, alpha=0.55, lw=0, zorder=0)   # filter weight vs energy
    for yl, c, name, kept, side in levels:
        bx.plot([x0, x0 + 0.75], [yl, yl], color=c, lw=1.8, solid_capstyle='butt', zorder=2)
        st, sc = ('kept', C_KEEP) if kept else ('removed', C_DROP)
        if side == 'left':
            bx.text(x0 - 0.06, yl, name, ha='right', va='center', fontsize=F3, color=c)
            bx.text(x0 + 0.81, yl, st, ha='left', va='center', fontsize=F3, color=sc, weight='bold' if kept else 'normal')
        else:
            bx.text(x0 + 0.96, yl, name + ': ', ha='left', va='center', fontsize=F3, color=c)
    bx.text(x0 + 0.38, 2.27, head, ha='center', va='top', fontsize=F2)
    bx.text(x0 + 0.38, 2.10, sub, ha='center', va='top', fontsize=F3, color='0.35')
lv_H = [(0.0, C_T, r'target $E_0$', True, 'left'), (0.38, C_Q, 'higher spin', True, 'left'), (1.0, C_R, r'$E_1$, spin $S$', False, 'left')]
column(0.95, lv_H, r'filter on $\hat H$', 'wrong-spin state is kept')
x1 = 3.00
lv_Hp = [(0.0, C_T, r'target', True), (1.0, C_R, r'$E_1$', False), (1.75, C_Q, 'higher spin', False)]
bx.fill_betweenx(yy, x1, x1 + 0.75 * gw, color=C_FILT, alpha=0.55, lw=0, zorder=0)
for yl, c, name, kept in lv_Hp:
    bx.plot([x1, x1 + 0.75], [yl, yl], color=c, lw=1.8, solid_capstyle='butt', zorder=2)
    st, sc = ('kept', C_KEEP) if kept else ('removed', C_DROP)
    tx = bx.text(x1 + 0.81, yl, name + ',', ha='left', va='center', fontsize=F3, color=c)
    bx.annotate(' ' + st, xycoords=tx, xy=(1, 0.5), va='center', ha='left', fontsize=F3, color=sc, weight='bold' if kept else 'normal')
bx.text(x1 + 0.38, 2.27, r"filter on $\hat H'$ (spin-shifted)", ha='center', va='top', fontsize=F2)
bx.text(x1 + 0.38, 2.10, 'only the target is kept', ha='center', va='top', fontsize=F3, color='0.35')
# dashed arrow: the spin shift raises the higher-spin state; label rotated along the arrow, just beside it
p0, p1 = (2.28, 0.48), (x1 - 0.04, 1.70)
bx.annotate('', xy=p1, xytext=p0, arrowprops=dict(arrowstyle='->', color=C_Q, lw=0.9, ls='--', shrinkA=0, shrinkB=0))
d0, d1 = bx.transData.transform(p0), bx.transData.transform(p1)
ang = np.degrees(np.arctan2(d1[1] - d0[1], d1[0] - d0[0]))
mid = (p0[0] + 0.62 * (p1[0] - p0[0]), p0[1] + 0.62 * (p1[1] - p0[1]))   # label then moved 6 pt further up along the arrow
nx, ny = -np.sin(np.radians(ang)), np.cos(np.radians(ang))       # unit normal, upper-left side of the arrow
bx.annotate('spin shift', xy=mid, xytext=(2.0 * nx + 6.0 * np.cos(np.radians(ang)), 2.0 * ny + 6.0 * np.sin(np.radians(ang))), textcoords='offset points', rotation=ang, rotation_mode='anchor',
            ha='center', va='bottom', fontsize=F3, color=C_Q)
# energy axis and legend for the filter
bx.annotate('', xy=(-0.05, 1.95), xytext=(-0.05, -0.25), arrowprops=dict(arrowstyle='->', color='0.4', lw=0.6))
bx.text(-0.10, 0.85, 'energy', rotation=90, fontsize=F3, color='0.4', ha='right', va='center')
bx.add_patch(plt.Rectangle((0.95, -0.54), 0.16, 0.09, color=C_FILT, alpha=0.55, lw=0))
bx.text(1.17, -0.495, r'filter weight, largest at the VQE energy $E(\psi)$', fontsize=F3, va='center', color='0.25')

# ---------------- (c) workflow
cx = fig.add_axes([0.715, 0.03, 0.28, 0.80]); cx.axis('off'); cx.set_xlim(0, 1); cx.set_ylim(0, 1)
letter(0.705, 0.97, 'c'); fig.text(0.73, 0.97, 'Workflow', fontsize=F1, va='top')
steps = [('ADAPT-VQE, optional spin penalty', 'removes contamination', C_Q),
         (r'Measure $E(\psi)$ and its spread $\sigma$', '', None),
         (r'Certificate sets filter length $\tau$', r'needs a gap bound $\Delta$', None),
         ('Spin-shifted Gaussian filter', 'removes both error terms', C_R),
         (r'$D$ with proven error $\leq$0.1%', '+ statistical error', None)]
n = len(steps); hstep = 1.0 / n
for k, (t1, t2, c) in enumerate(steps):
    yt = 1 - (k + 0.5) * hstep
    cx.add_patch(FancyBboxPatch((0.03, yt - 0.065), 0.94, 0.13, boxstyle='round,pad=0.005,rounding_size=0.02',
                                fc='white', ec='0.35', lw=0.6))
    cx.text(0.06, yt + (0.03 if t2 else 0), t1, va='center', fontsize=F2)
    if t2:
        cx.text(0.06, yt - 0.04, t2, va='center', fontsize=F3, color=c if c else '0.35', weight='bold' if c else 'normal')
    if k < n - 1:
        cx.annotate('', xy=(0.5, yt - hstep + 0.068), xytext=(0.5, yt - 0.068), arrowprops=dict(arrowstyle='->', color='0.35', lw=0.6))
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, 'fig0_overview.pdf')); fig.savefig(os.path.join(OUT, 'fig0_overview.png'), dpi=300)
r = fig.canvas.get_renderer()
T = [(t, t.get_window_extent(r)) for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip() and t.get_visible()]
ov = [(a.get_text(), b.get_text()) for i, (a, ba) in enumerate(T) for b, bb in T[i + 1:] if ba.overlaps(bb)]
out = [t.get_text() for t, b in T if not (fig.bbox.contains(b.x0, b.y0) and fig.bbox.contains(b.x1, b.y1))]
print('overlaps', ov, 'outside', out)

# clearance report (points) for the labels that were too close in earlier versions
def bb(obj): return obj.get_window_extent(r)
txt = {x.get_text(): x for x in fig.findobj(matplotlib.text.Text) if x.get_text().strip()}
boxes = [pp for pp in cx.patches]
k = 72 / fig.dpi
hs = [x for x in fig.findobj(matplotlib.text.Text) if x.get_text().strip() == 'removed' and x.axes is None]
print('gap "removed"(higher spin, right) -> panel c boxes: %.1f pt' % ((min(bb(pp).x0 for pp in boxes) - max(bb(x).x1 for x in fig.findobj(matplotlib.text.Text) if x.get_text().strip() == 'removed')) * k))
ss = txt['spin shift']; rm = [x for x in bx.texts if x.get_text() == 'removed'][0]
b1, b2 = bb(ss), bb(rm)
print('spin shift bottom-left corner vs left "removed": dy=%.1f pt, dx=%.1f pt' % ((b1.y0 - b2.y1) * k, (b1.x0 - b2.x1) * k))
leg = txt[r'filter weight, largest at the VQE energy $E(\psi)$']
print('legend top to lowest bell (y=-0.3):', round((bx.transData.transform((0, -0.3))[1] - bb(leg).y1) * k, 1), 'pt')

hd = [x for x in bx.texts if x.get_text() in ('wrong-spin state is kept', "filter on $\\hat H'$ (spin-shifted)", 'only the target is kept', 'filter on $\\hat H$')]
L = {x.get_text(): bb(x) for x in hd}
print('header gap left-sub -> right-title: %.1f pt' % ((L["filter on $\\hat H'$ (spin-shifted)"].x0 - L['wrong-spin state is kept'].x1) * k))
ol = [ln for ln in bx.lines if ln.get_color() == C_Q][-1]
print('spin shift right edge -> orange line start: %.1f pt' % ((bb(ol).x0 - bb(ss).x1) * k))
