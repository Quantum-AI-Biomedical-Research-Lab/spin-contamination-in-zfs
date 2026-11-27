"""LaTeX table bodies of supplementary tables that are kept in the repository only (formerly in the Supporting Information) from results/*.json:
spread under randomized selection, naphthalene repeats, finite-shot readout, error estimate with the exact s,
spin and variance penalties, refinements, spin densities, Gaussian filter and its cost.
Usage (from the repository root): python scripts/make_si_tables.py OUTDIR"""
import json
import os
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else 'tables'
os.makedirs(OUT, exist_ok=True)
S = json.load(open('results/spread_summary.json'))
SN = json.load(open('results/shotnoise.json'))
NAME = {'CH2': r'\ce{CH2}', 'NH': 'NH', 'NH_stretch': r'NH ($2r_\mathrm{e}$)', 'O2': r'\ce{O2}'}


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def rng(st):
    if st is None:
        return '-'
    if abs(st['max'] - st['min']) < 0.005:
        return f"${st['min']:+.2f}$"
    return f"${st['min']:+.2f}$ to ${st['max']:+.2f}$"


# ---- Table (repository only): error in D at chemical accuracy under perturbations
rows = []
for s in ['CH2', 'NH', 'NH_stretch', 'O2']:
    for p in ['GSD', 'QEB', 'sfGSD']:
        r = S[s][p]
        cells = [NAME[s] if p == 'GSD' else '', p, f"${r['standard']['err']:+.2f}$"]
        for v in ['tie', 'gauge', 'noise0.16', 'noise0.5', 'noise1.6']:
            cells.append(rng(r[v]['err']) if v in r else 'n.a.')
        rows.append(' & '.join(cells) + r'\\')
    if s != 'O2':
        rows.append(r'\midrule')
open(os.path.join(OUT, 'tab_si_spread.tex'), 'w').write('\n'.join(rows) + '\n')

# ---- Table (repository only): naphthalene repeats (natural orbitals, then ROHF orbitals when the repeats exist)
rows = []
SETS = [('naph_no', 'NO', 'naph_no_{p}', 'naph_spread_{p}_s{k}', json.load(open('results/naph_reference_no.json'))['D_exact_NO']),
        ('naph_rohf', 'ROHF', 'naph_{p}', 'naph_spread_rohf_{p}_s{k}', json.load(open('results/naph_reference.json'))['D_CASCI'])]
for key, olab, std, rep, ref in SETS:
    if key not in S or len(S[key]['GSD']['runs']) < 2:
        continue
    if rows:
        rows.append(r'\midrule')
    for p in ['GSD', 'sfGSD']:
        e0 = json.load(open(f'results/{std.format(p=p)}.json'))['E_exact']
        for k, m in enumerate(S[key][p]['runs']):
            label = 'standard' if k == 0 else f'repeat {k}'
            if k == 0:
                h = json.load(open(f'results/{std.format(p=p)}.json'))['history']
                ic = next(i for i, x in enumerate(h) if x['E'] - e0 < 1.6e-3)
                last = h[min(ic + 30, len(h) - 1)]
            else:
                last = json.load(open(f'results/{rep.format(p=p, k=k)}.json'))['history'][-1]
            e_last = 100 * (last['D'] - ref) / ref
            rows.append(f"{olab if (k == 0 and p == 'GSD') else ''} & {p if k == 0 else ''} & {label} & {m['n_chem']} & ${m['err']:+.2f}$ & ${m['err_proj']:+.2f}$ & "
                        f"{last['n']} & ${e_last:+.2f}$\\\\")
open(os.path.join(OUT, 'tab_si_naph.tex'), 'w').write('\n'.join(rows) + '\n')

# ---- Table (repository only): finite-shot readout
rows = []
for s in ['CH2', 'NH', 'NH_stretch', 'O2']:
    for p in ['GSD', 'QEB', 'sfGSD']:
        q = SN[s][p]
        cells = [NAME[s] if p == 'GSD' else '', p]
        for r in ['r0.01', 'r0.003']:
            x = q[r]
            cells.append(f"{100 * x['frac_chem_within_1pct']:.0f}")
            cells += [f"{100 * x[m]['frac_stop']:.0f}" for m in ['m10', 'm20', 'm30']]
        cells += [str(q['noiseless'][m]['n_stop']) if q['noiseless'][m]['n_stop'] is not None else '-'
                  for m in ['m10', 'm20', 'm30']]
        rows.append(' & '.join(cells) + r'\\')
    if s != 'O2':
        rows.append(r'\midrule')
open(os.path.join(OUT, 'tab_si_shotnoise.tex'), 'w').write('\n'.join(rows) + '\n')
for f in ['tab_si_spread.tex', 'tab_si_naph.tex', 'tab_si_shotnoise.tex']:
    print('==', f)
    print(open(os.path.join(OUT, f)).read())

# ---- Table (repository only): measurable estimate b with the exact s (errbar_ceiling.py)
C = json.load(open('results/errbar_ceiling.json'))
NAME2 = dict(NAME, naph_no='Naphthalene (NO)')
rows = []
for s in ['CH2', 'NH', 'NH_stretch', 'O2', 'naph_no']:
    for i, p in enumerate([q for q in ['GSD', 'QEB', 'sfGSD'] if C[s].get(q)]):
        e = C[s][p]
        rows.append(f"{NAME2[s] if i == 0 else ''} & {p} & {e['err_pct']:.2f} & {e['b_pct']:.1f} & "
                    f"{e['b_exact_s_pct']:.1f} & {e['s_var']:.3f} & {e['s_true']:.3f}\\\\")
    if s != 'naph_no':
        rows.append(r'\midrule')
open(os.path.join(OUT, 'tab_si_ceiling.tex'), 'w').write('\n'.join(rows) + '\n')
print(open(os.path.join(OUT, 'tab_si_ceiling.tex')).read())

# ---- Tables (repository only): spin penalty, refinements, penalty shot cost
NAMES4 = ['CH2', 'NH', 'NH_stretch', 'O2']
fmt = lambda x, d=2: '-' if x is None else ('$0.00$' if abs(x) < 0.5 * 10 ** -d else f"${x:+.{d}f}$")
# S13: mu scan (penalty_scan_*.json)
rows = []
for s in NAMES4:
    P = json.load(open(f'results/penalty_scan_{s}.json'))['scan']
    for i, p in enumerate(['GSD', 'QEB']):
        cells = [NAME[s] if i == 0 else '', p]
        for mu in [0.0, 0.03, 0.1, 0.3, 1.0, 3.0]:
            m = P[f'{p}_mu{mu}']['metrics']
            cells.append(f"{fmt(m.get('err'))} ({m['cnot_D1'] if m['cnot_D1'] is not None else '-'})")
        rows.append(' & '.join(cells) + r'\\')
    if s != 'O2':
        rows.append(r'\midrule')
open(os.path.join(OUT, 'tab_si_muscan.tex'), 'w').write('\n'.join(rows) + '\n')
# S14: robustness under selection noise (penalty_scan noise mu=0, penalty_noise3 mu=3)
rows = []
for s in NAMES4:
    P = json.load(open(f'results/penalty_scan_{s}.json'))['noise']
    N3 = json.load(open(f'results/penalty_noise3_{s}.json'))
    for i, p in enumerate(['GSD', 'QEB']):
        cells = [NAME[s] if i == 0 else '', p]
        for ms in [P[f'{p}_mu0.0'], N3[p]]:
            e = [m['err'] for m in ms]; c = [m['cnot_D1'] for m in ms if m['cnot_D1'] is not None]
            er = f"{fmt(min(e))} to {fmt(max(e))}" if max(e) - min(e) >= 0.005 else fmt(e[0])
            cr = (f"{min(c)}-{max(c)}" if min(c) != max(c) else f"{c[0]}") if c else '-'
            cells += [er, cr]
        rows.append(' & '.join(cells) + r'\\')
    if s != 'O2':
        rows.append(r'\midrule')
open(os.path.join(OUT, 'tab_si_robust.tex'), 'w').write('\n'.join(rows) + '\n')
# variance penalty (penalty_<s>.json); written but not included in the SI
rows = []
for s in NAMES4:
    P = json.load(open(f'results/penalty_{s}.json'))['runs']
    first = True
    for tag in ['GSD_A', 'GSD_W1', 'GSD_W3', 'GSD_W10', 'GSD_S1W3', 'sfGSD_A', 'sfGSD_W1', 'sfGSD_W3', 'sfGSD_W10']:
        m = P[tag]['metrics']; pool, var = tag.split('_')
        lab = {'A': 'none', 'W1': '$w=1$', 'W3': '$w=3$', 'W10': '$w=10$', 'S1W3': r'$\mu=1$, $w=3$'}[var]
        rows.append(f"{NAME[s] if first else ''} & {pool} & {lab} & {m['n_chem']} & {fmt(m.get('err'))} & "
                    f"{m.get('sigma_H_mEh', float('nan')):.1f} & {m['cnot_D1'] if m['cnot_D1'] is not None else '-'}\\\\")
        first = False
    if s != 'O2':
        rows.append(r'\midrule')
open(os.path.join(OUT, 'tab_si_variance.tex'), 'w').write('\n'.join(rows) + '\n')
# S15: refinements at chemical accuracy (fix_<s>.json) and naphthalene
rows = []
for s in NAMES4:
    F = json.load(open(f'results/fix_{s}.json'))['runs']
    first = True
    for tag in ['GSD', 'QEB', 'sfGSD', 'QEB_mu3']:
        r = F[tag]['refine']
        lab = tag.replace('_mu1', r', $\mu=1$').replace('_mu3', r', $\mu=3$')
        rows.append(f"{NAME[s] if first else ''} & {lab} & " + ' & '.join(fmt(r[f'k{k}']) for k in (1, 2, 4, 6)) +
                    f" & {fmt(r['si1'])} & {fmt(r['si2'], 3)}\\\\")
        first = False
    rows.append(r'\midrule')
KR, SI = json.load(open('results/krylov_naph.json')), json.load(open('results/shiftinvert_naph.json'))
for i, p in enumerate(['GSD', 'sfGSD']):
    res = KR[p]['res'][str(KR[p]['n_chem'])]
    rows.append(f"{'Naphthalene (NO)' if i == 0 else ''} & {p} & " + ' & '.join(fmt(res[str(k)]['err']) for k in (1, 2, 4, 6)) +
                f" & {fmt(SI['naph_' + p]['err'][1])} & {fmt(SI['naph_' + p]['err'][2], 3)}\\\\")
open(os.path.join(OUT, 'tab_si_refine.tex'), 'w').write('\n'.join(rows) + '\n')
# S16: shot cost of the penalty terms
SH = json.load(open('results/penalty_shots.json'))
rows = []
for s in NAMES4:
    q = SH[s]
    rows.append(f"{NAME[s]} & {q['H_terms']} ({q['H_groups']}) & {sci(q['H_shots'])} & {q['S2_terms']} ({q['S2_groups']}) & "
                f"{sci(q['S2_shots'])} & {q['S2_ratio']:.0f} & " +
                (f"{q['H2_terms']} ({q['H2_groups']}) & {sci(q['H2_shots'])} & {q['H2_ratio']:.0f}" if 'H2_terms' in q else '- & - & -') + r'\\')
open(os.path.join(OUT, 'tab_si_penshots.tex'), 'w').write('\n'.join(rows) + '\n')
for f in ['tab_si_muscan.tex', 'tab_si_robust.tex', 'tab_si_variance.tex', 'tab_si_refine.tex', 'tab_si_penshots.tex']:
    print('==', f); print(open(os.path.join(OUT, f)).read())

# ---- Tables (repository only): spin densities at the nuclei, Gaussian filter, and filter cost (property_filter.py, naph_filter.py)
import numpy as np
PF = {s: json.load(open(f'results/propfilter_{s}.json')) for s in NAMES4}
NF = json.load(open('results/naph_filter.json'))
f2 = lambda x: '$0.00$' if abs(x) < 0.005 else f"${x:+.2f}$"
rows = []
for s, ats in [('CH2', [0, 1]), ('NH', [0, 1]), ('NH_stretch', [1])]:
    d = PF[s]
    for k, a in enumerate(ats):
        r = d['runs']
        si = max(abs(r[t]['shiftinvert']['rho'][a]) for t in r)
        gs = max(abs(r[t]['gauss']['rule_spin']['rho'][a]) for t in r)
        x2 = max(abs(r[t]['rho']['cross_S+2'][a]) for t in r)
        rows.append(f"{NAME[s] if k == 0 else ''} & {d['atoms'][a]} & ${d['rho_exact'][a]:+.4f}$ & {f2(r['GSD']['rho']['naive'][a])} & "
                    f"{f2(r['GSD']['rho']['proj'][a])} & {f2(r['GSD']['rho']['cross_S+1'][a])} & {sci(max(x2, 1e-30)) if x2 > 0 else '0'} & "
                    f"{f2(r['sfGSD']['rho']['naive'][a])} & {f2(r['QEB_mu3']['rho']['naive'][a])} & ${si:.2f}$ & ${gs:.2f}$\\\\")
open(os.path.join(OUT, 'tab_si_rho.tex'), 'w').write('\n'.join(rows) + '\n')

rows = []
for s in NAMES4:
    for k, t in enumerate(['QEB', 'QEB_mu3']):
        g = PF[s]['runs'][t]['gauss']
        rows.append(f"{NAME[s] if k == 0 else ''} & {'QEB' if t == 'QEB' else r'QEB, $\mu=3$'} & {f2(PF[s]['runs'][t]['D']['naive'])} & "
                    + ' & '.join(f2(g[str(T)]['D']) for T in [5, 10, 20, 30, 40])
                    + f" & {g['rule']['T']:.1f} & {f2(g['rule']['D'])} & {f2(g['rule_spin']['D'])}\\\\")
    rows.append(r'\midrule')
for k, p in enumerate(['GSD', 'sfGSD']):
    g = NF['runs'][p]['gauss']
    rows.append(f"{'Naphthalene (NO)' if k == 0 else ''} & {p} & {f2(NF['runs'][p]['D'])} & - & "
                + ' & '.join(f2(g[str(T)]['D']) for T in [10, 20, 30, 40]) + f" & {g['rule']['T']:.1f} & {f2(g['rule']['D'])} & {f2(g['rule_spin']['D'])}\\\\")
open(os.path.join(OUT, 'tab_si_gauss.tex'), 'w').write('\n'.join(rows) + '\n')

rows = []
cost_summary = {}
items = [(s, PF[s]['gaps']['E1_triplet'], PF[s]['cost'], PF[s]['runs']['QEB_mu3']['gauss']['rule']['Ng']) for s in NAMES4]
items.append(('naph', NF['gap_E1'], NF['cost'], NF['runs']['GSD']['gauss']['rule']['Ng']))
for s, gap, c, Ng in items:
    T = 3.0 / gap
    q = c['lam_H'] * 2 * T * np.sqrt(2 / np.pi)
    ND = (c['sum_sigma_g'] / (0.01 * abs(c['X_mean']))) ** 2
    F = (c['sum_sqrt_m2_g'] / c['sum_sigma_g']) ** 2 / Ng ** 2
    tot = F * ND * q
    cost_summary[s] = dict(gap=gap, T=T, lam_H=c['lam_H'], queries=q, N_D=ND, F=F, total=tot)
    nm = 'Naphthalene (NO)' if s == 'naph' else NAME[s]
    rows.append(f"{nm} & {1e3*gap:.0f} & {T:.1f} & {c['lam_H']:.1f} & {q:.0f} & {sci(ND)} & {F:.2f} & {sci(tot)}\\\\")
open(os.path.join(OUT, 'tab_si_filtercost.tex'), 'w').write('\n'.join(rows) + '\n')
json.dump(cost_summary, open('results/filter_cost_summary.json', 'w'), indent=1)
for f in ['tab_si_rho.tex', 'tab_si_gauss.tex', 'tab_si_filtercost.tex']:
    print('==', f); print(open(os.path.join(OUT, f)).read())

# ---- Table (repository only): DF threshold scan for the compiled filter (filter_compile.py)
FC = json.load(open('results/filter_compile.json'))
rows = []
for r in FC['thresholds']:
    e_ = int(np.round(np.log10(r['thresh']))); m_ = r['thresh'] / 10 ** e_
    th = rf"$10^{{{e_}}}$" if abs(m_ - 1) < 1e-9 else rf"${m_:.0f}\times10^{{{e_}}}$"
    rows.append(f"{th} & {r['L']} & {r['Lxi']} & {r['lambda_DF']:.2f} & ${r['dE0_mEh']:+.2f}$ & ${r['dD_pct']:+.2f}$ & {r['gap_mEh']:.1f}\\\\")
open(os.path.join(OUT, 'tab_si_dfscan.tex'), 'w').write('\n'.join(rows) + '\n')
print(open(os.path.join(OUT, 'tab_si_dfscan.tex')).read())
