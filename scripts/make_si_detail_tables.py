"""SI tables with the per-system values behind the ranges quoted in Section 4 of the main text.

tables/tab_si_errors.tex  error analysis at the first ADAPT iterate with chemical accuracy, per system and pool
tables/tab_si_system.tex  per-system checks: contaminant-scan slope, lowest state with S >= 2 above E0, filtered reference
                          determinant, W_1 lengthening
tables/tab_si_finiteshot.tex is extended by make_finite_shot_tables.py (column: change of the median length, exact sigma).
Usage (from the repository root): python scripts/make_si_detail_tables.py [outdir]
"""
import json, os, sys
import numpy as np
OUT = sys.argv[1] if len(sys.argv) > 1 else 'tables'
os.makedirs(OUT, exist_ok=True)
R = lambda f: json.load(open(os.path.join('results', f)))
SM = ['CH2', 'NH', 'NH_stretch', 'O2']; P3 = ['GSD', 'QEB', 'sfGSD']
NAME = {'CH2': r'\ce{CH2}', 'NH': 'NH', 'NH_stretch': r'NH ($2r_\mathrm{e}$)', 'O2': r'\ce{O2}', 'TMM': 'TMM'}
T = {x['name']: x for x in R('theory_check.json')}
N = R('naph_summary.json'); nb = R('naph_bounds.json'); eb = R('errbar_small.json'); en = R('errbar_naph_no.json')
f2 = lambda x: f'{x:.2f}'.replace('-', '$-$')
sg = lambda x: (f'{x:+.2f}' if abs(x) >= 0.005 else '0.00').replace('-', '$-$')
DASH = '--'

# ---- error analysis per system and pool
rows = []
for s in SM:
    fx = R(f'fix_{s}.json')['runs']; pf = R(f'propfilter_{s}.json')['runs']
    for i, p in enumerate(P3):
        c = T[s][p]['chem']; m = fx[p]['metrics']
        cells = [NAME[s] if i == 0 else '', p, f2(max(0.0, 100 * (1 - c['w']))), sg(m['err'] - m['err_proj']), sg(m['err_proj']), f"{c['F']:.4f}",
                 f2(100 * c['b11_split'] / T[s]['D']), f2(eb[s][p]['estimate_at_chem_pct']), f2(T[s][p]['filter']['bound_pct']),
                 f2(100 * (1 - pf[p]['gauss']['rule_spin']['Ng']))]
        rows.append(' & '.join(cells) + r'\\')
    rows.append(r'\addlinespace')
for o, lab in [('rohf', 'Naphthalene (ROHF)'), ('no', 'Naphthalene (NO)')]:
    for i, p in enumerate(['GSD', 'sfGSD']):
        h = N[o][p]
        cells = [lab if i == 0 else '', p, f2(max(0.0, 100 * h['w_contam_chem'])), sg(h['D_err_chem_pct'] - h['Dproj_err_chem_pct']), sg(h['Dproj_err_chem_pct']),
                 f"{h['fid_chem']:.4f}"]
        if o == 'no':
            cells += [f2(nb[p]['b11_split_pct']), f2(en[p]['estimate_at_chem_pct']), f2(nb[p]['filter_bound_pct']),
                      f2(100 * (1 - nb[p]['filter_stored']['gauss']['rule_spin']['Ng']))]
        else:
            cells += [DASH] * 4
        rows.append(' & '.join(cells) + r'\\')
open(os.path.join(OUT, 'tab_si_errors.tex'), 'w').write('\n'.join(rows) + '\n')

# ---- per-system checks
M = R('measure.json'); g2 = R('filter_guarantee.json'); wb = R('w_bound.json'); wc = R('w_bound_cost.json')
def slope(s):
    a = np.array(M[s]['controlled_scan'])[:5]
    return np.polyfit(np.log10(a[:, 0]), np.log10(a[:, 1]), 1)[0]
def det(s):
    cv = g2[s]['refdet']['curve_shift']
    at3 = [x[1] for x in cv if abs(x[0] - 3) < 1e-9][0]
    return at3, min(x[1] for x in cv if x[0] <= 4 + 1e-9)
rows = []
for s in SM + ['TMM', 'naph_no']:
    lab = NAME.get(s, 'Naphthalene (NO)')
    sl = f'{slope(s):.4f}' if s in M else DASH
    if s in g2:
        a3, a4 = det(s)
        wl = max(wb[s][p]['T01_pauli_W'] / wb[s][p]['T01_stored'] for p in P3) if s != 'naph_no' else wc['W1']['x'] / wc['exact_W']['x']
        gq = R(f'propfilter_{s}.json')['gaps']['E_lowest_S2'] * 1e3 if s != 'naph_no' else nb['E_lowest_Sge2_mEh']
        cells = [lab, sl, f2(gq), f2(a3), f2(a4), f2(100 * (wl - 1))]
    else:
        cells = [lab, sl, DASH, DASH, DASH, DASH]
    rows.append(' & '.join(cells) + r'\\')
open(os.path.join(OUT, 'tab_si_system.tex'), 'w').write('\n'.join(rows) + '\n')
print(open(os.path.join(OUT, 'tab_si_errors.tex')).read()); print(open(os.path.join(OUT, 'tab_si_system.tex')).read())
