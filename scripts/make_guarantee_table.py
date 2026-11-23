"""Table (repository only) body: rigorous error bounds at the first iteration with chemical accuracy (eqs 9-11).
Reads results/theory_check.json and results/naph_bounds.json (+ results/naph_filter.json for the filtered error).
Usage (from the repository root): python scripts/make_guarantee_table.py <tables_dir>"""
import json, os, sys
out_dir = sys.argv[1] if len(sys.argv) > 1 else 'tables'
os.makedirs(out_dir, exist_ok=True)
LAB = {'CH2': r'\ce{CH2}', 'NH': 'NH', 'NH_stretch': r'NH ($2r_\mathrm{e}$)', 'O2': r'\ce{O2}', 'TMM': 'TMM'}
def e(x):
    if x <= 0 or x < 1e-12:
        return r'$<10^{-12}$'
    m, k = f'{x:.1e}'.split('e'); return rf'${m}\times10^{{{int(k)}}}$'
f2 = lambda x: f'{x:.2f}'; f1 = lambda x: f'{x:.1f}'
rows = []
for r in json.load(open('results/theory_check.json')):
    name, D = r['name'], abs(r['D'])
    first = True
    for p in ['GSD', 'QEB', 'sfGSD']:
        c, fl = r[p].get('chem'), r[p].get('filter')
        if c is None:
            continue
        lab = LAB[name] if first else ''; first = False
        rows.append(' & '.join([lab, p, str(int(c['n'])), e(1 - c['F']), e(c['eps_mEh'] / r['G_M_mEh']),
                                f2(100 * c['err'] / D), f1(100 * c['b11_split'] / D), f1(100 * c['b11'] / D),
                                f1(100 * c['b11_prior'] / D), f'{fl["err_pct"]:.3f}', f2(fl['bound_pct'])]) + r'\\')
    rows.append(r'\midrule')
nb = json.load(open('results/naph_bounds.json')); nf = json.load(open('results/naph_filter.json'))
first = True
for p in ['GSD', 'sfGSD']:
    c = nb[p]; lab = 'Naphthalene (NO)' if first else ''; first = False
    ferr = abs(nf['runs'][p]['gauss']['rule_spin']['D'])
    rows.append(' & '.join([lab, p, str(int(c['n'])), e(c['infid']), e(c['eckart_M']), f2(c['err_pct']), f1(c['b11_split_pct']),
                            f1(c['b11_pct']), f1(c['b11_prior_pct']), f'{ferr:.3f}', f2(c['filter_bound_pct'])]) + r'\\')
open(f'{out_dir}/tab_si_guarantee.tex', 'w').write('\n'.join(rows) + '\n')
print('\n'.join(rows))
