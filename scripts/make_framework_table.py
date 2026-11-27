"""Table 2 of the manuscript (tables/tab_framework.tex): certified accuracy and cost of the framework.

Per system, from results/filter_guarantee.json, results/naph_certificate.json and results/filter_cost_summary.json:
the gap; the largest infidelity of the ADAPT states at chemical accuracy over the pools; the filter length T(E1 - E0)
that proves |D_g - D| <= 0.1% of |D| (largest over the pools) (a) from eqs 9 and 11 with the exact fidelity and
exact-state constants, (b) from the measured-input certificate, eqs 12 and 13, with the true gap as the gap bound
and (c) with a gap bound 10% below the true gap; the spectral width W of X and its scalable bound W1 (results/w_bound.json); the simulation error per evolution that
keeps the implementation error below 0.01% of |D| (the largest actual error at T(E1 - E0) = 3 goes to
framework_table.json only, since Table 1 already lists the filtered errors); and, for the reference
determinant as input, eps/(G'/2). For naphthalene, (b) and (c) use a pre-filter (naph_certificate.py).
Usage (from the repository root): python scripts/make_framework_table.py [outdir]
"""
import json, os, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else 'tables'
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root (scripts/ is one level down)
R = lambda f: json.load(open(os.path.join(HERE, 'results', f)))
g, nc, wb = R('filter_guarantee.json'), R('naph_certificate.json'), R('w_bound.json')
NAME = {'CH2': r'\ce{CH2}', 'NH': 'NH', 'NH_stretch': r'NH ($2r_\mathrm{e}$)', 'O2': r'\ce{O2}', 'naph_no': 'Naphthalene (NO)'}
f2 = lambda v: '--' if v is None else f'{v:.2f}'
rows, summ = [], {}
for s in ['CH2', 'NH', 'NH_stretch', 'O2', 'naph_no']:
    d = g[s]; pools = ['GSD', 'sfGSD'] if s == 'naph_no' else ['GSD', 'QEB', 'sfGSD']
    infid = max(1 - d[p]['F'] for p in pools)
    tex = max(d[p]['T01_exact'] for p in pools)
    if s == 'naph_no':
        tc = max(nc[p]['chosen']['T01'] for p in pools)
        tc90 = max(nc[p]['chosen']['T01_gap90'] for p in pools)
        err3 = max(abs(next(y for x, y, n in d[p]['curve_shift'] if abs(x - 3.0) < 1e-9)) for p in pools)
        eps_sim = min(d[p]['eps_sim_for_1e-4'] for p in pools)
    else:
        tc = max(d[p]['cert_T01'] for p in pools); tc90 = max(d[p]['cert_T01_gap90'] for p in pools)
        err3 = max(d[p]['curve_shift'][30][1] for p in pools)
        eps_sim = min(d[p]['eps_sim_for_1e-4'] for p in pools)
    r = d['refdet']; ratio = r['eps_mEh'] / (d['Gp_mEh'] / 2)
    summ[s] = dict(infid=infid, T01_exact=tex, T01_cert=tc, T01_cert_gap90=tc90, W=d['W_over_D'], W1=wb[s]['W_pauli_over_D'], eps_sim=eps_sim,
                   err_T3=err3, ref_ratio=ratio, ref_T01_cert=r.get('cert_T01'))
    rows.append(f"{NAME[s]} & {d['gap_mEh']:.2f} & {100*infid:.2f} & {tex:.2f} & {tc:.2f} & {f2(tc90)} & {d['W_over_D']:.2f} & {wb[s]['W_pauli_over_D']:.2f} & "
                f"{eps_sim*1e6:.2f} & {ratio:.2f}\\\\")   # err_T3 is kept in framework_table.json only; Table 1 shows the filtered errors
os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, 'tab_framework.tex'), 'w').write('\n'.join(rows) + '\n')
json.dump(summ, open(os.path.join(HERE, 'results', 'framework_table.json'), 'w'), indent=1)
print('\n'.join(rows))
