"""Best case for the measurable error estimate b (formerly SI eq S5; repository only): replace the variance bound on s by the exact s.

b = 2|c_D| sigma_X(psi) [ (delta/(2(S+1)))^(1/2) + s_var ],   s_var = sigma_H / (sqrt(w) (E1 - E)).
Any exact-answer-free bound on s (for example from higher Hamiltonian moments) gives s >= s_true, so
b_best = b (a + s_true) / (a + s_var), with a = (delta/(2(S+1)))^(1/2) and s_true from the fidelity
|<T|T~>|^2 = |<T|psi>|^2 / w, is the smallest estimate that this form of b can reach.
Evaluated at the first iterate with chemical accuracy. Uses results/errbar_*.json and the run histories.
Writes results/errbar_ceiling.json.
"""
import json
import numpy as np

CHEM = 1.6e-3
out = {}


def entry(rows, hist, gap, Dex, e_hist0=0.0):
    n = min(len(rows), len(hist))
    ic = next((i for i in range(n) if rows[i]['dE'] < CHEM), None)
    if ic is None:
        return None
    r, h = rows[ic], hist[ic]
    a = np.sqrt(max(r['delta'], 0.0) / 6.0)
    s_var = r['sigma_H'] / (np.sqrt(r['w']) * (gap - r['dE']))
    s_true = np.sqrt(max(1.0 - h['fid'] / r['w'], 0.0))
    b = r['b_est']
    return {'n_chem': ic, 'err_pct': 100 * r['err'] / abs(Dex), 'b_pct': 100 * b / abs(Dex),
            'b_exact_s_pct': 100 * b * (a + s_true) / (a + s_var) / abs(Dex), 's_var': s_var, 's_true': s_true,
            'dE_check': abs(r['dE'] - h.get('dE', h['E'] - e_hist0))}


small = json.load(open('results/errbar_small.json'))
for name in ['CH2', 'NH', 'NH_stretch', 'O2']:
    run = json.load(open(f'results/{name}.json'))
    for p in ['GSD', 'QEB', 'sfGSD']:
        rows = json.load(open(f'results/errbar_{name}_{p}.json'))
        e = entry(rows, run['runs'][p]['history'], small[name]['gap_mEh'] * 1e-3, small[name]['D'])
        out.setdefault(name, {})[p] = e
ref = json.load(open('results/errbar_naph_no.json'))
for p in ['GSD', 'sfGSD']:
    rows = json.load(open(f'results/errbar_naph_no_{p}.json'))
    run = json.load(open(f'results/naph_no_{p}.json'))
    out.setdefault('naph_no', {})[p] = entry(rows, run['history'], ref['gap_mEh'] * 1e-3, ref['D'], run['E_exact'])
json.dump(out, open('results/errbar_ceiling.json', 'w'), indent=1)
for s in out:
    for p, e in out[s].items():
        if e:
            assert e['dE_check'] < 1e-8, (s, p)          # errbar rows and history refer to the same iterate
            print(f"{s:10s} {p:5s} err {e['err_pct']:5.2f}%  b {e['b_pct']:6.1f}%  b(exact s) {e['b_exact_s_pct']:5.1f}%  "
                  f"s_var {e['s_var']:.3f}  s_true {e['s_true']:.3f}")
