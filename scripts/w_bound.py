"""Scalable bound on the spectral width W of X for the accuracy certificate (Section 2.3, Section 4.5, SI Section S5).

The certificate (eq 13) needs the spectral width W = c_D (x_max - x_min) of X in the sector M_S = S. The reported
certificate uses the exact W from a Lanczos calculation in the sector, which scales like the exact problem. Any upper
bound on W keeps the certificate valid. A bound that is cheap for any system size is the Pauli 1-norm of the
Jordan-Wigner image of X without its identity term:  x_max - x_min <= 2 ||X - c_I|| <= 2 sum_{P != I} |c_P|.
For every system this script computes W_1 = 2 c_D sum_{P != I} |c_P|, its ratio to the exact W, and the filter length
T (E1 - E0) at which the certificate proves 0.1% with W_1 instead of W, using the stored measured inputs
(results/filter_guarantee.json; naphthalene: pre-filter T1 = 0.25/(E1 - E0) from results/naph_certificate.json),
with the true gap and with a gap bound 10% below it. As a check, the same code with the exact W reproduces the
stored lengths. Writes results/w_bound.json.
"""
import json, warnings; warnings.filterwarnings('ignore')
import numpy as np
import openfermion as of
from fockmap import d_component_op, d_prefactor
from zfs import HARTREE2CM
from filter_guarantee_lib import cert_inputs, cert_bound_pct

def pauli_W(Fzz, ncas, S, cD):
    q = of.jordan_wigner(d_component_op(Fzz, ncas, S))
    norm1 = sum(abs(c) for t, c in q.terms.items() if t != ())
    return 2 * abs(cD) * norm1, len(q.terms)

def bisect(fun, target, hi):
    lo = 0.0
    if fun(hi) > target:
        return None
    for _ in range(200):
        m = 0.5 * (lo + hi); lo, hi = (m, hi) if fun(m) > target else (lo, m)
    return hi

def cert_T(routes, W, gap, scale=1.0, target=0.1):
    best = None
    for r in routes:
        s, Dl, Df = r['sigma_mEh'] * 1e-3, scale * r['Delta_mEh'] * 1e-3, scale * r['Delta_filter_mEh'] * 1e-3
        ci = cert_inputs(s, Dl)
        if ci is None or not ci[0] < Df / 2:
            continue
        x = bisect(lambda T: cert_bound_pct(T, ci[0], ci[1], Df, W), target, 50.0 / gap)
        if x is not None:
            best = x * gap if best is None else min(best, x * gap)
    return best

def cert_T_naph(r, GM, Gp, W, gap, scale=1.0, target=0.1):
    x1 = r['x1']; dc = r['centre_offset_mEh'] * 1e-3; best = None
    for s, Dl, Df in [(r['sigmaH_mEh'] * 1e-3, GM, Gp), (r['sigmaHp_mEh'] * 1e-3, Gp, Gp)]:
        Dl, Df = scale * Dl, scale * Df
        ci = cert_inputs(s, Dl)
        if ci is None:
            continue
        ec = dc + ci[0]
        if not ec < Df / 2:
            continue
        x2 = bisect(lambda T: cert_bound_pct(T, ec, ci[1], Df, W), target, 50.0 / gap)
        if x2 is not None:
            tot = float(np.hypot(x1, x2 * gap)); best = tot if best is None else min(best, tot)
    return best

fg = json.load(open('results/filter_guarantee.json')); out = {}
from remedies import build
for name in ['CH2', 'NH', 'NH_stretch', 'O2']:
    sysd = build(name); rec = fg[name]
    W1, nterms = pauli_W(sysd['Fp'][2, 2], sysd['ncas'], sysd['S'], sysd['cD'])
    W1D = W1 / abs(sysd['Dex']); WD = rec['W_over_D']; gap = rec['gap_mEh'] * 1e-3
    o = {'W_exact_over_D': WD, 'W_pauli_over_D': W1D, 'ratio': W1D / WD, 'pauli_terms': nterms}
    for p in ['GSD', 'QEB', 'sfGSD']:
        rt = rec[p]['cert_routes']
        o[p] = {'T01_exact_W_check': cert_T(rt, WD, gap), 'T01_stored': rec[p]['cert_T01'],
                'T01_pauli_W': cert_T(rt, W1D, gap), 'T01_gap90_pauli_W': cert_T(rt, W1D, gap, 0.9),
                'T01_gap90_stored': rec[p]['cert_T01_gap90']}
    out[name] = o; print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items() if not isinstance(v, dict)},
                        {p: {k: round(v, 3) for k, v in o[p].items() if v is not None} for p in ['GSD', 'QEB', 'sfGSD']}, flush=True)

dat = np.load('results/naph_setup_no.npz')
Fp = np.einsum('ai,bj,ab...->ij...', dat['V'], dat['V'], dat['F']); cD = 1.5 * d_prefactor(1.0) * HARTREE2CM
nc = json.load(open('results/naph_certificate.json')); nb = json.load(open('results/naph_bounds.json'))
gap = nb['gap_mEh'] * 1e-3; GM = nb['G_M_mEh'] * 1e-3; Gp = nb['Gp_lower_mEh'] * 1e-3; WD = nc['W_over_D']
Dex = json.load(open('results/naph_reference_no.json'))['D_exact_NO']
W1, nterms = pauli_W(Fp[2, 2], 10, 1.0, cD); W1D = W1 / abs(Dex)
o = {'W_exact_over_D': WD, 'W_pauli_over_D': W1D, 'ratio': W1D / WD, 'pauli_terms': nterms}
for p in ['GSD', 'sfGSD']:
    r = next(q for q in nc[p]['prefilter'] if q['x1'] == 0.25)
    o[p] = {'T01_exact_W_check': cert_T_naph(r, GM, Gp, WD, gap), 'T01_stored': nc[p]['chosen']['T01'],
            'T01_pauli_W': cert_T_naph(r, GM, Gp, W1D, gap), 'T01_gap90_pauli_W': cert_T_naph(r, GM, Gp, W1D, gap, 0.9),
            'T01_gap90_stored': nc[p]['chosen']['T01_gap90']}
out['naph_no'] = o
print('naph_no', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items() if not isinstance(v, dict)},
      {p: {k: round(v, 3) for k, v in o[p].items() if v is not None} for p in ['GSD', 'sfGSD']})
json.dump(out, open('results/w_bound.json', 'w'), indent=1)
