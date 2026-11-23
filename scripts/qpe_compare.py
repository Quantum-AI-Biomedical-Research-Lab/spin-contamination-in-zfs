"""Gaussian filter versus phase estimation as a projector onto the naphthalene T1 state (Section 4.5, Table 3).

Same double-factorized block encoding and Toffoli count per controlled walk step as filter_compile.py.
  - Filter: mean walk steps of one Hadamard-test circuit (two controlled evolutions, t1, t2 ~ N(0, T^2)) at
    T = x/(E1 - E0) for the x values needed by the guarantees of filter_guarantee.json; QSVT degree of the
    coherent filter. Leakage of a state one gap above the target: exp(-x^2).
  - Textbook QPE with M walk steps: Fejer kernel, outcomes within (E1 - E0)/2 of the target accepted, averaged
    over the offset of the target phase from the grid. Phases are linearized, theta = E/lambda_DF, as in the
    chemical-accuracy cost of openfermion's compute_cost (pi lambda / 2 eps walk steps).
Writes results/qpe_projector_compare.json.
"""
import json, numpy as np
from scipy.special import jv

EPS = 1e-3
fc = json.load(open('results/filter_compile.json')); fg = json.load(open('results/filter_guarantee.json'))
lam = fc['chosen']['lambda_DF']; G = fc['E1'] - fc['E0']; step = fc['toffoli_per_controlled_step']

def walk_steps(x, eps=EPS):
    x = abs(x); kmax = int(x + 60 + 10 * np.sqrt(x + 1))
    tail = 2 * np.cumsum(np.abs(jv(np.arange(kmax, -1, -1), x)))[::-1]
    ok = np.nonzero(tail[1:] <= eps)[0]
    return int(ok[0]) if len(ok) else kmax

def cheb_degree(Tlam, eps=EPS):
    xs = np.cos(np.linspace(0, np.pi, 20001)); f = lambda x: np.exp(-0.5 * (x * Tlam) ** 2)
    for d in range(50, 20000, 50):
        if np.max(np.abs(np.polynomial.chebyshev.chebval(xs, np.polynomial.chebyshev.chebinterpolate(f, d)) - f(xs))) < eps:
            return d

QTAB = {}
def qf(v, eps=EPS):
    k = (round(v, 1), eps)
    if k not in QTAB:
        QTAB[k] = 2 * walk_steps(v, eps)
    return QTAB[k]

def filter_cost(x, eps=EPS):
    rng = np.random.default_rng(0); T = x / G
    t1 = rng.normal(0, T, 200000)[:20000]; t2 = rng.normal(0, T, 200000)[:20000]   # as in filter_compile.py
    ws = float(np.mean([qf(lam * abs(a), eps) + qf(lam * abs(b), eps) for a, b in zip(t1, t2)]))
    return {'x': x, 'eps_sim': eps, 'walk_steps': ws, 'toffoli': ws * step, 'qsvt_degree': cheb_degree(lam * T, eps), 'leak_1gap': float(np.exp(-x ** 2))}

def fejer(d, M):
    s = np.sin(d / 2); small = np.abs(s) < 1e-14
    return np.where(small, 1.0, (np.sin(M * d / 2) / (M * np.where(small, 1, s))) ** 2)

def qpe(M, nof=200):
    # Toffoli count: M controlled walk steps (as for the filter); compute_cost's chemical-accuracy QPE count,
    # results/filter_compile.json 'qpe_toffoli_chem_acc', uses the uncontrolled step (2730 instead of 2754 Toffolis)
    thk = 2 * np.pi * np.arange(M) / M; pt, pl = [], []
    for u in (np.arange(nof) + 0.5) / nof:
        th0 = 2 * np.pi * u / M; acc = np.abs(((thk - th0) + np.pi) % (2 * np.pi) - np.pi) < (G / 2) / lam
        pt.append(fejer(th0 - thk[acc], M).sum()); pl.append(fejer(th0 + G / lam - thk[acc], M).sum())
    return {'M': M, 'toffoli': M * step, 'target_accept': float(np.mean(pt)), 'leak_1gap_mean': float(np.mean(pl))}

ng = fg['naph_no']
xs = {'rule_3': 3.0}
for p in ['GSD', 'sfGSD']:
    for k in ['T1_exact', 'T01_exact', 'T1_meas', 'T01_meas']:
        xs[f'{p}_{k}'] = ng[p][k]
out = {'lambda_DF': lam, 'gap_Eh': G, 'toffoli_per_controlled_step': step, 'qpe_chem_acc_toffoli': fc['qpe_toffoli_chem_acc'],
       'qpe_chem_acc_iterations': fc['qpe_iterations'], 'filter': {}}
# certified protocol (naph_certificate.py, Table 2): total length and the simulation error of the error budget
nc = json.load(open('results/naph_certificate.json')); fgn = fg['naph_no']
x_cert = max(nc[p]['chosen']['T01'] for p in ['GSD', 'sfGSD']); eps_cert = min(fgn[p]['eps_sim_for_1e-4'] for p in ['GSD', 'sfGSD'])
cases = {k: (v, EPS) for k, v in xs.items()}
cases['rule_3_eps_cert'] = (3.0, eps_cert); cases['cert_T01'] = (x_cert, eps_cert)
done = {}
for k, (v, e) in cases.items():
    key = (round(v, 2), e)
    if key not in done:
        done[key] = filter_cost(v, e)
    out['filter'][k] = dict(done[key], x=v)
    print(k, f"{v:.2f}", f"eps {e:.1e}", f"{done[key]['toffoli']:.3e}", flush=True)
# measuring sigma of the pre-filtered state g_T1 psi (certificate input) by amplitude estimation of
# ||(H - E) phi|| / lambda_DF to 50% relative precision: pi / (2 * 0.5 a) repetitions of the coherent pre-filter
# (QSVT degree at T1) and one block-encoding call; trade-off with the certified total length (naph_certificate.json)
out['sigma_measurement'] = {}
for p in ['GSD', 'sfGSD']:
    for r in nc[p]['prefilter']:
        if r['x1'] == 0.0 or r['T01'] is None or r['sigmaH_mEh'] < 1e-3:
            continue
        a = r['sigmaH_mEh'] * 1e-3 / lam; reps = int(np.ceil(np.pi / a)); deg = cheb_degree(lam * r['x1'] / G, eps_cert)
        out['sigma_measurement'][f"{p}_x1_{r['x1']}"] = {'x1': r['x1'], 'sigma_mEh': r['sigmaH_mEh'], 'ae_reps': reps,
            'prefilter_degree': deg, 'toffoli': reps * (2 * deg + 1) * step, 'T01_total': r['T01']['total']}
        print(f"sigma measurement {p} x1={r['x1']:.2f}: sigma {r['sigmaH_mEh']:.2f} mEh, {reps} AE reps, degree {deg}, "
              f"{reps * (2 * deg + 1) * step:.2e} Toffolis, certified total T {r['T01']['total']:.2f}", flush=True)
out['qpe'] = {str(M): qpe(M) for M in [677, 1354, 2708, 5416, 10832, fc['qpe_iterations'], 29590]}
for k, v in out['filter'].items():
    print(f"filter {k:15s} x={v['x']:.2f} walk {v['walk_steps']:7.0f} Toffoli {v['toffoli']:.2e} ({v['toffoli']/fc['qpe_toffoli_chem_acc']:.2f} of QPE) QSVT deg {v['qsvt_degree']} leak {v['leak_1gap']:.1e}")
for k, v in out['qpe'].items():
    print(f"QPE M={v['M']:6d} Toffoli {v['toffoli']:.2e} target {v['target_accept']:.3f} leak {v['leak_1gap_mean']:.1e}")
json.dump(out, open('results/qpe_projector_compare.json', 'w'), indent=1)
