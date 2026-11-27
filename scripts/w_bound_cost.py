"""Compiled cost of the certified naphthalene protocol with the scalable spectral-width bound W_1 (eq 14), and the cost
of preparing the VQE state (Section 4.5, SI Section S5).

With W_1 in place of the exact W, the certificate needs a longer filter (results/w_bound.json) and, through item (i) of
the error budget, evolutions that are W_1/W times more accurate: eps_sim = 1e-4 Z / (2 W_1/|D| + 2e-4). The walk steps
of a Hadamard-test circuit and the Chebyshev degree of the coherent filter are recomputed exactly as in qpe_compare.py;
with the exact W the code reproduces results/certified_cost.json. State preparation: the ADAPT operators at chemical
accuracy (GSD n = 291, sfGSD n = 191) are counted as Pauli rotations, 8 per excitation component (an upper bound),
synthesized with 1.15 log2(1/eps_rot) + 9.2 T gates each (eps_rot = 1e-6 / number of rotations); a Toffoli is counted as
4 T gates for the comparison. Writes results/w_bound_cost.json.
"""
import json, warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.special import jv
import vqe_sparse as sps
fc = json.load(open('results/filter_compile.json')); fg = json.load(open('results/filter_guarantee.json'))
q = json.load(open('results/qpe_projector_compare.json')); cc = json.load(open('results/certified_cost.json'))
wb = json.load(open('results/w_bound.json'))
lam = fc['chosen']['lambda_DF']; G = fc['E1'] - fc['E0']; step = fc['toffoli_per_controlled_step']

def walk_steps(x, eps):
    x = abs(x); kmax = int(x + 60 + 10 * np.sqrt(x + 1))
    tail = 2 * np.cumsum(np.abs(jv(np.arange(kmax, -1, -1), x)))[::-1]
    ok = np.nonzero(tail[1:] <= eps)[0]; return int(ok[0]) if len(ok) else kmax

def cheb_degree(Tlam, eps):
    xs = np.cos(np.linspace(0, np.pi, 20001)); f = lambda x: np.exp(-0.5 * (x * Tlam) ** 2)
    for d in range(50, 20000, 50):
        if np.max(np.abs(np.polynomial.chebyshev.chebval(xs, np.polynomial.chebyshev.chebinterpolate(f, d)) - f(xs))) < eps:
            return d

QT = {}
def qf(v, eps):
    k = (round(v, 1), eps)
    if k not in QT:
        QT[k] = 2 * walk_steps(v, eps)
    return QT[k]

def filter_cost(x, eps):
    rng = np.random.default_rng(0); T = x / G
    t1 = rng.normal(0, T, 200000)[:20000]; t2 = rng.normal(0, T, 200000)[:20000]
    return float(np.mean([qf(lam * abs(a), eps) + qf(lam * abs(b), eps) for a, b in zip(t1, t2)])), cheb_degree(lam * T, eps)

fgn = fg['naph_no']; WD = fgn['W_over_D']; W1D = wb['naph_no']['W_pauli_over_D']
eps_ex = min(fgn[p]['eps_sim_for_1e-4'] for p in ['GSD', 'sfGSD']); Z = eps_ex * (2 * WD + 2e-4) / 1e-4
eps_w1 = 1e-4 * Z / (2 * W1D + 2e-4)
x_ex = q['filter']['cert_T01']['x']; x_w1 = max(wb['naph_no'][p]['T01_pauli_W'] for p in ['GSD', 'sfGSD'])
rate = cc['toffoli_per_circuit'] / cc['circuit_time_s']; nae = cc['ae_repetitions']
out = {}
for tag, x, e in [('exact_W', x_ex, eps_ex), ('W1_length_only', x_w1, eps_ex), ('W1', x_w1, eps_w1)]:
    ws, deg = filter_cost(x, e); tof = ws * step; tae = nae * 2 * deg * step
    out[tag] = {'x': x, 'eps_sim': e, 'walk_steps': ws, 'toffoli_circuit': tof, 'circuit_min': tof / rate / 60,
                'qsvt_degree': deg, 'ae_toffoli': tae, 'ae_days': tae / rate / 86400, 'frac_qpe': tof / fc['qpe_toffoli_chem_acc']}
spc = sps.SparseSpace(10, (6, 4)); prep = {}
for p in ['GSD', 'sfGSD']:
    o = json.load(open(f'results/naph_no_{p}.json')); h = o['history']
    n = h[next(i for i, x in enumerate(h) if x['dE'] < 1.6e-3)]['n']
    pool = sps.pool_sf(spc) if p == 'sfGSD' else sps.pool_gsd(spc)
    lab = {tuple(op.label) if isinstance(op.label, (list, tuple)) else op.label: op for op in pool}
    nrot = sum(8 * len(lab[tuple(l)].comps) for l in o['ops'][:n])
    tper = 1.15 * np.log2(nrot / 1e-6) + 9.2
    prep[p] = {'operators': n, 'rotations_max': nrot, 'T_gates': nrot * tper,
               'fraction_of_circuit': nrot * tper / (4 * out['exact_W']['toffoli_circuit'])}
out['state_preparation'] = prep
json.dump(out, open('results/w_bound_cost.json', 'w'), indent=1)
for k, v in out.items():
    print(k, {a: (f'{b:.4g}' if isinstance(b, float) else b) for a, b in v.items()})
