"""Gate-level (logical) compilation of the Gaussian filter for 20-qubit triplet naphthalene.

Route: every Hadamard-test sample needs controlled real-time evolutions exp(-iHt1), exp(-iHt2). Each evolution is
compiled by qubitization (Low and Chuang) with a double-factorized (DF) block encoding of the active-space
Hamiltonian (von Burg et al.; Toffoli costing of Lee et al. 2021, as implemented in openfermion.resource_estimates):
  1. DF threshold: the largest threshold whose truncated Hamiltonian changes E0 by < 0.1 mEh and D by < 0.05%
     (exact diagonalization of the truncated Hamiltonian, so the filter converges to an eigenstate whose D is still
     accurate).
  2. lambda_DF and the Toffoli count per walk step and logical qubits from compute_cost (chi = 10, beta = 20 bits).
     The QPE control register is removed and one Hadamard-test control qubit added. Controlling a walk step costs
     at most one extra Toffoli per reflected ancilla qubit; we add that number.
  3. Walk steps for exp(-iHt) with error eps: 2q, with q the smallest degree of the Jacobi-Anger expansion with
     2 sum_{k>q} |J_k(lambda t)| <= eps (quantum signal processing on the walk; cos and sin parts in one sequence,
     a conservative count). Averaged over t1, t2 ~ N(0, T^2), T = 3/(E1 - E0).
  4. Shots: F N_D from naph_filter.json (Hadamard-test estimator, D to 1%). Surface-code footprint and run time
     from openfermion's cost_estimator (physical error rate 1e-3, 1 us cycle).
Writes results/filter_compile.json.
"""
import json
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.special import jv
from pyscf import fci
from openfermion.resource_estimates.df.factorize_df import factorize
from openfermion.resource_estimates.df.compute_cost_df import compute_cost
from openfermion.resource_estimates.surface_code_compilation.physical_costing import cost_estimator
from zfs import d_tensor, HARTREE2CM

t0 = time.time()
dat = np.load('results/naph_setup_no.npz')
h1, eri, ecore, F, V = dat['h1'], dat['eri'], float(dat['ecore']), dat['F'], dat['V']
norb, nelec = 10, (6, 4)
EPS = 1e-3                   # error of each simulated evolution (operator norm)
rng = np.random.default_rng(0)


def solve(eri_x):
    """lowest two triplets (no spin constraint, so that no root is skipped) and D of the lowest"""
    cis = fci.direct_spin1.FCI(); cis.conv_tol = 1e-11; cis.max_cycle = 300
    e, c = cis.kernel(h1, eri_x, norb, nelec, ecore=ecore, nroots=4)
    trip = [k for k in range(4) if abs(fci.spin_op.spin_square(c[k], norb, nelec)[0] - 2) < 1e-4]
    _, d2 = cis.make_rdm12s(c[trip[0]], norb, nelec)
    return float(e[trip[0]]), float(e[trip[1]]), float(1.5 * (V.T @ d_tensor(F, d2, 1.0) @ V)[2, 2] * HARTREE2CM)


def lambda_df(df_factors):
    T = h1 - 0.5 * np.einsum('illj->ij', eri) + np.einsum('llij->ij', eri)
    lamT = float(np.sum(np.abs(np.linalg.eigvalsh(T))))
    lamF = sum(0.25 * np.sum(np.abs(np.linalg.eigvalsh(df_factors[:, :, k]))) ** 2 for k in range(df_factors.shape[2]))
    return lamT + float(lamF)


def walk_steps(x, eps=EPS):
    """smallest q with 2 sum_{k>q} |J_k(x)| <= eps (Jacobi-Anger truncation of exp(-i x cos theta))"""
    x = abs(x)
    kmax = int(x + 60 + 10 * np.sqrt(x + 1))
    tail = 2 * np.cumsum(np.abs(jv(np.arange(kmax, -1, -1), x)))[::-1]      # tail[q] = 2 sum_{k>=q} |J_k|
    ok = np.nonzero(tail[1:] <= eps)[0]
    return int(ok[0]) if len(ok) else kmax


E0, E1, Dex = solve(eri)
out = {'E0': E0, 'E1': E1, 'D_exact': Dex, 'thresholds': []}
print(f"exact: E0={E0:.8f} gap={1e3*(E1-E0):.2f} mEh D={Dex:.6f}  t={time.time()-t0:.0f}s", flush=True)
chosen = None
for thr in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
    eri_rr, LR, L, Lxi = factorize(eri, thr)
    e0, e1, D = solve(eri_rr)
    lam = lambda_df(LR)
    rec = {'thresh': thr, 'L': int(L), 'Lxi': int(Lxi), 'lambda_DF': lam, 'dE0_mEh': 1e3 * (e0 - E0), 'dD_pct': 100 * (D - Dex) / Dex,
           'gap_mEh': 1e3 * (e1 - e0)}
    out['thresholds'].append(rec)
    print(rec, f"t={time.time()-t0:.0f}s", flush=True)
    if chosen is None and abs(rec['dE0_mEh']) < 0.1 and abs(rec['dD_pct']) < 0.05:
        chosen = rec
out['chosen'] = chosen
lam, L, Lxi = chosen['lambda_DF'], chosen['L'], chosen['Lxi']
n = 2 * norb
stps = compute_cost(n, lam, 1.6e-3, L=L, Lxi=Lxi, chi=10, beta=20, stps=20000)[0]
step, total_qpe, anc = compute_cost(n, lam, 1.6e-3, L=L, Lxi=Lxi, chi=10, beta=20, stps=stps)
iters = total_qpe // step
ac1 = 2 * int(np.ceil(np.log2(iters + 1))) - 1
nL = int(np.ceil(np.log2(L + 1)))
reflected = nL + 2 + int(np.ceil(np.log2(n // 2))) + 2 + 10 + 1        # prepare registers that the walk reflects
logical = anc - ac1 + 1
cstep = step + reflected
gap = E1 - E0; T = 3.0 / gap
t1 = rng.normal(0, T, 200000); t2 = rng.normal(0, T, 200000); td = rng.normal(0, np.sqrt(2) * T, 200000)
qtab = {}
qf = lambda x: qtab.setdefault(round(x, 1), 2 * walk_steps(x))
ws_num = np.array([qf(lam * abs(a)) + qf(lam * abs(b)) for a, b in zip(t1[:20000], t2[:20000])])
ws_den = np.array([qf(lam * abs(a)) for a in td[:20000]])
ws_max = qf(lam * 4 * T) * 2                                            # both times at 4 standard deviations
NF = json.load(open('results/naph_filter.json'))
c = NF['cost']; Ng = NF['runs']['GSD']['gauss']['rule']['Ng']
ND = (c['sum_sigma_g'] / (0.01 * abs(c['X_mean']))) ** 2
Fshot = (c['sum_sqrt_m2_g'] / c['sum_sigma_g']) ** 2 / Ng ** 2
shots = Fshot * ND
tof_circ = float(ws_num.mean()) * cstep
tof_total = shots * tof_circ + ND * float(ws_den.mean()) * cstep * 0.0   # denominator: ~1% of the shots suffice (variance <= 1)
best, params = cost_estimator(logical, int(np.ceil(tof_circ)), physical_error_rate=1e-3)
best_max, _ = cost_estimator(logical, int(np.ceil(ws_max * cstep)), physical_error_rate=1e-3)
out.update({'lambda_pauli': c['lam_H'], 'T': T, 'eps': EPS, 'toffoli_per_walk_step': step, 'toffoli_per_controlled_step': cstep,
            'logical_qubits': int(logical), 'qpe_register_removed': int(ac1),
            'walk_steps_mean_numerator': float(ws_num.mean()), 'walk_steps_mean_denominator': float(ws_den.mean()),
            'walk_steps_max_4sigma': int(ws_max), 'lambda_T_over_walk': float(ws_num.mean() / (lam * 2 * T * np.sqrt(2 / np.pi))),
            'toffoli_per_circuit_mean': tof_circ, 'toffoli_per_circuit_max': ws_max * cstep,
            'shots': shots, 'N_D': ND, 'F': Fshot, 'toffoli_total': tof_total,
            'physical_qubits': best.physical_qubit_count, 'circuit_time_s': best.duration.total_seconds(),
            'circuit_time_max_s': best_max.duration.total_seconds(), 'code_distance': params.logical_data_qubit_distance,
            'total_time_days_sequential': shots * best.duration.total_seconds() / 86400})
print(json.dumps({k: v for k, v in out.items() if k != 'thresholds'}, indent=1, default=float), flush=True)
json.dump(out, open('results/filter_compile.json', 'w'), indent=1, default=float)

# ---- reference points: (a) one QPE energy estimate to chemical accuracy with the same block encoding;
# (b) coherent alternative: Gaussian filter as a QSVT polynomial in H/lambda and amplitude estimation of <X>
out['qpe_toffoli_chem_acc'] = int(total_qpe)
out['qpe_iterations'] = int(iters)


def cheb_degree(Tlam, x0=0.0, eps=EPS):
    """smallest Chebyshev degree approximating exp(-(x - x0)^2 Tlam^2 / 2) on [-1, 1] to max error eps"""
    xs = np.cos(np.linspace(0, np.pi, 20001))
    f = lambda x: np.exp(-0.5 * ((x - x0) * Tlam) ** 2)
    for d in range(50, 20000, 50):
        c = np.polynomial.chebyshev.chebinterpolate(f, d)
        if np.max(np.abs(np.polynomial.chebyshev.chebval(xs, c) - f(xs))) < eps:
            return d
    return None


dq = cheb_degree(lam * T)
alphaX = c['X_l1'] + abs(c.get('X_identity', 0.0))
epsX = 0.01 * abs(c['X_mean'])
n_ae = int(np.ceil(np.pi * alphaX / (2 * epsX)))           # coherent repetitions for amplitude estimation (leading order)
tof_ae = n_ae * 2 * dq * cstep                             # filter and its inverse in every repetition
out.update({'qsvt_degree': dq, 'ae_repetitions': n_ae, 'ae_toffoli_total': tof_ae,
            'toffoli_rate_per_s': tof_circ / best.duration.total_seconds(),
            'ae_time_days': tof_ae / (tof_circ / best.duration.total_seconds()) / 86400})
print({k: out[k] for k in ['qpe_toffoli_chem_acc', 'qpe_iterations', 'qsvt_degree', 'ae_repetitions', 'ae_toffoli_total',
                           'toffoli_rate_per_s', 'ae_time_days']}, flush=True)
json.dump(out, open('results/filter_compile.json', 'w'), indent=1, default=float)
