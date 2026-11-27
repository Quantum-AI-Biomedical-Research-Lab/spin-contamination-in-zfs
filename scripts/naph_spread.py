"""Spread of 20-qubit naphthalene ADAPT-VQE trajectories (CASSCF natural orbitals) under perturbed operator selection.

Each run selects the next operator by argmax |g_k + xi_k|, xi_k ~ N(0, sigma_g^2), with sigma_g = sqrt(2) dE_noise
(a two-term parameter-shift gradient from two energy estimates, each with standard deviation dE_noise). Parameters are
optimized on the exact energy, so the perturbation acts on the operator choice only: it breaks ties and near-ties the
way finite measurement precision would. The run stops `extra` operators after the energy error first falls below
1.6 mEh, at max_ops, or at the time limit.

Usage (from the repository root): python scripts/naph_spread.py GSD|sfGSD SEED [dE_noise_mEh=0.16] [extra=30] [max_ops=330] [hours=5] [orbitals=no|rohf]
Writes results/naph_spread_<pool>_s<seed>.json (natural orbitals) or results/naph_spread_rohf_<pool>_s<seed>.json.
"""
import json
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pyscf import fci
from zfs import d_tensor, HARTREE2CM
import vqe_sparse as sps

pname, seed = sys.argv[1], int(sys.argv[2])
dE_noise = float(sys.argv[3]) * 1e-3 if len(sys.argv) > 3 else 0.16e-3
extra = int(sys.argv[4]) if len(sys.argv) > 4 else 30
max_ops = int(sys.argv[5]) if len(sys.argv) > 5 else 330
hours = float(sys.argv[6]) if len(sys.argv) > 6 else 5.0
orb = sys.argv[7] if len(sys.argv) > 7 else 'no'
assert orb in ('no', 'rohf')
sigma_g = np.sqrt(2.0) * dE_noise
rng = np.random.default_rng(1000 + seed)

dat = np.load('results/naph_setup_no.npz' if orb == 'no' else 'results/naph_setup.npz')
ncas, nelecas, S = 10, (6, 4), 1.0
space = sps.SparseSpace(ncas, nelecas)
H = sps.Hamiltonian(dat['h1'], dat['eri'], float(dat['ecore']), ncas, nelecas)
F, V, ci_ex, e_ex = dat['F'], dat['V'], dat['ci'].ravel(), float(dat['e_casci'])
shape = (space.na, space.nb)


def DE(Dt):
    Dp = V.T @ Dt @ V
    return 1.5 * Dp[2, 2] * HARTREE2CM, 0.5 * (Dp[0, 0] - Dp[1, 1]) * HARTREE2CM


def proj_triplet(x):
    y = x.copy()
    for s in (2, 3, 4, 5):
        c = s * (s + 1)
        y = (fci.spin_op.contract_ss(y.reshape(shape), ncas, nelecas).ravel() - c * y) / (2.0 - c)
    return y


def observe(psi):
    c = psi.reshape(shape)
    s2 = float(fci.spin_op.spin_square(c, ncas, nelecas)[0])
    _, d2 = fci.direct_spin1.make_rdm12s(c, ncas, nelecas)
    Dv, _ = DE(d_tensor(F, d2, S))
    pt = proj_triplet(psi); wt = float(pt @ pt)
    if wt < 1 - 1e-12:
        _, d2p = fci.direct_spin1.make_rdm12s((pt / np.sqrt(wt)).reshape(shape), ncas, nelecas)
        Dp, _ = DE(d_tensor(F, d2p, S))
    else:
        Dp = Dv
    return {'S2': s2, 'D': Dv, 'D_proj': Dp, 'w_target': wt, 'fid': float((psi @ ci_ex) ** 2)}


def select(grads):
    return int(np.argmax(np.abs(grads + rng.normal(0.0, sigma_g, grads.shape))))


pool = {'GSD': lambda: sps.pool_gsd(space), 'sfGSD': lambda: sps.pool_sf(space)}[pname]()
ref = np.zeros(space.dim); ref[0] = 1.0
out = {'pool': pname, 'seed': seed, 'orbitals': orb, 'dE_noise': dE_noise, 'sigma_g': sigma_g, 'E_exact': e_ex,
       'extra_after_chem': extra}
fn = f'results/naph_spread_{pname}_s{seed}.json' if orb == 'no' else f'results/naph_spread_rohf_{pname}_s{seed}.json'
hist_live, chosen_live = [], []


def log(rec):
    hist_live.append({k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in rec.items()})
    print(f"{pname} s{seed} n={rec['n']:3d} dE={rec['E'] - e_ex:9.2e} D={rec['D']:.5f} t={rec['t']:.0f}s", flush=True)
    out['history'] = hist_live
    json.dump(out, open(fn, 'w'), default=float)


t0 = time.time()
hist, chosen, thetas, psi = sps.adapt_vqe(space, H, ref, pool, observe=observe, max_ops=max_ops, gtol=1e-5,
                                         etol=1e-8, e_exact=e_ex, time_budget=3600 * hours, log=log,
                                         select=select, extra_after=extra)
out.update({'history': hist, 'ops': [list(c) for c in chosen], 'time_s': time.time() - t0, 'finished': True})
json.dump(out, open(fn, 'w'), default=float)
print(f'{pname} s{seed} finished: {len(chosen)} operators, time {time.time() - t0:.0f} s', flush=True)
