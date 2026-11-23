"""Penalized ADAPT-VQE for 20-qubit naphthalene in CASSCF natural orbitals (sparse implementation).

Cost F = <H> + mu (<S^2> - 2) + w (<H^2> - <H>^2), used for both the operator selection and the parameter
optimization (see penalty.py). The variance is evaluated with H shifted by a constant (the exact energy) for
numerical stability; the variance and the gradients do not depend on the shift.
The run stops `extra` operators after the energy error first falls below 1.6 mEh, at max_ops, or at the time limit.

Usage (from the repository root): python scripts/naph_penalty.py GSD|sfGSD MU W [extra=30] [max_ops=330] [hours=10]
Writes results/naph_penalty_<pool>_mu<MU>_w<W>.json.
"""
import json
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import minimize
from pyscf import fci
from zfs import d_tensor, HARTREE2CM
import vqe_sparse as sps

pname, mu, w = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
extra = int(sys.argv[4]) if len(sys.argv) > 4 else 30
max_ops = int(sys.argv[5]) if len(sys.argv) > 5 else 330
hours = float(sys.argv[6]) if len(sys.argv) > 6 else 10.0
CHEM = 1.6e-3

dat = np.load('results/naph_setup_no.npz')
ncas, nelecas, S = 10, (6, 4), 1.0
space = sps.SparseSpace(ncas, nelecas)
Hfull = sps.Hamiltonian(dat['h1'], dat['eri'], float(dat['ecore']), ncas, nelecas)
F_int, V, ci_ex, e_ex = dat['F'], dat['V'], dat['ci'].ravel(), float(dat['e_casci'])
shape = (space.na, space.nb)
Hs = lambda x: Hfull(x, shape) - e_ex * x            # shifted Hamiltonian
S2 = lambda x: fci.spin_op.contract_ss(x.reshape(shape), ncas, nelecas).ravel()


def M_apply(psi):
    Hp = Hs(psi)
    E = psi @ Hp
    F = E + w * (Hp @ Hp - E * E)
    Mp = (1 - 2 * w * E) * Hp
    if w:
        Mp = Mp + w * Hs(Hp)
    if mu:
        s2p = S2(psi)
        F += mu * (psi @ s2p - S * (S + 1))
        Mp = Mp + mu * s2p
    return F, Mp


def cost_and_grad(thetas, gens, ref):
    psi = sps._state(gens, thetas, ref)
    F, lam = M_apply(psi)
    grad = np.zeros(len(gens))
    phi = psi
    for k in range(len(gens) - 1, -1, -1):
        grad[k] = 2 * lam @ gens[k].dA(phi)
        phi = gens[k].apply(-thetas[k], phi)
        lam = gens[k].apply(-thetas[k], lam)
    return F, grad


def DE(Dt):
    Dp = V.T @ Dt @ V
    return 1.5 * Dp[2, 2] * HARTREE2CM


def proj_triplet(x):
    y = x.copy()
    for s in (2, 3, 4, 5):
        c = s * (s + 1)
        y = (S2(y) - c * y) / (2.0 - c)
    return y


def observe(psi):
    c = psi.reshape(shape)
    Hp = Hs(psi); E = psi @ Hp
    _, d2 = fci.direct_spin1.make_rdm12s(c, ncas, nelecas)
    Dv = DE(d_tensor(F_int, d2, S))
    pt = proj_triplet(psi); wt = float(pt @ pt)
    if wt < 1 - 1e-12:
        _, d2p = fci.direct_spin1.make_rdm12s((pt / np.sqrt(wt)).reshape(shape), ncas, nelecas)
        Dp = DE(d_tensor(F_int, d2p, S))
    else:
        Dp = Dv
    return {'dE': float(E), 'sigma_H': float(np.sqrt(max(Hp @ Hp - E * E, 0.0))), 'S2': float(psi @ S2(psi)),
            'D': float(Dv), 'D_proj': float(Dp), 'fid': float((psi @ ci_ex) ** 2)}


pool = {'GSD': lambda: sps.pool_gsd(space), 'sfGSD': lambda: sps.pool_sf(space)}[pname]()
ref = np.zeros(space.dim); ref[0] = 1.0
fn = f'results/naph_penalty_{pname}_mu{mu:g}_w{w:g}.json'
out = {'pool': pname, 'mu': mu, 'w': w, 'E_exact': e_ex, 'extra_after_chem': extra}
gens, thetas, chosen = [], np.zeros(0), []
psi = ref.copy()
hist = [dict(n=0, t=0.0, **observe(psi))]
t0 = time.time()
# resume from a checkpoint (operators and parameters are saved after every iteration)
try:
    old = json.load(open(fn))
except (FileNotFoundError, json.JSONDecodeError):
    old = {}
if old.get('thetas') and not old.get('finished'):
    bylabel = {tuple(o.label): o for o in pool}
    for lab in old['ops']:
        o = bylabel[tuple(tuple(x) if isinstance(x, list) else x for x in lab)]
        gens.append(sps.BlockGenerator(sps.op_matrix(space, o))); chosen.append(o.label)
    thetas = np.array(old['thetas'])
    psi = sps._state(gens, thetas, ref)
    hist = old['history']
    t0 -= hist[-1]['t']
    print(f'resumed at n={len(gens)}', flush=True)
for it in range(max_ops):
    _, Mp = M_apply(psi)
    ps, sig = psi.reshape(shape), Mp.reshape(shape)
    g = np.array([2 * sps.op_gradient(space, o, sig, ps) for o in pool])
    if np.abs(g).max() < 1e-5 or hist[-1]['dE'] < 1e-8:
        break
    ic = next((r['n'] for r in hist if r['dE'] < CHEM), None)
    if ic is not None and hist[-1]['n'] >= ic + extra:
        break
    if time.time() - t0 > 3600 * hours:
        break
    k = int(np.argmax(np.abs(g)))
    gens.append(sps.BlockGenerator(sps.op_matrix(space, pool[k]))); chosen.append(pool[k].label)
    res = minimize(cost_and_grad, np.append(thetas, 0.0), args=(gens, ref), jac=True, method='BFGS',
                   options={'gtol': 1e-9, 'maxiter': 4000})
    thetas = res.x
    psi = sps._state(gens, thetas, ref)
    rec = dict(n=len(gens), t=time.time() - t0, **observe(psi))
    hist.append(rec)
    print(f"{pname} mu={mu:g} w={w:g} n={rec['n']:3d} dE={rec['dE']:9.2e} sigmaH={1e3 * rec['sigma_H']:6.1f}mEh "
          f"S2={rec['S2']:.6f} D={rec['D']:.5f} t={rec['t']:.0f}s", flush=True)
    out.update({'history': hist, 'ops': [list(c) for c in chosen], 'thetas': thetas.tolist()})
    json.dump(out, open(fn + '.tmp', 'w'), default=float)
    __import__('os').replace(fn + '.tmp', fn)
out.update({'history': hist, 'ops': [list(c) for c in chosen], 'time_s': time.time() - t0, 'finished': True})
json.dump(out, open(fn, 'w'), default=float)
print(f'{pname} mu={mu:g} w={w:g} finished: {len(chosen)} operators, time {time.time() - t0:.0f} s', flush=True)
