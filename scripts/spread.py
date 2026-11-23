"""Spread of 12-qubit ADAPT-VQE results under (i) random tie-breaking, (ii) random orbital gauge within degenerate
orbital pairs, and (iii) shot noise in the operator selection.

  tie       next operator chosen uniformly among those with |g_k| >= (1 - 1e-6) max |g| (exact and numerical ties)
  gauge     standard selection, but the degenerate active orbitals (pi pairs of NH and O2) are rotated by random
            angles before the CASCI; the exact problem is unchanged, the ADAPT path need not be
  noise<x>  next operator = argmax |g_k + xi_k|, xi_k ~ N(0, (sqrt(2) x mEh)^2): a two-term parameter-shift gradient
            from two energy estimates of standard deviation x mEh. Parameters are optimized on the exact energy.

For every run the trajectory (energy error, naive and projected D, <S^2>, cumulative CNOTs) is stored.
Usage (from the repository root): python scripts/spread.py [systems...]       writes results/spread_<system>_<pool>.json
"""
import json
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo, fci
import remedies
import theory_check as tc
from vqe import pool_gsd, pool_sf, adapt_vqe
from cnot_counts import cnots

NSEED_TIE, NSEED_GAUGE, NSEED_NOISE = 20, 20, 10
NOISE_MEH = [0.16, 0.5, 1.6]
DEGEN_SYSTEMS = {'NH', 'NH_stretch', 'O2'}


def setup_rot(name, seed):
    """theory_check.setup with random rotations inside degenerate pairs of active ROHF orbitals."""
    atom, ncas, nelecas, _ = tc.SYSTEMS[name]
    mol = gto.M(atom=atom, basis='cc-pvdz', spin=nelecas[0] - nelecas[1], verbose=0)
    mf = scf.ROHF(mol).run(conv_tol=1e-11)
    S = 0.5 * (nelecas[0] - nelecas[1])
    mc = mcscf.CASCI(mf, ncas, nelecas)
    C = tc.axis_gauge(mol, mf, mc.ncore, ncas)          # random rotations relative to the fixed axis gauge
    act = list(range(mc.ncore, mc.ncore + ncas))
    e = mf.mo_energy
    rng = np.random.default_rng(500 + seed)
    pairs, used = [], set()
    for i in act:
        for j in act:
            if j > i and i not in used and j not in used and abs(e[i] - e[j]) < 1e-5:
                pairs.append((i, j)); used |= {i, j}
    for i, j in pairs:
        t = rng.uniform(0, np.pi)
        ci, cj = C[:, i].copy(), C[:, j].copy()
        C[:, i], C[:, j] = np.cos(t) * ci + np.sin(t) * cj, -np.sin(t) * ci + np.cos(t) * cj
    mc.mo_coeff = C
    mc.fcisolver.conv_tol = 1e-12
    mc.fix_spin_(ss=S * (S + 1)); mc.kernel()
    F = tc.ss_mo_ints(mol, mc.mo_coeff[:, mc.ncore:mc.ncore + ncas])
    h1, ecore = mc.get_h1eff()
    eri = ao2mo.restore(1, mc.get_h2eff(), ncas)
    return (mol, mc, F, S, ncas, nelecas, h1, ecore, eri), len(pairs)


def build(name, gauge_seed=None):
    if gauge_seed is None:
        sysd = remedies.build(name); npairs = 0
    else:
        res, npairs = setup_rot(name, gauge_seed)
        orig = remedies.setup
        remedies.setup = lambda n: res
        try:
            sysd = remedies.build(name)
        finally:
            remedies.setup = orig
    return sysd, npairs


def s2_matrix(sysd, name):
    _, ncas, nelecas, _ = tc.SYSTEMS[name]
    sp = sysd['space']; na, nb = len(sp.sa), len(sp.sb); dim = sp.dim
    M = np.zeros((dim, dim))
    for i in range(dim):
        e = np.zeros(dim); e[i] = 1
        M[:, i] = fci.spin_op.contract_ss(e.reshape(na, nb), ncas, nelecas).ravel()
    return 0.5 * (M + M.T)


def projector_S1(S2m):
    w, U = np.linalg.eigh(S2m)
    Ut = U[:, np.abs(w - 2.0) < 1e-6]
    return Ut @ Ut.T


def one_run(sysd, P1, S2m, pool, pname, select):
    H, Xz, cD = sysd['H'], sysd['Xz'], sysd['cD']
    states = []
    score = None
    if select is not None:
        score = lambda psi, grads, mats: select(grads)
    hist, chosen, thetas, psi = adapt_vqe(sysd['space'], H, sysd['ref'], pool,
                                          observe=lambda p: (states.append(p.copy()), {})[1], max_ops=80,
                                          e_exact=sysd['E0'], score=score)
    cn = np.cumsum([0] + [cnots(tuple(l), pname == 'QEB')[0] for l in chosen])
    rows = []
    for n, p in enumerate(states):
        pt = P1 @ p; w = float(pt @ pt)
        Dp = float(cD * pt @ Xz @ pt / w)
        rows.append({'n': n, 'cnot': int(cn[n]), 'dE': float(p @ H @ p - sysd['E0']), 'D': float(cD * p @ Xz @ p),
                     'D_proj': Dp, 'S2': float(p @ S2m @ p)})
    return {'rows': rows, 'ops': [list(c) for c in chosen]}


def onehot(k, n):
    s = np.zeros(n); s[k] = 1.0
    return s


def make_select(kind, rng, x=None):
    if kind == 'tie':
        def sel(g):
            a = np.abs(g); cand = np.flatnonzero(a >= (1 - 1e-6) * a.max())
            return onehot(rng.choice(cand), len(g))
        return sel
    if kind == 'noise':
        sg = np.sqrt(2.0) * x * 1e-3
        return lambda g: np.abs(g + rng.normal(0.0, sg, g.shape))
    return None


if __name__ == '__main__':
    systems = sys.argv[1:] or ['CH2', 'NH', 'NH_stretch', 'O2']
    for name in systems:
        t0 = time.time()
        base, _ = build(name)
        S2m = s2_matrix(base, name); P1 = projector_S1(S2m)
        pools = {'GSD': pool_gsd(base['space']), 'QEB': pool_gsd(base['space'], qubit=True),
                 'sfGSD': pool_sf(base['space'])}
        out = {}
        for pname, pool in pools.items():
            res = {'D_exact': base['Dex'], 'standard': one_run(base, P1, S2m, pool, pname, None)}
            res['tie'] = [one_run(base, P1, S2m, pool, pname, make_select('tie', np.random.default_rng(s)))
                          for s in range(NSEED_TIE)]
            for x in NOISE_MEH:
                res[f'noise{x}'] = [one_run(base, P1, S2m, pool, pname,
                                            make_select('noise', np.random.default_rng(100 + s), x))
                                    for s in range(NSEED_NOISE)]
            out[pname] = res
            print(name, pname, 'done', round(time.time() - t0), 's', flush=True)
        if name in DEGEN_SYSTEMS:
            for s in range(NSEED_GAUGE):
                sysd, npairs = build(name, gauge_seed=s)
                pg = {'GSD': pool_gsd(sysd['space']), 'QEB': pool_gsd(sysd['space'], qubit=True),
                      'sfGSD': pool_sf(sysd['space'])}
                for pname, pool in pg.items():
                    r = one_run(sysd, P1, S2m, pool, pname, None)
                    r['D_exact'] = sysd['Dex']; r['npairs'] = npairs
                    out[pname].setdefault('gauge', []).append(r)
            print(name, 'gauge done', round(time.time() - t0), 's', flush=True)
        for pname in out:
            json.dump(out[pname], open(f'results/spread_{name}_{pname}.json', 'w'))
