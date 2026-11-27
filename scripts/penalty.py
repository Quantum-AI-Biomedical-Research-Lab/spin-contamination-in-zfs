"""Penalized ADAPT-VQE: select and optimize one cost that vanishes only at the exact target state.

    F(psi) = <H> + mu (<S^2> - S(S+1)) + w (<H^2> - <H>^2)

  mu  linear spin penalty. In the M_S = S sector <S^2> - S(S+1) >= 2(S+1)(1 - w_T) >= 0, so the term is a direct
      penalty on the contaminant weight; the exact triplet ground state is still the minimum of F.
  w   energy-variance weight. sigma_H^2 = 0 only for eigenstates, and eqs 6-7 bound the target-state error in D
      by about 2 sigma_X sigma_H / (E1 - E). The exact ground state is still the minimum of F for w >= 0.

Both the operator selection (largest |dF/dtheta| at theta = 0) and the parameter optimization use F, so an operator
chosen for its effect on the penalty terms is actually moved by the optimizer (the failure mode of remedy B).
dF/dtheta is the gradient of <psi|M|psi> with M = (1 - 2 w E) H + mu S^2 + w H^2 at the current point, evaluated
with the same adjoint sweep as the energy gradient.

Variants (all with the same Hamiltonian and orbitals within one process, so that they share the O2 orbital
orientation):  A (mu = w = 0), S<mu>, W<w>, S<mu>W<w>.
Stopping as for standard ADAPT: max |dF/dtheta| < 1e-5, energy error < 1e-10, or 80 operators.
Usage (from the repository root): python scripts/penalty.py [systems...]     writes results/penalty_<system>.json
"""
import json
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import minimize
from pyscf import fci
import remedies
from vqe import pool_gsd, pool_sf, Generator, state
from cnot_counts import cnots

CHEM = 1.6e-3
VARIANTS = {'GSD': [('A', 0.0, 0.0), ('S0.1', 0.1, 0.0), ('S1', 1.0, 0.0), ('W1', 0.0, 1.0), ('W3', 0.0, 3.0),
                    ('W10', 0.0, 10.0), ('S1W3', 1.0, 3.0)],
            'sfGSD': [('A', 0.0, 0.0), ('W1', 0.0, 1.0), ('W3', 0.0, 3.0), ('W10', 0.0, 10.0)]}


def spin_matrix(space, ncas, nelecas):
    na, nb, dim = len(space.sa), len(space.sb), space.dim
    S2 = np.zeros((dim, dim))
    for i in range(dim):
        e = np.zeros(dim); e[i] = 1
        S2[:, i] = fci.spin_op.contract_ss(e.reshape(na, nb), ncas, nelecas).ravel()
    return 0.5 * (S2 + S2.T)


def make_cost(H, S2, mu, w, starget):
    def M_apply(psi):
        Hp = H @ psi
        E = psi @ Hp
        s2p = S2 @ psi
        F = E + mu * (psi @ s2p - starget) + w * (Hp @ Hp - E * E)
        Mp = (1 - 2 * w * E) * Hp + mu * s2p + (w * (H @ Hp) if w else 0.0)
        return F, Mp

    def cost_and_grad(thetas, gens, ref):
        psi = state(gens, thetas, ref)
        F, lam = M_apply(psi)
        grad = np.zeros(len(gens))
        phi = psi
        for k in range(len(gens) - 1, -1, -1):
            grad[k] = 2 * lam @ gens[k].dA(phi)
            phi = gens[k].apply(-thetas[k], phi)
            lam = gens[k].apply(-thetas[k], lam)
        return F, grad
    return M_apply, cost_and_grad


def run(sysd, S2, PT, pool, pname, mu, w, S=1.0, max_ops=80, noise=0.0, rng=None, keep_states=False):
    """noise: standard error (Eh) of each energy in a two-term parameter-shift estimate of the selection gradient."""
    H, Xz, cD, E0, ref = sysd['H'], sysd['Xz'], sysd['cD'], sysd['E0'], sysd['ref']
    M_apply, cost_and_grad = make_cost(H, S2, mu, w, S * (S + 1))
    mats = [M for M, _ in pool]
    gens, thetas, chosen, states = [], np.zeros(0), [], [ref.copy()]
    psi = ref.copy()
    t0 = time.time()
    for it in range(max_ops):
        _, Mp = M_apply(psi)
        g = np.array([2 * Mp @ (A @ psi) for A in mats])
        if np.abs(g).max() < 1e-5 or psi @ (H @ psi) - E0 < 1e-10:
            break
        k = int(np.argmax(np.abs(g + rng.normal(0.0, np.sqrt(2.0) * noise, g.shape) if noise else g)))
        gens.append(Generator(mats[k])); chosen.append(pool[k][1])
        res = minimize(cost_and_grad, np.append(thetas, 0.0), args=(gens, ref), jac=True, method='BFGS',
                       options={'gtol': 1e-9, 'maxiter': 3000})
        thetas = res.x
        psi = state(gens, thetas, ref)
        states.append(psi.copy())
    qubit = pname == 'QEB'
    cn = np.cumsum([0] + [cnots(tuple(lab), qubit)[0] for lab in chosen])
    rows = []
    for n, p in enumerate(states):
        Hp = H @ p; E = p @ Hp
        t = PT @ (PT.T @ p); wt = t @ t; t /= np.sqrt(wt)
        rows.append({'n': n, 'cnot': int(cn[n]), 'dE': float(E - E0), 'sigma_H': float(np.sqrt(max(Hp @ Hp - E * E, 0))),
                     'S2': float(p @ S2 @ p), 'D': float(cD * p @ Xz @ p), 'D_proj': float(cD * t @ Xz @ t)})
    return {'rows': rows, 'ops': [list(map(lambda z: z if isinstance(z, str) else int(z), c)) for c in chosen],
            'time_s': time.time() - t0, 'mu': mu, 'w': w, **({'states': states} if keep_states else {})}


def metrics(rows, Dex):
    ic = next((i for i, r in enumerate(rows) if r['dE'] < CHEM), None)
    ok = [abs(r['D'] - Dex) < 0.01 * abs(Dex) for r in rows]
    i1 = next((i for i in range(len(rows)) if all(ok[i:])), None)
    out = {'n_chem': ic, 'n_final': len(rows) - 1, 'dE_final': rows[-1]['dE'],
           'n_D1': i1, 'cnot_D1': rows[i1]['cnot'] if i1 is not None else None}
    if ic is not None:
        r = rows[ic]
        out.update({'cnot_chem': r['cnot'], 'err': 100 * (r['D'] - Dex) / Dex,
                    'err_proj': 100 * (r['D_proj'] - Dex) / Dex, 'sigma_H_mEh': 1e3 * r['sigma_H'],
                    'S2dev': r['S2'] - 2.0})
    return out


if __name__ == '__main__':
    for name in (sys.argv[1:] or ['CH2', 'NH', 'NH_stretch', 'O2']):
        sysd = remedies.build(name)
        space = sysd['space']
        ncas = space.norb; nelecas = space.nelec
        S2 = spin_matrix(space, ncas, nelecas)
        ws, Vs = np.linalg.eigh(S2); PT = Vs[:, np.abs(ws - 2.0) < 1e-6]
        out = {'D_exact': sysd['Dex'], 'runs': {}}
        for pname in ['GSD', 'sfGSD']:
            pool = pool_gsd(space) if pname == 'GSD' else pool_sf(space)
            for tag, mu, w in VARIANTS[pname]:
                r = run(sysd, S2, PT, pool, pname, mu, w)
                r['metrics'] = metrics(r['rows'], sysd['Dex'])
                out['runs'][f'{pname}_{tag}'] = r
                m = r['metrics']
                print(f"{name:10s} {pname:5s} {tag:5s} n_chem={m['n_chem']} cnot_chem={m.get('cnot_chem')} "
                      f"err={m.get('err', float('nan')):+.3f}% proj={m.get('err_proj', float('nan')):+.3f}% "
                      f"sigmaH={m.get('sigma_H_mEh', float('nan')):.1f}mEh S2dev={m.get('S2dev', float('nan')):.1e} "
                      f"| D1: n={m['n_D1']} cnot={m['cnot_D1']} | final n={m['n_final']} dE={m['dE_final']:.1e} "
                      f"t={r['time_s']:.0f}s", flush=True)
                json.dump(out, open(f'results/penalty_{name}.json', 'w'))
