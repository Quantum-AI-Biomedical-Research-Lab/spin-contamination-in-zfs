"""Krylov refinement of D: evaluate D on the lowest Ritz vector of span{psi, H psi, ..., H^(k-1) psi}, k = 1..6,
at every iterate of standard ADAPT (GSD, QEB, sfGSD) and of spin-penalized ADAPT (GSD and QEB, mu = 3 Eh).
k = 1 is the plain VQE value and k = 2 the single Lanczos step (remedy C1).
Usage (from the repository root): python scripts/krylov_test.py [systems...]     writes results/krylov_<system>.json"""
import json, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import remedies, penalty
from vqe import pool_gsd, pool_sf

KMAX = 6


def krylov_ritz(psi, Hm, k):
    V = [psi / np.linalg.norm(psi)]
    for j in range(k - 1):
        w = Hm(V[-1])
        for v in V:                      # full reorthogonalization (twice for stability)
            w = w - (v @ w) * v
        for v in V:
            w = w - (v @ w) * v
        nw = np.linalg.norm(w)
        if nw < 1e-10:
            break
        V.append(w / nw)
    B = np.array(V).T
    HB = np.array([Hm(v) for v in V]).T
    T = B.T @ HB; T = 0.5 * (T + T.T)
    e, c = np.linalg.eigh(T)
    x = B @ c[:, 0]
    return x / np.linalg.norm(x)


def analyse(states, rows, sysd, Hm, Xm, Dex):
    cD, E0 = sysd['cD'], sysd['E0']
    for r, psi in zip(rows, states):
        for k in range(2, KMAX + 1):
            x = krylov_ritz(psi, Hm, k)
            r[f'D_k{k}'] = float(cD * x @ Xm(x)); r[f'dE_k{k}'] = float(x @ Hm(x) - E0)
    res = {}
    ic = next((i for i, r in enumerate(rows) if r['dE'] < 1.6e-3), None)
    for k in range(1, KMAX + 1):
        key = 'D' if k == 1 else f'D_k{k}'
        ok = [abs(r[key] - Dex) < 0.01 * abs(Dex) for r in rows]
        i1 = next((i for i in range(len(rows)) if all(ok[i:])), None)
        res[k] = {'err_chem': 100 * (rows[ic][key] - Dex) / Dex if ic is not None else None,
                  'n_D1': i1, 'cnot_D1': rows[i1]['cnot'] if i1 is not None and 'cnot' in rows[i1] else None}
    return res, ic


if __name__ == '__main__':
    for name in (sys.argv[1:] or ['CH2', 'NH', 'NH_stretch', 'O2']):
        sysd = remedies.build(name); space = sysd['space']; H, Xz = sysd['H'], sysd['Xz']
        S2 = penalty.spin_matrix(space, space.norb, space.nelec)
        ws, Vs = np.linalg.eigh(S2); PT = Vs[:, np.abs(ws - 2.0) < 1e-6]
        out = {'D_exact': sysd['Dex']}
        for pname, mu in [('GSD', 0.0), ('QEB', 0.0), ('sfGSD', 0.0), ('GSD', 3.0), ('QEB', 3.0)]:
            pool = pool_sf(space) if pname == 'sfGSD' else pool_gsd(space, qubit=(pname == 'QEB'))
            r = penalty.run(sysd, S2, PT, pool, pname, mu, 0.0, keep_states=True)
            res, ic = analyse(r['states'], r['rows'], sysd, lambda v: H @ v, lambda v: Xz @ v, sysd['Dex'])
            tag = f'{pname}_mu{mu:g}'
            out[tag] = {'metrics': res, 'n_chem': ic, 'rows': r['rows']}
            print(f"{name:10s} {tag:9s} " + ' | '.join(f"k={k}: {res[k]['err_chem']:+.3f}% ({res[k]['cnot_D1']})" for k in res), flush=True)
        json.dump(out, open(f'results/krylov_{name}.json', 'w'))
