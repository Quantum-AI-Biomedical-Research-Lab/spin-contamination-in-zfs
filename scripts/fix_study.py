"""Fix study on the 12-qubit systems: all variants of one system share one Hamiltonian and one set of orbitals.

Ansatz variants (penalty.py, w = 0):
  GSD, QEB, sfGSD           standard ADAPT
  GSD_mu1, QEB_mu1          spin-penalized ADAPT, mu = 1 Eh
  GSD_mu3, QEB_mu3          spin-penalized ADAPT, mu = 3 Eh
Refinements of the state at the first iterate with chemical accuracy:
  krylov k = 2..6           lowest Ritz vector of span{psi, H psi, ..., H^(k-1) psi}   (k = 2 is remedy C1)
  si1, si2                  one and two exact shift-invert steps psi <- (H - sigma)^-1 psi, sigma = E(psi) - 3 mEh
Writes results/fix_<system>.json (trajectories without states) and prints a summary.
Usage (from the repository root): python scripts/fix_study.py [systems...]
"""
import json
import sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import remedies
import penalty
from vqe import pool_gsd, pool_sf
from krylov_test import krylov_ritz
from shiftinvert_test import refine

VARIANTS = [('GSD', 0.0), ('QEB', 0.0), ('sfGSD', 0.0), ('GSD', 1.0), ('QEB', 1.0), ('GSD', 3.0), ('QEB', 3.0)]


def main(name):
    sysd = remedies.build(name); space = sysd['space']
    H, Xz, cD, E0, Dex = sysd['H'], sysd['Xz'], sysd['cD'], sysd['E0'], sysd['Dex']
    S2 = penalty.spin_matrix(space, space.norb, space.nelec)
    ws, Vs = np.linalg.eigh(S2); PT = Vs[:, np.abs(ws - 2.0) < 1e-6]
    Dof = lambda x: float(cD * x @ Xz @ x)
    out = {'D_exact': Dex, 'runs': {}}
    for pname, mu in VARIANTS:
        pool = pool_sf(space) if pname == 'sfGSD' else pool_gsd(space, qubit=(pname == 'QEB'))
        r = penalty.run(sysd, S2, PT, pool, pname, mu, 0.0, keep_states=True)
        states = r.pop('states')
        m = penalty.metrics(r['rows'], Dex)
        ic = m['n_chem']
        psi = states[ic]
        ref = {'k1': 100 * (Dof(psi) - Dex) / Dex}
        for k in range(2, 7):
            x = krylov_ritz(psi, lambda v: H @ v, k)
            ref[f'k{k}'] = 100 * (Dof(x) - Dex) / Dex
        for i, (x, info) in enumerate(refine(psi, lambda v: H @ v, space.dim), start=1):
            ref[f'si{i}'] = 100 * (Dof(x) - Dex) / Dex
            ref[f'si{i}_dE'] = float(x @ H @ x - E0)
            ref[f'si{i}_S2dev'] = float(x @ S2 @ x - 2.0)
        tag = pname if mu == 0 else f'{pname}_mu{mu:g}'
        r['metrics'] = m; r['refine'] = ref
        out['runs'][tag] = r
        print(f"{name:10s} {tag:9s} n_chem={ic} cnot_chem={m['cnot_chem']} err={m['err']:+.3f}% proj={m['err_proj']:+.3f}% "
              f"S2dev={m['S2dev']:.1e} D1cnot={m['cnot_D1']} | krylov k6 {ref['k6']:+.3f}% | si1 {ref['si1']:+.3f}% "
              f"si2 {ref['si2']:+.4f}%", flush=True)
    json.dump(out, open(f'results/fix_{name}.json', 'w'))


if __name__ == '__main__':
    for name in (sys.argv[1:] or ['CH2', 'NH', 'NH_stretch', 'O2']):
        main(name)
