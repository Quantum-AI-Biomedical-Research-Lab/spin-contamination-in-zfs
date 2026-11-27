"""Shot cost of the penalty terms on the exact active-space triplet (QWC grouping, optimal allocation, seed 0).
  <H>          to 1.6 mEh                              (reference, as in measure.py)
  <S^2>        to 1.6 mEh / mu   for mu = 3 Eh          (spin penalty)
  <(H - E0)^2> to 1.6 mEh / w    for w = 3 / Eh         (variance penalty; CH2 and O2 only, many terms)
Usage (from the repository root): python scripts/penalty_shots.py     writes results/penalty_shots.json"""
import json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import openfermion as of
from pyscf import ao2mo
from measure import setup, shots
from fockmap import pyscf_to_fock, hamiltonian_op
EPS, MU, W = 1.6e-3, 3.0, 3.0
out = {}
for name in ['CH2', 'NH', 'NH_stretch', 'O2']:
    mol, mc, F, S, ncas, nelecas = setup(name)
    nq = 2 * ncas
    psi = pyscf_to_fock(mc.ci, ncas, nelecas)
    h1, ecore = mc.get_h1eff(); eri = ao2mo.restore(1, mc.get_h2eff(), ncas)
    Hop = hamiltonian_op(h1, eri, ecore, ncas)
    nH = shots(Hop, psi, EPS, nq)
    S2op = of.s_squared_operator(ncas)
    nS = shots(S2op, psi, EPS / MU, nq)
    res = {'H_terms': nH[2], 'H_groups': nH[3], 'H_shots': nH[1], 'S2_terms': nS[2], 'S2_groups': nS[3],
           'S2_shots': nS[1], 'S2_ratio': nS[1] / nH[1]}
    if name in ('CH2', 'O2'):
        Hq = of.jordan_wigner(Hop) - mc.e_tot
        H2 = Hq * Hq; H2.compress(1e-10)
        n2 = shots_q = None
        from measure import shots as _s
        # shots() takes a fermion operator; build the qubit version directly
        const = H2.terms.pop((), 0.0)
        tot = 0.0
        groups = of.group_into_tensor_product_basis_sets(H2, seed=0)
        tg = 0.0
        for _, g in groups.items():
            G = of.get_sparse_operator(g, n_qubits=nq); Gp = G @ psi
            tg += np.sqrt(max((np.vdot(Gp, Gp) - np.vdot(psi, Gp) ** 2).real, 0.0))
        res.update({'H2_terms': len(H2.terms), 'H2_groups': len(groups), 'H2_shots': (tg / (EPS / W)) ** 2,
                    'H2_ratio': (tg / (EPS / W)) ** 2 / nH[1]})
    out[name] = res
    print(name, {k: (f'{v:.3g}' if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
    json.dump(out, open('results/penalty_shots.json', 'w'), indent=1)
