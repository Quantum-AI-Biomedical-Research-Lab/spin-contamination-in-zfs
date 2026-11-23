"""Sensitivity of the QWC shot estimates to the randomized greedy grouping of OpenFermion.

For each 12-qubit system: energy (1.6 mEh), principal-axis D (1% of |D|) and the full D tensor with shared
grouping (every Cartesian component to 1% of |D|), each regrouped with seeds 0..NSEED-1.
Writes results/shots_seeds.json with min / median / max over seeds.
"""
import json
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import openfermion as of
from pyscf import ao2mo, fci
from theory_check import setup
from fockmap import pyscf_to_fock, hamiltonian_op, d_component_op, d_prefactor
from zfs import d_tensor, HARTREE2CM

NSEED = 10


def pauli_vectors(terms, psi, nq):
    return {t: of.get_sparse_operator(of.QubitOperator(t), n_qubits=nq) @ psi for t in terms}


def grouped_sigmas(qops, Pv, psi, seed):
    union = of.QubitOperator()
    for q in qops:
        for t in q.terms:
            union += of.QubitOperator(t, 1.0)
    groups = of.group_into_tensor_product_basis_sets(union, seed=seed)
    sig = np.zeros((len(groups), len(qops)))
    for gi, (_, g) in enumerate(groups.items()):
        for ci, q in enumerate(qops):
            v = np.zeros_like(psi); m = 0.0
            for t in g.terms:
                c = q.terms.get(t)
                if c is not None:
                    v = v + c * Pv[t]
            if np.any(v):
                sig[gi, ci] = np.sqrt(max((np.vdot(v, v) - np.vdot(psi, v) ** 2).real, 0.0))
    return sig, len(groups)


def stats(x):
    x = np.asarray(x, float)
    return {'min': float(x.min()), 'median': float(np.median(x)), 'max': float(x.max())}


out = {}
for name in ['CH2', 'NH', 'NH_stretch', 'O2', 'TMM']:
    mol, mc, F, S, ncas, nelecas, h1, ecore, eri = setup(name)
    nq = 2 * ncas
    psi = pyscf_to_fock(mc.ci, ncas, nelecas)
    _, d2 = fci.direct_spin1.make_rdm12s(mc.ci, ncas, nelecas)
    Dt = d_tensor(F, d2, S)
    w, V = np.linalg.eigh(Dt); V = V[:, np.argsort(np.abs(w))]
    Dval = abs(1.5 * (V.T @ Dt @ V)[2, 2] * HARTREE2CM)
    pref = d_prefactor(S) * HARTREE2CM
    Fp = np.einsum('ai,bj,ab...->ij...', V, V, F)
    qH = of.jordan_wigner(hamiltonian_op(h1, eri, ecore, ncas)); qH.compress(1e-14); qH.terms.pop((), None)
    qZ = of.jordan_wigner(d_component_op(Fp[2, 2], ncas, S)); qZ.compress(1e-14); qZ.terms.pop((), None)
    comps = [(0, 0), (1, 1), (0, 1), (0, 2), (1, 2)]
    qC = []
    for a, b in comps:
        q = of.jordan_wigner(d_component_op(F[a, b], ncas, S)); q.compress(1e-14); q.terms.pop((), None); qC.append(q)
    terms = set(qH.terms) | set(qZ.terms) | set().union(*[set(q.terms) for q in qC])
    Pv = pauli_vectors(terms, psi, nq)
    eps_H, eps_Z, eps_C = 1.6e-3, 0.01 * Dval / (1.5 * pref), 0.01 * Dval / pref
    r = {'H': [], 'Dz': [], 'full': [], 'gH': [], 'gD': [], 'gfull': []}
    for seed in range(NSEED):
        s, g = grouped_sigmas([qH], Pv, psi, seed); r['H'].append((s[:, 0].sum() / eps_H) ** 2); r['gH'].append(g)
        s, g = grouped_sigmas([qZ], Pv, psi, seed); r['Dz'].append((s[:, 0].sum() / eps_Z) ** 2); r['gD'].append(g)
        s, g = grouped_sigmas(qC, Pv, psi, seed)
        a = s.max(axis=1); a = a / a.sum(); m = a > 0
        r['full'].append(max(((s[m, c] ** 2) / a[m]).sum() / eps_C ** 2 for c in range(len(qC)))); r['gfull'].append(g)
    ratio = np.array(r['Dz']) / np.array(r['H'])
    out[name] = {'D_cm': Dval, 'nterms_H': len(qH.terms), 'nterms_D': len(qZ.terms),
                 'nterms_full': len(set().union(*[set(q.terms) for q in qC])),
                 **{k: stats(v) for k, v in r.items()}, 'ratio_D_over_H': stats(ratio),
                 'ratio_full_over_Dz': stats(np.array(r['full']) / np.array(r['Dz']))}
    o = out[name]
    print(f"{name:10s} H {o['H']['median']:.2e} [{o['H']['min']:.2e},{o['H']['max']:.2e}] groups {o['gH']['median']:.0f}"
          f" | Dz {o['Dz']['median']:.2e} [{o['Dz']['min']:.2e},{o['Dz']['max']:.2e}] groups {o['gD']['median']:.0f}"
          f" | D/H {o['ratio_D_over_H']['median']:.2f} | full {o['full']['median']:.2e} groups {o['gfull']['median']:.0f}"
          f" full/Dz {o['ratio_full_over_Dz']['median']:.2f} [{o['ratio_full_over_Dz']['min']:.2f},{o['ratio_full_over_Dz']['max']:.2f}]",
          flush=True)
json.dump(out, open('results/shots_seeds.json', 'w'), indent=1)
