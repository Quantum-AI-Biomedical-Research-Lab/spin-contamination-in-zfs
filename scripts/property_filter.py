"""Two follow-up tests on the 12-qubit systems (one Hamiltonian and one set of orbitals per system).

(1) Spin density at the nuclei (the Fermi-contact part of the isotropic hyperfine coupling).
    rho_A = <sum_i delta(r_i - R_A) 2 s_z(i)> = sum_pq phi_p(R_A) phi_q(R_A) (gamma^a - gamma^b)_pq over the active
    orbitals (core orbitals are doubly occupied and carry no spin density). rho is the q = 0 component of a rank-1
    spin tensor, so only S+1 contaminants enter at first order. The contamination term is split into the cross
    terms with the S+1 and S+2 parts of |Q> and the second-order remainder; the same split is given for D.

(2) Gaussian spectral filter g(H) = exp[-(H - mu)^2 T^2 / 2] with mu = E(psi), applied to the ADAPT state at
    chemical accuracy. g(H) = E_t[exp(-i (H - mu) t)] with t ~ N(0, T^2), so filtered expectation values
    <psi|g O g|psi> / <psi|g^2|psi> follow from Hadamard tests with controlled real-time evolutions
    (randomized linear combination of unitaries). Simulated exactly here (full diagonalization).

Cost ingredients written per system:
    lam_H     Pauli 1-norm of the JW Hamiltonian without the identity (block-encoding normalization, Eh)
    X groups  QWC groups of the D operator X_zz; on the exact triplet: sum_g sigma_g (direct measurement) and
              sum_g sqrt(<X_g^2>) (interferometric Hadamard-test measurement of a group); ||c_X||_1.
Writes results/propfilter_<system>.json.  Usage (from the repository root): python scripts/property_filter.py [systems...]
"""
import json
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import openfermion as of
from pyscf import fci
import remedies
import penalty
from vqe import pool_gsd, pool_sf
from shiftinvert_test import refine
from fockmap import hamiltonian_op, d_component_op, pyscf_to_fock

VARIANTS = [('GSD', 0.0), ('QEB', 0.0), ('sfGSD', 0.0), ('QEB', 3.0)]
TS = [5, 10, 15, 20, 30, 40, 60, 80]          # filter width parameter T in Eh^-1 (atomic units of time)
CHEM = 1.6e-3
import os
MU_F = float(os.environ.get('MU_F', '0.05'))                                   # spin shift (Eh) of the filtered Hamiltonian H + MU_F (S^2 - S(S+1))


def nuclear_orbital_values(mol, mc, ncas):
    C = mc.mo_coeff[:, mc.ncore:mc.ncore + ncas]
    return mol.eval_gto('GTOval', mol.atom_coords()) @ C          # (natm, ncas)


def main(name):
    t0 = time.time()
    sysd = remedies.build(name)
    space = sysd['space']; H, Xz, cD, E0, Dex = sysd['H'], sysd['Xz'], sysd['cD'], sysd['E0'], sysd['Dex']
    mol, mc, ncas, nelecas, S = sysd['mol'], sysd['mc'], sysd['ncas'], sysd['nelecas'], sysd['S']
    na, nb = len(space.sa), len(space.sb)
    S2 = penalty.spin_matrix(space, ncas, nelecas)
    ws, Vs = np.linalg.eigh(S2)
    P = {k: Vs[:, np.abs(ws - k * (k + 1)) < 1e-6] for k in (1, 2, 3)}
    PT = P[1]
    phi = nuclear_orbital_values(mol, mc, ncas)
    atoms = [mol.atom_symbol(i) for i in range(mol.natm)]

    def rho(v):
        v = v / np.linalg.norm(v)
        ga, gb = fci.direct_spin1.make_rdm1s(v.reshape(na, nb), ncas, nelecas)
        return np.einsum('ap,aq,pq->a', phi, phi, ga - gb)

    def trho(bra, ket):
        ta, tb = fci.direct_spin1.trans_rdm1s(bra.reshape(na, nb), ket.reshape(na, nb), ncas, nelecas)
        return np.einsum('ap,aq,pq->a', phi, phi, ta - tb)

    T = mc.ci.ravel() / np.linalg.norm(mc.ci)
    rho_ex = rho(T)
    Dof = lambda v: float(cD * v @ Xz @ v / (v @ v))
    rel = lambda a, b: 100 * (a - b) / b
    E, V = np.linalg.eigh(H)
    s2E = np.einsum('ij,ik,kj->j', V, S2, V)
    trip = np.abs(s2E - 2) < 1e-4
    gaps = {'E1_triplet': float(E[trip][1] - E[trip][0]),
            'E_lowest_S2': float(E[np.abs(s2E - 6) < 1e-4].min() - E0) if np.any(np.abs(s2E - 6) < 1e-4) else None}

    out = {'D_exact': Dex, 'atoms': atoms, 'rho_exact': rho_ex.tolist(), 'gaps': gaps, 'runs': {}}
    for pname, mu in VARIANTS:
        pool = pool_sf(space) if pname == 'sfGSD' else pool_gsd(space, qubit=(pname == 'QEB'))
        r = penalty.run(sysd, S2, PT, pool, pname, mu, 0.0, keep_states=True)
        ic = next(i for i, q in enumerate(r['rows']) if q['dE'] < CHEM)
        psi = r['states'][ic]; psi = psi / np.linalg.norm(psi)
        t = PT @ (PT.T @ psi); w = t @ t; tt = t / np.sqrt(w)
        Q = psi - t                                  # = sqrt(1-w) |Q>
        Q2 = P[2] @ (P[2].T @ Q); Q3 = P[3] @ (P[3].T @ Q) if P[3].shape[1] else 0 * Q
        rec = {'n_chem': ic, 'S2dev': float(psi @ S2 @ psi - 2), 'w': float(w)}
        # D split
        rec['D'] = {'naive': rel(Dof(psi), Dex), 'proj': rel(Dof(tt), Dex),
                    'cross_S+1': 100 * 2 * cD * float(t @ Xz @ Q2) / Dex,
                    'cross_S+2': 100 * 2 * cD * float(t @ Xz @ Q3) / Dex}
        rec['D']['quad'] = rec['D']['naive'] - rec['D']['proj'] - rec['D']['cross_S+1'] - rec['D']['cross_S+2']
        # spin density split (per atom)
        rn, rp = rho(psi), rho(tt)
        rec['rho'] = {'naive': rel(rn, rho_ex).tolist(), 'proj': rel(rp, rho_ex).tolist(),
                      'cross_S+1': (100 * 2 * trho(t, Q2) / rho_ex).tolist(),
                      'cross_S+2': (100 * 2 * trho(t, Q3) / rho_ex).tolist() if P[3].shape[1] else [0.0] * mol.natm}
        # exact shift-invert step (as in the main text)
        x1 = refine(psi, lambda v: H @ v, space.dim, nsteps=1)[0][0]
        rec['shiftinvert'] = {'D': rel(Dof(x1), Dex), 'rho': rel(rho(x1), rho_ex).tolist()}
        # Gaussian filter, mu = E(psi)
        c = V.T @ psi; Epsi = float(psi @ H @ psi)
        gf = {}
        for TT in TS:
            g = np.exp(-0.5 * ((E - Epsi) * TT) ** 2)
            y = V @ (g * c); Ng = float(y @ y); y /= np.sqrt(Ng)
            gf[TT] = {'D': rel(Dof(y), Dex), 'rho': rel(rho(y), rho_ex).tolist(), 'Ng': Ng,
                      'dE_mEh': 1e3 * float(y @ H @ y - E0), 'S2dev': float(y @ S2 @ y - 2)}
        # rule T = 3 / (E1 - E0), with and without a spin shift of the filtered Hamiltonian
        Trule = 3.0 / gaps['E1_triplet']
        for key, Es, cen in [('rule', E, Epsi), ('rule_spin', E + MU_F * (s2E - 2), Epsi)]:
            g = np.exp(-0.5 * ((Es - cen) * Trule) ** 2)
            y = V @ (g * c); Ng = float(y @ y); y /= np.sqrt(Ng)
            gf[key] = {'T': Trule, 'D': rel(Dof(y), Dex), 'rho': rel(rho(y), rho_ex).tolist(), 'Ng': Ng,
                       'dE_mEh': 1e3 * float(y @ H @ y - E0), 'S2dev': float(y @ S2 @ y - 2)}
        rec['gauss'] = gf
        tag = pname if mu == 0 else f'{pname}_mu{mu:g}'
        out['runs'][tag] = rec
        print(f"{name:10s} {tag:9s} D naive {rec['D']['naive']:+.3f} proj {rec['D']['proj']:+.3f} "
              f"x(S+1) {rec['D']['cross_S+1']:+.3f} x(S+2) {rec['D']['cross_S+2']:+.3f} | rho naive "
              + ' '.join(f"{a}:{v:+.3f}" for a, v in zip(atoms, rec['rho']['naive'])) + ' x(S+2) '
              + ' '.join(f"{v:+.1e}" for v in rec['rho']['cross_S+2']) + ' | gauss D '
              + ' '.join(f"T{TT}:{gf[TT]['D']:+.3f}" for TT in TS) + f" rule(T={gf['rule']['T']:.1f}):{gf['rule']['D']:+.3f} spin:{gf['rule_spin']['D']:+.3f}", flush=True)

    # cost ingredients
    Hq = of.jordan_wigner(hamiltonian_op(sysd['h1'], sysd['eri'], sysd['ecore'], ncas)); Hq.compress(1e-12)
    Hq.terms.pop((), None)
    lamH = float(sum(abs(v) for v in Hq.terms.values()))
    Xq = of.jordan_wigner(d_component_op(sysd['Fp'][2, 2], ncas, S)); Xq.compress(1e-14)
    X0 = Xq.terms.pop((), 0.0)
    nq = 2 * ncas
    psiF = pyscf_to_fock(mc.ci, ncas, nelecas).real; psiF /= np.linalg.norm(psiF)
    Xs = of.get_sparse_operator(Xq + of.QubitOperator((), X0), n_qubits=nq)
    Xmean = float((psiF @ (Xs @ psiF)).real)
    groups = of.group_into_tensor_product_basis_sets(Xq, seed=0)
    sig = sec = 0.0
    for _, gop in groups.items():
        G = of.get_sparse_operator(gop, n_qubits=nq); Gp = (G @ psiF)
        m1 = float((psiF @ Gp).real); m2 = float((np.vdot(Gp, Gp)).real)
        sig += np.sqrt(max(m2 - m1 * m1, 0)); sec += np.sqrt(m2)
    S2q = of.jordan_wigner(of.s_squared_operator(ncas)); S2q.compress(1e-12); S2q.terms.pop((), None)
    lamS2 = float(sum(abs(v) for v in S2q.terms.values()))
    out['cost'] = {'lam_S2': lamS2, 'lam_H': lamH, 'X_mean': Xmean, 'X_identity': float(np.real(X0)), 'X_l1': float(sum(abs(v) for v in Xq.terms.values())),
                   'X_terms': len(Xq.terms), 'X_groups': len(groups), 'sum_sigma_g': sig, 'sum_sqrt_m2_g': sec}
    print(f"{name}: gaps {gaps}  lam_H={lamH:.2f} Eh  X: <X>={Xmean:.4g} l1={out['cost']['X_l1']:.4g} "
          f"sum sigma_g={sig:.4g} sum sqrt<Xg^2>={sec:.4g}  t={time.time()-t0:.0f}s", flush=True)
    out['MU_F'] = MU_F
    json.dump(out, open(f'results/propfilter_{name}' + ('' if MU_F == 0.05 else f'_muf{MU_F:g}') + '.json', 'w'), indent=1)


if __name__ == '__main__':
    for nm in (sys.argv[1:] or ['CH2', 'NH', 'NH_stretch', 'O2']):
        main(nm)
