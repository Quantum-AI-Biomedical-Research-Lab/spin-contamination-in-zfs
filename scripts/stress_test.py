"""Stress tests of the theory in the manuscript.

T1  Full first-order ZFS Hamiltonian P H_SS P in the M = 1, 0, -1 triplet manifold, built from
    explicit spin matrices (no Wigner-Eckart step), versus S.D.S from the M_S = S formula.
T3  Selection rules and rank-k scaling under controlled contamination:
      rank 0  spin-free one-body operator     -> error ~ (<S^2>-2)^1    (no cross term)
      rank 1  spin-density operator           -> error ~ (<S^2>-2)^1/2, couples to S+1 only
      rank 2  spin-spin D operator            -> error ~ (<S^2>-2)^1/2, couples to S+1 and S+2
"""
import sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import openfermion as of
from pyscf import fci
from zfs import d_tensor, HARTREE2CM, ALPHA, G_E
from fockmap import pyscf_to_fock, d_component_op, d_prefactor
from theory_check import setup
from vqe import CISpace

S_MAT = [np.array([[0, .5], [.5, 0]], complex), np.array([[0, -.5j], [.5j, 0]], complex),
         np.array([[.5, 0], [0, -.5]], complex)]


def hss_op(F, norb):
    """Full H_SS = (g^2 a^2/4) sum_{i<j} sum_ab s_a(i) s_b(j) f_ab(ij) as a fermion operator."""
    pref = G_E ** 2 * ALPHA ** 2 / 4.0
    H = of.FermionOperator()
    for a in range(3):
        for b in range(3):
            M = np.einsum('ij,kl->ijkl', S_MAT[a], S_MAT[b])
            for p in range(norb):
                for q in range(norb):
                    for r in range(norb):
                        for t in range(norb):
                            v = F[a, b, p, q, r, t]
                            if abs(v) < 1e-13:
                                continue
                            for s1 in range(2):
                                for s1p in range(2):
                                    for s2 in range(2):
                                        for s2p in range(2):
                                            m = M[s1, s1p, s2, s2p]
                                            if m == 0:
                                                continue
                                            i, j, k, l = 2 * p + s1, 2 * r + s2, 2 * t + s2p, 2 * q + s1p
                                            if i == j or k == l:
                                                continue
                                            H += of.FermionOperator(((i, 1), (j, 1), (k, 0), (l, 0)),
                                                                    0.5 * v * m * 1e6)  # scaled; undone below
    return H, pref * 1e-6


def t1(name):
    mol, mc, F, S, ncas, nelecas, *_ = setup(name)
    nq = 2 * ncas
    _, d2 = fci.direct_spin1.make_rdm12s(mc.ci, ncas, nelecas)
    D = d_tensor(F, d2, S)                       # Hartree, from the M_S = S formula
    psi1 = pyscf_to_fock(mc.ci, ncas, nelecas)
    Sm = of.get_sparse_operator(of.hermitian_conjugated(of.s_plus_operator(ncas)), n_qubits=nq)
    psi0 = Sm @ psi1; psi0 /= np.linalg.norm(psi0)
    psim = Sm @ psi0; psim /= np.linalg.norm(psim)
    basis = np.array([psi1, psi0, psim]).T
    Hop, scale = hss_op(F, ncas)
    Hs = of.get_sparse_operator(Hop, n_qubits=nq) * scale
    Hm = basis.conj().T @ (Hs @ basis)
    # spin-1 matrices in the basis M = +1, 0, -1 (Condon-Shortley; S- psi gives the standard phase)
    r2 = np.sqrt(2)
    Sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / r2
    Sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / r2
    Sz = np.diag([1., 0, -1])
    Sv = [Sx, Sy, Sz]
    Heff = sum(D[a, b] * Sv[a] @ Sv[b] for a in range(3) for b in range(3))
    Hm_tl = Hm - np.trace(Hm) / 3 * np.eye(3)
    err = np.abs(Hm_tl - Heff).max() / np.abs(Heff).max()
    ev_full = np.linalg.eigvalsh(Hm_tl) * HARTREE2CM
    ev_eff = np.linalg.eigvalsh(Heff) * HARTREE2CM
    print(f"T1 {name:10s} max|<M|H_SS|M'> - <M|S.D.S|M'>| / max = {err:.2e}; "
          f"sublevels (cm-1) full {np.round(ev_full, 5)} vs S.D.S {np.round(ev_eff, 5)}; "
          f"trace part {np.trace(Hm).real/3*HARTREE2CM:+.2e} cm-1")


def t3(name):
    mol, mc, F, S, ncas, nelecas, *_ = setup(name)
    space = CISpace(ncas, nelecas); na, nb = len(space.sa), len(space.sb); dim = space.dim
    S2m = np.zeros((dim, dim))
    for i in range(dim):
        e = np.zeros(dim); e[i] = 1
        S2m[:, i] = fci.spin_op.contract_ss(e.reshape(na, nb), ncas, nelecas).ravel()
    ws, Vs = np.linalg.eigh(0.5 * (S2m + S2m.T))
    sec = {s: Vs[:, np.abs(ws - s * (s + 1)) < 1e-6] for s in (1, 2, 3)}
    T = mc.ci.ravel()
    # operators in the determinant basis
    rng = np.random.default_rng(7)
    h = rng.standard_normal((ncas, ncas)); h = h + h.T
    O0 = sum(h[p, q] * space.E(p, q) for p in range(ncas) for q in range(ncas)).toarray()
    O1 = sum(h[p, q] * (space.single(p, q, 0) - space.single(p, q, 1)) for p in range(ncas)
             for q in range(ncas)).toarray()
    _, d2 = fci.direct_spin1.make_rdm12s(mc.ci, ncas, nelecas)
    w, V = np.linalg.eigh(d_tensor(F, d2, S)); V = V[:, np.argsort(np.abs(w))]
    Fp = np.einsum('ai,bj,ab...->ij...', V, V, F)
    Xf = of.get_sparse_operator(d_component_op(Fp[2, 2], ncas, S), n_qubits=2 * ncas)
    A = np.array([pyscf_to_fock(np.eye(dim)[k].reshape(na, nb), ncas, nelecas) for k in range(dim)]).T
    O2 = (A.T @ (Xf @ A)).real
    print(f"T3 {name}: sector dims S=1:{sec[1].shape[1]} S=2:{sec[2].shape[1]} S=3:{sec[3].shape[1]}")
    for lab, O in (('rank0 spin-free', O0), ('rank1 spin density', O1), ('rank2 D_zz', O2)):
        O = 0.5 * (O + O.T)
        c2 = np.linalg.norm(sec[2].T @ (O @ T)); c3 = np.linalg.norm(sec[3].T @ (O @ T))
        # controlled contamination along the most strongly coupling S=2 (or S=3) direction;
        # for rank 0 the coupling vanishes, so use a random S=2 direction instead
        for s, c in ((2, c2), (3, c3)):
            v = sec[s].T @ (O @ T)
            q = sec[s] @ (v / np.linalg.norm(v)) if np.linalg.norm(v) > 1e-10 else \
                sec[s] @ rng.standard_normal(sec[s].shape[1])
            q /= np.linalg.norm(q)
            gap = s * (s + 1) - 2
            ds, errs = [], []
            for eps in np.logspace(-4, -2, 5):
                ph = np.sqrt(1 - eps ** 2) * T + eps * q
                ds.append(gap * eps ** 2); errs.append(abs(ph @ O @ ph - T @ O @ T))
            slope = np.polyfit(np.log10(ds), np.log10(np.maximum(errs, 1e-300)), 1)[0]
            print(f"   {lab:19s} ||P_S={s} O T|| = {c:9.2e}   error vs <S2>-2 slope (S={s} contaminant) = {slope:.3f}")


if __name__ == '__main__':
    which = sys.argv[1]
    for n in sys.argv[2:]:
        (t1 if which == 't1' else t3)(n)
