"""Map PySCF CI vectors to OpenFermion (Jordan-Wigner, interleaved spin-orbitals) state vectors,
and build second-quantised operators (Hamiltonian, D-tensor components, S^2)."""
import numpy as np
from pyscf.fci import cistring
import openfermion as of
from zfs import ALPHA, G_E


def _parity(seq):
    seq = list(seq); s = 1
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] > seq[j]:
                s = -s
    return s


def pyscf_to_fock(ci, norb, nelec):
    na, nb = nelec
    sa = cistring.make_strings(range(norb), na)
    sb = cistring.make_strings(range(norb), nb)
    nq = 2 * norb
    psi = np.zeros(2 ** nq)
    for ia, a in enumerate(sa):
        oa = [p for p in range(norb) if (a >> p) & 1]
        for ib, b in enumerate(sb):
            c = ci[ia, ib]
            if c == 0:
                continue
            ob = [p for p in range(norb) if (b >> p) & 1]
            so = [2 * p for p in oa] + [2 * p + 1 for p in ob]
            sign = _parity(so)
            idx = 0
            for q in so:
                idx |= 1 << (nq - 1 - q)   # OpenFermion: qubit 0 is most significant
            psi[idx] = sign * c
    return psi


def hamiltonian_op(h1, eri, ecore, norb):
    H = of.FermionOperator((), ecore)
    for p in range(norb):
        for q in range(norb):
            if abs(h1[p, q]) > 1e-14:
                for s in range(2):
                    H += of.FermionOperator(((2 * p + s, 1), (2 * q + s, 0)), h1[p, q])
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for t in range(norb):
                    v = eri[p, q, r, t]
                    if abs(v) < 1e-14:
                        continue
                    for s1 in range(2):
                        for s2 in range(2):
                            if 2 * p + s1 == 2 * r + s2 or 2 * q + s1 == 2 * t + s2:
                                continue
                            H += of.FermionOperator(((2 * p + s1, 1), (2 * r + s2, 1),
                                                     (2 * t + s2, 0), (2 * q + s1, 0)), 0.5 * v)
    return H


_sz = np.array([[0.5, 0], [0, -0.5]], complex)
_sx = np.array([[0, 0.5], [0.5, 0]], complex)
_sy = np.array([[0, -0.5j], [0.5j, 0]], complex)
# M[s1,s1',s2,s2'] = sz sz - (s.s)/3
_M = (np.einsum('ab,cd->abcd', _sz, _sz)
      - (np.einsum('ab,cd->abcd', _sx, _sx) + np.einsum('ab,cd->abcd', _sy, _sy)
         + np.einsum('ab,cd->abcd', _sz, _sz)) / 3.0).real


def d_prefactor(S, g=G_E):
    return (g * g * ALPHA ** 2 / 4.0) * 3.0 / (S * (2 * S - 1))


def d_component_op(Fab, norb, S, g=G_E):
    """Unscaled fermion operator X_ab; D_ab = d_prefactor(S) * <X_ab>.
    (The prefactor is applied afterwards because OpenFermion discards
    coefficients below 1e-8.)"""
    pref = 1.0
    O = of.FermionOperator()
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for t in range(norb):
                    v = Fab[p, q, r, t]
                    if abs(v) < 1e-14:
                        continue
                    for s1 in range(2):
                        for s1p in range(2):
                            for s2 in range(2):
                                for s2p in range(2):
                                    m = _M[s1, s1p, s2, s2p]
                                    if m == 0:
                                        continue
                                    i, j = 2 * p + s1, 2 * r + s2
                                    k, l = 2 * t + s2p, 2 * q + s1p
                                    if i == j or k == l:
                                        continue
                                    O += of.FermionOperator(((i, 1), (j, 1), (k, 0), (l, 0)),
                                                            0.5 * pref * v * m)
    return O
