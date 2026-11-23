"""Spin-spin zero-field splitting from (spin-resolved) reduced density matrices.

Conventions
-----------
H_SS = (g^2 alpha^2 / 4) sum_{i<j} sum_ab s_a(i) s_b(j) f_ab(r_ij),
f_ab(r) = (r^2 delta_ab - 3 r_a r_b) / r^5          (atomic units)

Projected onto the |S, M_S=S> component (Wigner-Eckart), H_eff = S.D.S with

D_ab = (g^2 alpha^2/4) * 3/(S(2S-1)) * < sum_{i<j} f_ab(ij) [ s_z(i)s_z(j) - s(i).s(j)/3 ] >

The s(i).s(j) part is evaluated with the Dirac identity
s(i).s(j) = -1/2 P^r_ij - 1/4 (valid on antisymmetric states),
so only PySCF's spin-resolved 2-RDMs (aa, ab, bb) are needed.

PySCF 2-RDM convention: dm2[p,q,r,s] = < p^+ r^+ s q >, E2 = 1/2 sum (pq|rs) dm2[pqrs].
"""
import numpy as np
from pyscf import ao2mo
from pyscf.data import nist

ALPHA = nist.ALPHA
G_E = 2.00231930436
HARTREE2CM = 219474.6313705


def ss_mo_ints(mol, C):
    """Traceless spin-dipole integrals F[a,b,p,q,r,s] = (pq| f_ab |rs) in the MO basis C.

    f_ab is obtained as the traceless part of the symmetrised (d_a rho_1 | d_b rho_2)
    Coulomb-derivative integrals (int2e_ip1ip2); the contact term only affects the trace.
    """
    n = C.shape[1]
    eri = ao2mo.general(mol, (C, C, C, C), intor='int2e_ip1ip2', comp=9,
                        aosym='s1', compact=False)
    h = np.asarray(eri).reshape(3, 3, n, n, n, n)
    h = h + h.transpose(0, 1, 3, 2, 4, 5)
    h = h + h.transpose(0, 1, 2, 3, 5, 4)
    tr = (h[0, 0] + h[1, 1] + h[2, 2]) / 3.0
    for x in range(3):
        h[x, x] -= tr
    # symmetrise in a<->b (exact up to numerical noise)
    h = 0.5 * (h + h.transpose(1, 0, 2, 3, 4, 5))
    return h


def spin_bracket(F, dm2s):
    """X_ab = < sum_{i<j} f_ab(ij) [s_z s_z - s.s/3] > from spin-resolved 2-RDMs."""
    dm2aa, dm2ab, dm2bb = dm2s
    dm2ba = dm2ab.transpose(2, 3, 0, 1)
    gz = dm2aa + dm2bb - dm2ab - dm2ba          # same-spin minus opposite-spin pairs
    dm2 = dm2aa + dm2bb + dm2ab + dm2ba          # spin-summed
    # sum_{i<j} f s_z s_z = 1/2 * 1/4 * sum F gz
    A = 0.125 * np.einsum('xypqrs,pqrs->xy', F, gz)
    # sum_{i<j} f s.s = -1/4 sum (ps|f|rq) dm2[pqrs] - 1/8 sum (pq|f|rs) dm2[pqrs]
    Bx = np.einsum('xypsrq,pqrs->xy', F, dm2)
    Bc = np.einsum('xypqrs,pqrs->xy', F, dm2)
    B = -0.25 * Bx - 0.125 * Bc
    return A - B / 3.0


def d_tensor(F, dm2s, S, g=G_E):
    """D tensor in Hartree."""
    X = spin_bracket(F, dm2s)
    return (g * g * ALPHA ** 2 / 4.0) * 3.0 / (S * (2 * S - 1)) * X


def d_and_e(D):
    """Return (D, E, eigenvalues, eigenvectors) with |E| <= |D|/3 convention."""
    D = 0.5 * (D + D.T)
    w, v = np.linalg.eigh(D)
    # choose z as the eigenvalue of largest magnitude
    order = np.argsort(np.abs(w))
    dx, dy, dz = w[order[0]], w[order[1]], w[order[2]]
    Dval = 1.5 * dz
    Eval = 0.5 * (dx - dy)
    if Eval * Dval < 0:  # convention: E has the same sign as D
        Eval = -Eval
    return Dval, Eval, w, v
