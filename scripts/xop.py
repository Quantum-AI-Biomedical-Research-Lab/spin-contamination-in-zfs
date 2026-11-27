"""Apply the spin-spin operator X_ab (D_ab = d_prefactor(S) <X_ab>) to a CI vector without building it.

Using M = s_z s_z - s.s/3 = (2/3) s_z (x) s_z - (1/6)(s+ (x) s- + s- (x) s+) and the fact that the
one-body contraction sum_x M[s,x,x,s'] vanishes,
    X = 1/2 sum_pqrs F_pqrs [ (1/6) Z_pq Z_rs - (1/6) (T+_pq T-_rs + T-_pq T+_rs) ],
with Z_pq = E^a_pq - E^b_pq, T+_pq = a+_{p alpha} a_{q beta}, T-_pq = a+_{p beta} a_{q alpha}.
Determinants are ordered |alpha string>|beta string> (PySCF), so beta operators pass N_alpha alpha creators.
"""
import numpy as np
import scipy.sparse as sp
from pyscf.fci import cistring


def _strings(norb, n):
    s = [int(x) for x in cistring.make_strings(range(norb), n)]
    return s, {x: i for i, x in enumerate(s)}


def _one(norb, n_from, n_to, ops):
    """Rectangular map between string spaces for a product of single-spin operators (rightmost first)."""
    sf, _ = _strings(norb, n_from); st, it = _strings(norb, n_to)
    rows, cols, vals = [], [], []
    for j, s in enumerate(sf):
        sign, t = 1, s
        ok = True
        for orb, dag in reversed(ops):
            occ = (t >> orb) & 1
            if dag:
                if occ:
                    ok = False; break
                sign *= cistring.cre_sign(orb, t); t |= (1 << orb)
            else:
                if not occ:
                    ok = False; break
                sign *= cistring.des_sign(orb, t); t ^= (1 << orb)
        if ok:
            rows.append(it[t]); cols.append(j); vals.append(float(sign))
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(st), len(sf)))


class XOperator:
    def __init__(self, norb, nelec, F):
        """F: (norb,)*4 spin-dipole integrals of one tensor component (e.g. principal zz)."""
        self.n, (self.na, self.nb) = norb, nelec
        n, na, nb = norb, self.na, self.nb
        self.F = F.reshape(n * n, n * n)
        self.dims = {k: (cistring.num_strings(n, k[0]), cistring.num_strings(n, k[1]))
                     for k in [(na, nb), (na - 1, nb + 1), (na + 1, nb - 1)]}
        pq = [(p, q) for p in range(n) for q in range(n)]
        self.Ea = [_one(n, na, na, [(p, 1), (q, 0)]) for p, q in pq]
        self.Eb = [_one(n, nb, nb, [(p, 1), (q, 0)]) for p, q in pq]
        # T-_pq = a+_{p b} a_{q a}: (na,nb) -> (na-1,nb+1); beta creator passes na-1 alpha electrons
        self.Tm_a = [_one(n, na, na - 1, [(q, 0)]) for p, q in pq]
        self.Tm_b = [_one(n, nb, nb + 1, [(p, 1)]) * (-1) ** (na - 1) for p, q in pq]
        # T+_pq = a+_{p a} a_{q b}: (na-1,nb+1) -> (na,nb); beta annihilator passes na-1 alpha electrons
        self.Tp_a_back = [_one(n, na - 1, na, [(p, 1)]) for p, q in pq]
        self.Tp_b_back = [_one(n, nb + 1, nb, [(q, 0)]) * (-1) ** (na - 1) for p, q in pq]
        # T+_pq on (na,nb) -> (na+1,nb-1); beta annihilator passes na alpha electrons
        self.Tp_a = [_one(n, na, na + 1, [(p, 1)]) for p, q in pq]
        self.Tp_b = [_one(n, nb, nb - 1, [(q, 0)]) * (-1) ** na for p, q in pq]
        # T-_pq on (na+1,nb-1) -> (na,nb); beta creator passes na alpha electrons
        self.Tm_a_back = [_one(n, na + 1, na, [(q, 0)]) for p, q in pq]
        self.Tm_b_back = [_one(n, nb - 1, nb, [(p, 1)]) * (-1) ** na for p, q in pq]

    @staticmethod
    def _ap(Ma, Mb, C):
        return (Ma @ C) @ Mb.T if Mb is not None else Ma @ C

    def __call__(self, x):
        n2 = self.n * self.n
        C = x.reshape(self.dims[(self.na, self.nb)])
        out = np.zeros_like(C)
        # Z Z part
        V = np.array([(self.Ea[k] @ C) - (C @ self.Eb[k].T) for k in range(n2)])        # Z_rs psi
        W = np.tensordot(self.F, V, axes=(1, 0))                                         # sum_rs F_pqrs Z_rs psi
        for k in range(n2):
            out += (1.0 / 12.0) * ((self.Ea[k] @ W[k]) - (W[k] @ self.Eb[k].T))
        # T+_pq T-_rs: through (na-1, nb+1)
        U = np.array([(self.Tm_a[k] @ C) @ self.Tm_b[k].T for k in range(n2)])
        Y = np.tensordot(self.F, U, axes=(1, 0))
        for k in range(n2):
            out -= (1.0 / 12.0) * ((self.Tp_a_back[k] @ Y[k]) @ self.Tp_b_back[k].T)
        # T-_pq T+_rs: through (na+1, nb-1)
        if self.nb >= 1:
            U = np.array([(self.Tp_a[k] @ C) @ self.Tp_b[k].T for k in range(n2)])
            Y = np.tensordot(self.F, U, axes=(1, 0))
            for k in range(n2):
                out -= (1.0 / 12.0) * ((self.Tm_a_back[k] @ Y[k]) @ self.Tm_b_back[k].T)
        return out.ravel()


if __name__ == '__main__':
    # validation against the explicit OpenFermion operator (12 qubits)
    import warnings
    warnings.filterwarnings('ignore')
    import openfermion as of
    from theory_check import setup
    from fockmap import pyscf_to_fock, d_component_op
    from vqe import CISpace
    for name in ['CH2', 'O2']:
        mol, mc, F, S, ncas, nelecas, *_ = setup(name)
        sp_ = CISpace(ncas, nelecas); na, nb = len(sp_.sa), len(sp_.sb)
        X = XOperator(ncas, nelecas, F[0, 1] + 0.3 * F[2, 2])
        Xf = of.get_sparse_operator(d_component_op(F[0, 1] + 0.3 * F[2, 2], ncas, S), n_qubits=2 * ncas)
        A = np.array([pyscf_to_fock(np.eye(sp_.dim)[k].reshape(na, nb), ncas, nelecas) for k in range(sp_.dim)]).T
        Xd = (A.T @ (Xf @ A)).real
        rng = np.random.default_rng(1)
        err = 0
        for _ in range(3):
            v = rng.standard_normal(sp_.dim)
            err = max(err, np.abs(X(v) - Xd @ v).max() / np.abs(Xd @ v).max())
        print(f'{name}: max relative |X_ci v - X_openfermion v| = {err:.1e}')
