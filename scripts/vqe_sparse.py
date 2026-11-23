"""Sparse ADAPT-VQE in the determinant basis for larger active spaces (tested to 20 qubits).

Differences from vqe.py (dense):
  * H is applied on the fly with PySCF contract_2e; no Hamiltonian matrix is stored.
  * Pool operators are stored in factorized form, A = sum_c coef_c (K_c - K_c^T), K_c = Ma_c (x) Mb_c,
    with Ma, Mb single-spin string maps; pool gradients are evaluated without building full matrices.
  * QEB (qubit-excitation) operators: K_qeb = P |K| P with P the Jordan-Wigner parity of each determinant.
  * Chosen operators are exponentiated exactly by diagonalizing the connected blocks of A.
Pool definitions, ordering and normalization are identical to vqe.py, so both codes pick the same operators.
"""
import itertools
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.optimize import minimize
from pyscf import fci
from pyscf.fci import cistring


class SparseSpace:
    def __init__(self, norb, nelec):
        self.norb, self.nelec = norb, nelec
        na, nb = nelec
        self.sa = [int(s) for s in cistring.make_strings(range(norb), na)]
        self.sb = [int(s) for s in cistring.make_strings(range(norb), nb)]
        self.ia = {s: i for i, s in enumerate(self.sa)}
        self.ib = {s: i for i, s in enumerate(self.sb)}
        self.na, self.nb = len(self.sa), len(self.sb)
        self.dim = self.na * self.nb
        # JW (interleaved) parity: (-1)^{#(p in alpha, q in beta, q < p)}
        cnt = np.zeros((self.na, self.nb), dtype=np.int64)
        for i, a in enumerate(self.sa):
            occ = [p for p in range(norb) if (a >> p) & 1]
            for j, b in enumerate(self.sb):
                cnt[i, j] = sum(bin(b & ((1 << p) - 1)).count('1') for p in occ)
        self.P = np.where(cnt % 2 == 0, 1.0, -1.0)
        self._cache = {}

    @staticmethod
    def _apply(string, ops):
        sign = 1
        for orb, dag in reversed(ops):
            occ = (string >> orb) & 1
            if dag:
                if occ:
                    return None, 0
                sign *= cistring.cre_sign(orb, string)
                string |= (1 << orb)
            else:
                if not occ:
                    return None, 0
                sign *= cistring.des_sign(orb, string)
                string ^= (1 << orb)
        return string, sign

    def spin_mat(self, ops, spin):
        key = (tuple(ops), spin)
        if key in self._cache:
            return self._cache[key]
        strs, idx = (self.sa, self.ia) if spin == 0 else (self.sb, self.ib)
        rows, cols, vals = [], [], []
        for j, s in enumerate(strs):
            t, sg = self._apply(s, ops)
            if t is None:
                continue
            rows.append(idx[t]); cols.append(j); vals.append(float(sg))
        n = len(strs)
        M = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        self._cache[key] = M
        return M


class Op:
    """A = sum_c coef_c (K_c - K_c^T), K_c = Ma (x) Mb (None = identity); qubit=True -> QEB form."""

    def __init__(self, comps, label, qubit=False):
        self.comps, self.label, self.qubit = comps, label, qubit

    def is_zero(self):
        for c, Ma, Mb in self.comps:
            if (Ma is None or Ma.nnz) and (Mb is None or Mb.nnz):
                return False
        return True


def _abs(M):
    if M is None:
        return None
    A = M.copy(); A.data = np.abs(A.data); return A


def _bilinear(X, Ma, Mb, Y):
    """<X| Ma (x) Mb |Y> for X, Y as (na, nb) arrays."""
    Z = Y if Ma is None else Ma @ Y
    Z = Z if Mb is None else (Mb @ Z.T).T
    return float(np.sum(X * Z))


def op_gradient(space, op, sig, psi):
    """<sig| A |psi>."""
    if op.qubit:
        sig = space.P * sig; psi = space.P * psi
    g = 0.0
    for c, Ma, Mb in op.comps:
        if op.qubit:
            Ma, Mb = _abs(Ma), _abs(Mb)
        g += c * (_bilinear(sig, Ma, Mb, psi) - _bilinear(psi, Ma, Mb, sig))
    return g


def op_matrix(space, op):
    na, nb = space.na, space.nb
    A = sp.csr_matrix((space.dim, space.dim))
    for c, Ma, Mb in op.comps:
        if op.qubit:
            Ma, Mb = _abs(Ma), _abs(Mb)
        K = sp.kron(Ma if Ma is not None else sp.identity(na), Mb if Mb is not None else sp.identity(nb),
                    format='csr')
        if op.qubit:
            Pd = sp.diags(space.P.ravel())
            K = Pd @ K @ Pd
        A = A + c * (K - K.T)
    A = A.tocsr(); A.eliminate_zeros()
    return A


# ------------------------------------------------------------------ pools (same order as vqe.py)
def pool_gsd(space, qubit=False):
    n = space.norb
    ops = []
    for spin in (0, 1):
        for p, q in itertools.permutations(range(n), 2):
            if p > q:
                M = space.spin_mat([(p, 1), (q, 0)], spin)
                ops.append(Op([(1.0, M, None) if spin == 0 else (1.0, None, M)], ('s', spin, p, q), qubit))
    pairs = list(itertools.combinations(range(n), 2))
    for spin in (0, 1):
        for (p, r), (q, s) in itertools.product(pairs, pairs):
            if (p, r) > (q, s):
                M = space.spin_mat([(p, 1), (r, 1), (s, 0), (q, 0)], spin)
                ops.append(Op([(1.0, M, None) if spin == 0 else (1.0, None, M)], ('dss', spin, p, r, q, s), qubit))
    ab = list(itertools.product(range(n), range(n)))
    for (p, r), (q, s) in itertools.product(ab, ab):
        if (p, r) > (q, s):
            ops.append(Op([(1.0, space.spin_mat([(p, 1), (q, 0)], 0), space.spin_mat([(r, 1), (s, 0)], 1))],
                          ('dab', p, r, q, s), qubit))
    return [o for o in ops if not o.is_zero()]


def pool_sf(space):
    n = space.norb
    ops = []
    for p, q in itertools.permutations(range(n), 2):
        if p > q:
            ops.append(Op([(1.0, space.spin_mat([(p, 1), (q, 0)], 0), None),
                           (1.0, None, space.spin_mat([(p, 1), (q, 0)], 1))], ('E', p, q)))
    seen = set()
    for p, q, r, s in itertools.product(range(n), repeat=4):
        fwd = {(p, q, r, s), (r, s, p, q)}
        bwd = {(q, p, s, r), (s, r, q, p)}
        if fwd & bwd:                       # A = -A  ->  zero generator
            continue
        key = min(fwd | bwd)
        if key in seen:
            continue
        seen.add(key)
        comps = []
        if p != r and q != s:
            comps.append((1.0, space.spin_mat([(p, 1), (r, 1), (s, 0), (q, 0)], 0), None))
            comps.append((1.0, None, space.spin_mat([(p, 1), (r, 1), (s, 0), (q, 0)], 1)))
        comps.append((1.0, space.spin_mat([(p, 1), (q, 0)], 0), space.spin_mat([(r, 1), (s, 0)], 1)))
        comps.append((1.0, space.spin_mat([(r, 1), (s, 0)], 0), space.spin_mat([(p, 1), (q, 0)], 1)))
        ops.append(Op(comps, ('e', p, q, r, s)))
    return [o for o in ops if not o.is_zero()]


# ------------------------------------------------------------------ exact exponentials by blocks
class BlockGenerator:
    def __init__(self, A):
        self.A = A
        ncomp, lab = connected_components(abs(A) + abs(A.T), directed=False)
        sizes = np.bincount(lab, minlength=ncomp)
        self.groups = []
        order = np.argsort(lab, kind='stable')
        starts = np.concatenate([[0], np.cumsum(sizes)])
        bysize = {}
        for k in range(ncomp):
            if sizes[k] > 1:
                bysize.setdefault(int(sizes[k]), []).append(order[starts[k]:starts[k + 1]])
        Ad = A.tolil() if A.shape[0] < 0 else A
        for m, idx_list in bysize.items():
            idx = np.array(idx_list)                       # (nblocks, m)
            blocks = np.stack([A[i][:, i].toarray() for i in idx]) if m > 1 else None
            w, V = np.linalg.eigh(1j * blocks)
            self.groups.append((idx, w, V))

    def apply(self, theta, x):
        y = x.copy()
        for idx, w, V in self.groups:
            xs = x[idx]                                    # (nb, m)
            t = np.einsum('bji,bj->bi', V.conj(), xs)
            t *= np.exp(-1j * theta * w)
            y[idx] = np.einsum('bij,bj->bi', V, t).real
        return y

    def dA(self, x):
        return self.A @ x


# ------------------------------------------------------------------ ADAPT-VQE
class Hamiltonian:
    def __init__(self, h1, eri, ecore, norb, nelec):
        self.h2 = fci.direct_spin1.absorb_h1e(h1, eri, norb, nelec, 0.5)
        self.ecore, self.norb, self.nelec = ecore, norb, nelec

    def __call__(self, x, shape):
        return fci.direct_spin1.contract_2e(self.h2, x.reshape(shape), self.norb, self.nelec).ravel() \
            + self.ecore * x


def _state(gens, thetas, ref):
    psi = ref.copy()
    for g, t in zip(gens, thetas):
        psi = g.apply(t, psi)
    return psi


def _energy_grad(thetas, gens, ref, H, shape):
    psi = _state(gens, thetas, ref)
    lam = H(psi, shape)
    E = psi @ lam
    grad = np.zeros(len(gens))
    phi = psi
    for k in range(len(gens) - 1, -1, -1):
        grad[k] = 2 * lam @ gens[k].dA(phi)
        phi = gens[k].apply(-thetas[k], phi)
        lam = gens[k].apply(-thetas[k], lam)
    return E, grad


def adapt_vqe(space, H, ref, pool, observe=None, max_ops=200, gtol=1e-5, etol=1e-10, e_exact=None,
              time_budget=None, log=None, select=None, extra_after=None, chem=1.6e-3):
    """select: optional callable(grads) -> index of the operator to add (default argmax |grads|, standard ADAPT);
    the convergence test always uses the exact gradients. extra_after: if set (with e_exact), stop this many
    operators after the energy error first falls below chem."""
    shape = (space.na, space.nb)
    gens, thetas, chosen, history = [], np.zeros(0), [], []
    psi = ref.copy()
    t0 = time.time()
    rec = {'n': 0, 'E': psi @ H(psi, shape), 't': 0.0}
    if observe:
        rec.update(observe(psi))
    history.append(rec)
    for it in range(max_ops):
        sig = H(psi, shape).reshape(shape); ps = psi.reshape(shape)
        grads = np.array([2 * op_gradient(space, o, sig, ps) for o in pool])
        gmax = float(np.abs(grads).max())
        k = int(np.argmax(np.abs(grads))) if select is None else int(select(grads))
        if gmax < gtol or (e_exact is not None and history[-1]['E'] - e_exact < etol):
            break
        if extra_after is not None and e_exact is not None:
            ic = next((r['n'] for r in history if r['E'] - e_exact < chem), None)
            if ic is not None and history[-1]['n'] >= ic + extra_after:
                break
        if time_budget is not None and time.time() - t0 > time_budget:
            break
        gens.append(BlockGenerator(op_matrix(space, pool[k]))); chosen.append(pool[k].label)
        res = minimize(_energy_grad, np.append(thetas, 0.0), args=(gens, ref, H, shape), jac=True,
                       method='BFGS', options={'gtol': 1e-9, 'maxiter': 4000})
        thetas = res.x
        psi = _state(gens, thetas, ref)
        rec = {'n': len(gens), 'E': float(res.fun), 'gmax': float(gmax), 't': time.time() - t0}
        if observe:
            rec.update(observe(psi))
        history.append(rec)
        if log:
            log(rec)
    return history, chosen, thetas, psi
