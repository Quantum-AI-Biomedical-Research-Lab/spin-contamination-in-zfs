"""ADAPT-VQE in the PySCF determinant basis with spin-orbital (GSD), qubit-excitation (QEB)
and spin-free / spin-adapted (sfGSD) operator pools."""
import itertools
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from pyscf import fci
from pyscf.fci import cistring
from fockmap import _parity


# ---------------------------------------------------------------- determinant space
class CISpace:
    def __init__(self, norb, nelec):
        self.norb, self.nelec = norb, nelec
        na, nb = nelec
        self.sa = list(cistring.make_strings(range(norb), na))
        self.sb = list(cistring.make_strings(range(norb), nb))
        self.ia = {s: i for i, s in enumerate(self.sa)}
        self.ib = {s: i for i, s in enumerate(self.sb)}
        self.dim = len(self.sa) * len(self.sb)
        # JW-interleaved parity of every determinant (used for qubit-excitation ops)
        par = np.empty((len(self.sa), len(self.sb)))
        for i, a in enumerate(self.sa):
            oa = [p for p in range(norb) if (a >> p) & 1]
            for j, b in enumerate(self.sb):
                ob = [p for p in range(norb) if (b >> p) & 1]
                par[i, j] = _parity([2 * p for p in oa] + [2 * p + 1 for p in ob])
        self.par = par.ravel()

    # apply a product of single-spin operators (leftmost first) to a string
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

    def _single_spin_matrix(self, ops, spin):
        strs, idx = (self.sa, self.ia) if spin == 0 else (self.sb, self.ib)
        rows, cols, vals = [], [], []
        for j, s in enumerate(strs):
            t, sg = self._apply(s, ops)
            if t is None:
                continue
            rows.append(idx[t]); cols.append(j); vals.append(sg)
        n = len(strs)
        return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    def op(self, ops_a, ops_b):
        """Sparse matrix of (alpha-part)(beta-part); each part must conserve its electron count."""
        na, nb = len(self.sa), len(self.sb)
        Ma = self._single_spin_matrix(ops_a, 0) if ops_a else sp.identity(na, format='csr')
        Mb = self._single_spin_matrix(ops_b, 1) if ops_b else sp.identity(nb, format='csr')
        return sp.kron(Ma, Mb, format='csr')

    # spin-orbital excitation operators
    def single(self, p, q, spin):
        o = [(p, 1), (q, 0)]
        return self.op(o, None) if spin == 0 else self.op(None, o)

    def double_same(self, p, r, s, q, spin):
        o = [(p, 1), (r, 1), (s, 0), (q, 0)]
        return self.op(o, None) if spin == 0 else self.op(None, o)

    def double_ab(self, p, q, r, s):
        """a+_{p a} a+_{r b} a_{s b} a_{q a} = (a+_pa a_qa)(a+_rb a_sb)."""
        return self.op([(p, 1), (q, 0)], [(r, 1), (s, 0)])

    # spin-free operators
    def E(self, p, q):
        return self.single(p, q, 0) + self.single(p, q, 1)

    def e(self, p, q, r, s):
        """e_pqrs = sum_{st} a+_{p s} a+_{r t} a_{s t} a_{q s}."""
        M = self.double_same(p, r, s, q, 0) + self.double_same(p, r, s, q, 1)
        M = M + self.double_ab(p, q, r, s) + self.double_ab(r, s, p, q)
        return M


# ---------------------------------------------------------------- pools
def _dedupe(mats, labels, tol=1e-12):
    keep, seen = [], set()
    for M, lab in zip(mats, labels):
        M = M.tocsr(); M.eliminate_zeros()
        if M.nnz == 0:
            continue
        coo = M.tocoo()
        order = np.lexsort((coo.col, coo.row))
        r, c, v = coo.row[order], coo.col[order], coo.data[order]
        nrm = v[0]
        key = (tuple(r), tuple(c), tuple(np.round(v / nrm, 9)))
        if key in seen:
            continue
        seen.add(key)
        keep.append((M, lab))
    return keep


def pool_gsd(space, qubit=False):
    n = space.norb
    mats, labs = [], []

    def add(T, lab):
        if qubit:  # strip fermionic signs in the JW (interleaved) basis
            T = T.tocoo()
            v = np.abs(T.data) * space.par[T.row] * space.par[T.col]
            T = sp.csr_matrix((v, (T.row, T.col)), shape=T.shape)
        mats.append((T - T.T).tocsr()); labs.append(lab)

    for spin in (0, 1):
        for p, q in itertools.permutations(range(n), 2):
            if p > q:
                add(space.single(p, q, spin), ('s', spin, p, q))
    pairs = list(itertools.combinations(range(n), 2))
    for spin in (0, 1):
        for (p, r), (q, s) in itertools.product(pairs, pairs):
            if (p, r) > (q, s):
                add(space.double_same(p, r, s, q, spin), ('dss', spin, p, r, q, s))
    ab = list(itertools.product(range(n), range(n)))
    for (p, r), (q, s) in itertools.product(ab, ab):
        if (p, r) > (q, s):
            add(space.double_ab(p, q, r, s), ('dab', p, r, q, s))
    return _dedupe(mats, labs)


def pool_sf(space):
    """Spin-free generalized singles and doubles: commute with S^2, S_z, S_+-."""
    n = space.norb
    mats, labs = [], []
    for p, q in itertools.permutations(range(n), 2):
        if p > q:
            T = space.E(p, q); mats.append((T - T.T).tocsr()); labs.append(('E', p, q))
    for p, q, r, s in itertools.product(range(n), repeat=4):
        T = space.e(p, q, r, s)
        A = (T - T.T).tocsr()
        mats.append(A); labs.append(('e', p, q, r, s))
    return _dedupe(mats, labs)


# ---------------------------------------------------------------- exponentials
class Generator:
    """exp(theta*A) for a real antisymmetric sparse A via an eigen-decomposition on its support."""

    def __init__(self, A):
        A = A.tocsr()
        sup = np.unique(np.concatenate([A.tocoo().row, A.tocoo().col]))
        self.sup = sup
        self.A = A
        Ad = A[sup][:, sup].toarray()
        w, V = np.linalg.eigh(1j * Ad)          # iA is Hermitian: A = V diag(-i w) V^H
        self.w, self.V = w, V
        self.VH = V.conj().T

    def apply(self, theta, x):
        y = x.copy()
        xs = x[self.sup]
        y[self.sup] = (self.V @ (np.exp(-1j * theta * self.w) * (self.VH @ xs))).real
        return y

    def dA(self, x):
        return self.A @ x


# ---------------------------------------------------------------- ADAPT-VQE
def state(gens, thetas, ref):
    psi = ref.copy()
    for g, t in zip(gens, thetas):
        psi = g.apply(t, psi)
    return psi


def energy_and_grad(thetas, gens, ref, H):
    psi = state(gens, thetas, ref)
    lam = H @ psi
    E = psi @ lam
    grad = np.zeros(len(gens))
    phi = psi
    for k in range(len(gens) - 1, -1, -1):
        grad[k] = 2 * lam @ gens[k].dA(phi)
        phi = gens[k].apply(-thetas[k], phi)
        lam = gens[k].apply(-thetas[k], lam)
    return E, grad


def adapt_vqe(space, H, ref, pool, observe=None, max_ops=80, gtol=1e-5, etol=1e-10,
              e_exact=None, verbose=False, score=None):
    """score: optional callable(psi, grads, pool_mats) -> selection scores (default |grads|, standard ADAPT).
    The convergence test always uses the energy gradients."""
    gens, thetas, chosen, history = [], np.zeros(0), [], []
    pool_mats = [M for M, _ in pool]
    psi = ref.copy()
    rec = {'n': 0, 'E': psi @ (H @ psi)}
    if observe:
        rec.update(observe(psi))
    history.append(rec)
    for it in range(max_ops):
        sigma = H @ psi
        grads = np.array([2 * sigma @ (A @ psi) for A in pool_mats])
        gmax = np.abs(grads).max()
        k = int(np.argmax(np.abs(grads) if score is None else score(psi, grads, pool_mats)))
        if gmax < gtol:
            break
        if e_exact is not None and history[-1]['E'] - e_exact < etol:
            break
        gens.append(Generator(pool_mats[k])); chosen.append(pool[k][1])
        x0 = np.append(thetas, 0.0)
        res = minimize(energy_and_grad, x0, args=(gens, ref, H), jac=True, method='BFGS',
                       options={'gtol': 1e-9, 'maxiter': 2000})
        thetas = res.x
        psi = state(gens, thetas, ref)
        rec = {'n': len(gens), 'E': res.fun, 'gmax': gmax, 'g_chosen': abs(grads[k]), 'op': pool[k][1]}
        if observe:
            rec.update(observe(psi))
        history.append(rec)
        if verbose:
            print(len(gens), rec)
    return history, chosen, thetas, psi
