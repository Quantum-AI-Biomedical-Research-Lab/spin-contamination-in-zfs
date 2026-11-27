"""Validation of the determinant-space operator machinery used by ADAPT-VQE."""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pyscf import gto, scf, mcscf, fci, ao2mo
import openfermion as of
from vqe import CISpace, pool_gsd, pool_sf, Generator
from fockmap import pyscf_to_fock

rng = np.random.default_rng(1)
mol = gto.M(atom='C 0 0 0; H 0 0.98963 -0.42083; H 0 -0.98963 -0.42083',
            basis='cc-pvdz', spin=2, verbose=0)
mf = scf.ROHF(mol).run()
mc = mcscf.CASCI(mf, 6, (4, 2))
h1, ecore = mc.get_h1eff()
eri = ao2mo.restore(1, mc.get_h2eff(), 6)
n, nelec = 6, (4, 2)
space = CISpace(n, nelec)

# 1. Hamiltonian assembled from spin-free E_pq, e_pqrs vs PySCF contract_2e
H = sum(h1[p, q] * space.E(p, q) for p in range(n) for q in range(n))
for p in range(n):
    for q in range(n):
        for r in range(n):
            for s in range(n):
                if abs(eri[p, q, r, s]) > 1e-14:
                    H = H + 0.5 * eri[p, q, r, s] * space.e(p, q, r, s)
x = rng.standard_normal(space.dim)
h2 = fci.direct_spin1.absorb_h1e(h1, eri, n, nelec, 0.5)
ref = fci.direct_spin1.contract_2e(h2, x.reshape(len(space.sa), -1), n, nelec).ravel()
print('1. |H_ours x - H_pyscf x| / |H x| =', np.linalg.norm(H @ x - ref) / np.linalg.norm(ref))

# 2. symmetry of pools under S^2
def ss(v):
    return fci.spin_op.contract_ss(v.reshape(len(space.sa), -1), n, nelec).ravel()
sf = pool_sf(space)
gsd = pool_gsd(space)
qeb = pool_gsd(space, qubit=True)
print('2. pool sizes: sfGSD', len(sf), ' GSD', len(gsd), ' QEB', len(qeb))
def max_comm(pool):
    m = 0.0
    for A, _ in pool:
        m = max(m, np.linalg.norm(ss(A @ x) - A @ ss(x)) / np.linalg.norm(x))
    return m
print('   max ||[S^2, A] x||: sfGSD', max_comm(sf), ' GSD', max_comm(gsd), ' QEB', max_comm(qeb))

# 3. QEB matrices vs explicit qubit-excitation operators in OpenFermion
def qop_raise(i):
    return of.QubitOperator(f'X{i}', 0.5) + of.QubitOperator(f'Y{i}', -0.5j)
def qop_lower(i):
    return of.QubitOperator(f'X{i}', 0.5) + of.QubitOperator(f'Y{i}', 0.5j)
y = rng.standard_normal(space.dim)
xF, yF = pyscf_to_fock(x.reshape(len(space.sa), -1), n, nelec), pyscf_to_fock(y.reshape(len(space.sa), -1), n, nelec)
worst = 0.0
for A, lab in qeb[::7]:
    if lab[0] == 's':
        _, spin, p, q = lab
        T = qop_raise(2 * p + spin) * qop_lower(2 * q + spin)
    elif lab[0] == 'dss':
        _, spin, p, r, q, s = lab
        T = qop_raise(2 * p + spin) * qop_raise(2 * r + spin) * qop_lower(2 * s + spin) * qop_lower(2 * q + spin)
    else:
        _, p, r, q, s = lab
        T = qop_raise(2 * p) * qop_raise(2 * r + 1) * qop_lower(2 * s + 1) * qop_lower(2 * q)
    Aq = T - of.hermitian_conjugated(T)
    M = of.get_sparse_operator(Aq, n_qubits=2 * n)
    worst = max(worst, abs((yF @ (M @ xF)).real - y @ (A @ x)))
print('3. max |<y|A_QEB|x>_ours - <y|A_QEB|x>_OpenFermion| =', worst)

# 4. Generator exponential vs scipy expm
from scipy.linalg import expm
A = sf[123][0]
G = Generator(A)
print('4. |exp(0.37A)x (eig) - expm|  =', np.linalg.norm(G.apply(0.37, x) - expm(0.37 * A.toarray()) @ x))
