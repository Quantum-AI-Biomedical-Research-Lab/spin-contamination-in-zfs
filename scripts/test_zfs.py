"""Validation of the spin-spin ZFS implementation."""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pyscf import gto, scf, mcscf, fci
from zfs import ss_mo_ints, d_tensor, spin_bracket, ALPHA, d_and_e, HARTREE2CM


def det_rdm2s(occa, occb, n):
    """Spin-resolved 2-RDMs of a single determinant (PySCF convention)."""
    da = np.zeros((n, n)); db = np.zeros((n, n))
    for i in occa: da[i, i] = 1
    for i in occb: db[i, i] = 1
    aa = np.einsum('pq,rs->pqrs', da, da) - np.einsum('ps,rq->pqrs', da, da)
    bb = np.einsum('pq,rs->pqrs', db, db) - np.einsum('ps,rq->pqrs', db, db)
    ab = np.einsum('pq,rs->pqrs', da, db)
    return aa, ab, bb


print('=== Test 1: two point spins (tight s Gaussians), triplet M_S=1 ===')
R = 4.0  # bohr
for axis in ([0, 0, 1], [1, 0, 0], [1, 1, 0]):
    u = np.array(axis, float); u /= np.linalg.norm(u)
    A = -0.5 * R * u; B = 0.5 * R * u
    mol = gto.M(atom=[['ghost-H', A], ['ghost-H', B]], unit='Bohr',
                basis={'ghost-H': gto.basis.parse('''
H S
  2000.0 1.0
''')}, spin=0, charge=0, verbose=0)
    S = mol.intor('int1e_ovlp')
    C = np.linalg.inv(np.linalg.cholesky(S)).T  # orthonormal (overlap ~0 anyway)
    F = ss_mo_ints(mol, C)
    D = d_tensor(F, det_rdm2s([0, 1], [], 2), S=1.0, g=2.0)
    r = B - A
    f = (R * R * np.eye(3) - 3 * np.outer(r, r)) / R ** 5
    Dref = ALPHA ** 2 / 2 * f
    print(axis, 'max|D-Dref|/max|Dref| =', np.abs(D - Dref).max() / np.abs(Dref).max())

print('\n=== Test 2: ROHF triplet CH2 vs PySCF-properties McWeeny UHF formula ===')
from pyscf.prop.zfs import uhf as zfs_uhf
mol = gto.M(atom='''C 0 0 0.1027; H 0 0.9893 -0.3082; H 0 -0.9893 -0.3082''',
            basis='cc-pvdz', spin=2, verbose=0)
mf = scf.ROHF(mol).run()
mfu = mf.to_uhf()
dma, dmb = mfu.make_rdm1()
d_prop = zfs_uhf.direct_spin_spin(None, mol, (dma, dmb))
d_prop = d_prop - np.trace(d_prop) / 3 * np.eye(3)
C = mf.mo_coeff
n = C.shape[1]
occa = [i for i in range(n) if mf.mo_occ[i] > 0]
occb = [i for i in range(n) if mf.mo_occ[i] > 1.5]
F = ss_mo_ints(mol, C)
D_all = d_tensor(F, det_rdm2s(occa, occb, n), S=1.0, g=2.0)
print('PySCF-prop (traceless) [cm-1]:\n', np.round(d_prop * HARTREE2CM, 5))
print('this work, all orbitals [cm-1]:\n', np.round(D_all * HARTREE2CM, 5))
print('max abs diff (cm-1):', np.abs(d_prop - D_all).max() * HARTREE2CM)

print('\n=== Test 3: frozen-core invariance (same ROHF determinant, active space only) ===')
ncore = len(occb) - 1  # keep one doubly occupied orbital in the active space
act = list(range(ncore, ncore + 6))
Ca = C[:, act]
Fa = ss_mo_ints(mol, Ca)
occa_a = [i - ncore for i in occa if i in act]
occb_a = [i - ncore for i in occb if i in act]
D_act = d_tensor(Fa, det_rdm2s(occa_a, occb_a, len(act)), S=1.0, g=2.0)
print('max abs diff active vs all (cm-1):', np.abs(D_act - D_all).max() * HARTREE2CM)
print('D, E (cm-1):', [x * HARTREE2CM for x in d_and_e(D_all)[:2]])
