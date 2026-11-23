"""Naphthalene T1, CAS(10,10)/cc-pVDZ in CASSCF natural orbitals: reference for the 20-qubit runs.

The exact solution in this orbital basis is the CASSCF wavefunction itself. Natural orbitals are ordered by
decreasing occupation, so the high-spin reference determinant is the lowest alpha and beta strings.
Writes results/naph_setup_no.npz and results/naph_reference_no.json.
"""
import json
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pyscf import gto, scf, mcscf, fci, ao2mo
from zfs import ss_mo_ints, d_tensor, HARTREE2CM

t0 = time.time()
geom = json.load(open('results/naph_T1_geometry.json'))
mol = gto.M(atom=[[s, tuple(x)] for s, x in geom], basis='cc-pvdz', spin=2, verbose=0)
mf = scf.ROHF(mol).run(conv_tol=1e-10)
ncas, nelecas = 10, (6, 4)
ref_info = json.load(open('results/naph_reference.json'))
act = ref_info['active_rohf_indices']
mcs = mcscf.CASSCF(mf, ncas, nelecas)
mcs.natorb = True
mcs.fix_spin_(ss=2.0)
mcs.conv_tol = 1e-10
mcs.kernel(mcs.sort_mo([i + 1 for i in act]))
# re-solve the CI problem in the final natural-orbital basis (CASCI in CASSCF natural orbitals)
mc = mcscf.CASCI(mf, ncas, nelecas)
mc.fcisolver.conv_tol = 1e-12
mc.fix_spin_(ss=2.0)
mc.kernel(mcs.mo_coeff)
dm1a, dm1b = fci.direct_spin1.make_rdm1s(mc.ci, ncas, nelecas)
occ = np.diag(dm1a + dm1b)
print(f'CASSCF E = {mcs.e_tot:.10f} (converged {mcs.converged}); CASCI in its natural orbitals E = {mc.e_tot:.10f}')
print('natural occupations (diag of 1-RDM in active NO basis):', np.round(occ, 4))
assert np.all(np.diff(occ) <= 1e-6), 'natural orbitals not ordered by decreasing occupation'
Cact = mc.mo_coeff[:, mc.ncore:mc.ncore + ncas]
F = ss_mo_ints(mol, Cact)
_, d2 = fci.direct_spin1.make_rdm12s(mc.ci, ncas, nelecas)
D = d_tensor(F, d2, 1.0)
w, V = np.linalg.eigh(D); V = V[:, np.argsort(np.abs(w))]
Dp = V.T @ D @ V
Dex, Eex = 1.5 * Dp[2, 2] * HARTREE2CM, 0.5 * (Dp[0, 0] - Dp[1, 1]) * HARTREE2CM
na = fci.cistring.num_strings(ncas, 6); nb = fci.cistring.num_strings(ncas, 4)
ref = np.zeros((na, nb)); ref[0, 0] = 1.0
_, d2r = fci.direct_spin1.make_rdm12s(ref, ncas, nelecas)
Dr = V.T @ d_tensor(F, d2r, 1.0) @ V
h1, ecore = mc.get_h1eff()
eri = ao2mo.restore(1, mc.get_h2eff(), ncas)
e_ref = fci.direct_spin1.energy(h1, eri, ref, ncas, nelecas) + ecore
# lowest excited triplet in this active space (for the error estimate)
mc2 = mcscf.CASCI(mf, ncas, nelecas); mc2.fcisolver.nroots = 3; mc2.fix_spin_(ss=2.0)
mc2.fcisolver.conv_tol = 1e-10
mc2.kernel(mcs.mo_coeff)
e_roots = [float(e) for e in mc2.e_tot]
out = {'E_CASSCF': mcs.e_tot, 'E_CASCI_NO': mc.e_tot, 'E_refdet_NO': e_ref, 'casscf_converged': bool(mcs.converged),
       'D_exact_NO': Dex, 'E_zfs_exact_NO': Eex, 'D_refdet_NO': 1.5 * Dr[2, 2] * HARTREE2CM,
       'natural_occupations': occ.tolist(), 'triplet_roots': e_roots, 'gap_mEh': 1e3 * (e_roots[1] - e_roots[0]),
       'setup_time_s': time.time() - t0}
json.dump(out, open('results/naph_reference_no.json', 'w'), indent=1)
np.savez('results/naph_setup_no.npz', h1=h1, eri=eri, ecore=ecore, F=F, V=V, ci=mc.ci, e_casci=mc.e_tot,
         e1=e_roots[1])
for k, v in out.items():
    print(k, v)
