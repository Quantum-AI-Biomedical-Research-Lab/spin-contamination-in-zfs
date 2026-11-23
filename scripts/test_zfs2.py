"""Validation II: correlated (CASCI) states — frozen-core embedding and operator route."""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pyscf import gto, scf, mcscf, fci, ao2mo
from pyscf.fci import cistring
import openfermion as of
from zfs import ss_mo_ints, d_tensor, d_and_e, HARTREE2CM
from fockmap import pyscf_to_fock, hamiltonian_op, d_component_op, d_prefactor

mol = gto.M(atom='C 0 0 0; H 0 0.98963 -0.42083; H 0 -0.98963 -0.42083',
            basis='cc-pvdz', spin=2, verbose=0)
mf = scf.ROHF(mol).run()
mc = mcscf.CASSCF(mf, 6, (4, 2))
mc.fix_spin_(ss=2.0)
mc.kernel()
print('CASSCF(6,6) E =', mc.e_tot, ' <S^2> =', mc.fcisolver.spin_square(mc.ci, 6, (4, 2)))

C = mc.mo_coeff
ncore = mc.ncore
Cact = C[:, ncore:ncore + 6]
F = ss_mo_ints(mol, Cact)
dm1s, dm2s = fci.direct_spin1.make_rdm12s(mc.ci, 6, (4, 2))
D_act = d_tensor(F, dm2s, S=1.0)
print('D tensor active-only (cm-1):\n', np.round(D_act * HARTREE2CM, 5))
print('D, E (cm-1):', [float(x * HARTREE2CM) for x in d_and_e(D_act)[:2]])

# ---- embed in an enlarged space containing the frozen core explicitly
nbig = ncore + 6
na, nb = 4 + ncore, 2 + ncore
sa = cistring.make_strings(range(6), 4)
sb = cistring.make_strings(range(6), 2)
core_mask = (1 << ncore) - 1
ci_big = np.zeros((cistring.num_strings(nbig, na), cistring.num_strings(nbig, nb)))
for ia, a in enumerate(sa):
    A = (a << ncore) | core_mask
    for ib, b in enumerate(sb):
        B = (b << ncore) | core_mask
        ci_big[cistring.str2addr(nbig, na, A), cistring.str2addr(nbig, nb, B)] = mc.ci[ia, ib]
Cbig = C[:, :nbig]
Fbig = ss_mo_ints(mol, Cbig)
_, dm2s_big = fci.direct_spin1.make_rdm12s(ci_big, nbig, (na, nb))
D_big = d_tensor(Fbig, dm2s_big, S=1.0)
print('max |D_big - D_act| (cm-1):', np.abs(D_big - D_act).max() * HARTREE2CM)
# energy check of the embedding
h1 = Cbig.T @ mf.get_hcore() @ Cbig
eri = ao2mo.restore(1, ao2mo.kernel(mol, Cbig), nbig)
e_big = fci.direct_spin1.energy(h1, eri, ci_big, nbig, (na, nb)) + mol.energy_nuc()
print('embedding energy check: |E_big - E_CASSCF| =', abs(e_big - mc.e_tot))

# ---- operator route in the active space (OpenFermion, JW)
h1eff, ecore = mc.get_h1eff()
eri_act = ao2mo.restore(1, mc.get_h2eff(), 6)
H = hamiltonian_op(h1eff, eri_act, ecore, 6)
Hs = of.get_sparse_operator(H, n_qubits=12)
psi = pyscf_to_fock(mc.ci, 6, (4, 2))
print('Fock-map energy check: |<H> - E_CASSCF| =', abs(psi @ (Hs @ psi) - mc.e_tot))
for a in range(3):
    Oa = of.get_sparse_operator(d_component_op(F[a, a], 6, S=1.0), n_qubits=12)
    val = d_prefactor(1.0) * (psi @ (Oa @ psi)).real
    print(f'D_{"xyz"[a]*2}: operator route {val*HARTREE2CM:.8f}  RDM route {D_act[a,a]*HARTREE2CM:.8f} cm-1')
