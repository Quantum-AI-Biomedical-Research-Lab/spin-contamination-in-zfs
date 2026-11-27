"""Naphthalene T1, full pi space CAS(10,10)/cc-pVDZ: references for the 20-qubit ADAPT-VQE runs.

Writes results/naph_setup.npz (integrals, exact state, D-tensor frame) and results/naph_reference.json.
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

# pi orbitals: weight on out-of-plane (z) functions
labels = mol.ao_labels()
pz = np.array([('pz' in l) or ('dxz' in l) or ('dyz' in l) for l in labels])
S = mol.intor('int1e_ovlp')
C = mf.mo_coeff
wpi = np.einsum('mi,mn,ni->i', C[pz], S[np.ix_(pz, pz)], C[pz])
pi = [i for i in range(C.shape[1]) if wpi[i] > 0.9]
homo = max(i for i in range(len(mf.mo_occ)) if mf.mo_occ[i] > 0)
act = sorted(sorted(pi, key=lambda i: abs(i - homo))[:ncas])
mc = mcscf.CASCI(mf, ncas, nelecas)
mc.mo_coeff = mc.sort_mo([i + 1 for i in act])
mc.fcisolver.conv_tol = 1e-12
mc.fix_spin_(ss=2.0)
mc.kernel()
Cact = mc.mo_coeff[:, mc.ncore:mc.ncore + ncas]
wact = np.einsum('mi,mn,ni->i', Cact[pz], S[np.ix_(pz, pz)], Cact[pz])
print(f'ROHF E = {mf.e_tot:.8f}; active orbitals {act} (ROHF indices), pi weights {np.round(wact, 3)}')
print(f'CASCI(10,10) E = {mc.e_tot:.8f}, <S^2> = {fci.spin_op.spin_square(mc.ci, ncas, nelecas)[0]:.6f}')

F = ss_mo_ints(mol, Cact)
_, d2 = fci.direct_spin1.make_rdm12s(mc.ci, ncas, nelecas)
D = d_tensor(F, d2, 1.0)
w, V = np.linalg.eigh(D); V = V[:, np.argsort(np.abs(w))]


def DE(Dt):
    Dp = V.T @ Dt @ V
    return 1.5 * Dp[2, 2] * HARTREE2CM, 0.5 * (Dp[0, 0] - Dp[1, 1]) * HARTREE2CM


Dex, Eex = DE(D)
if Dex * Eex > 0:            # convention used for naphthalene: report E with sign of the molecular axes below
    pass
h1, ecore = mc.get_h1eff()
eri = ao2mo.restore(1, mc.get_h2eff(), ncas)
# ROHF determinant in the same active space
na = fci.cistring.num_strings(ncas, 6); nb = fci.cistring.num_strings(ncas, 4)
ref = np.zeros((na, nb)); ref[0, 0] = 1.0
_, d2r = fci.direct_spin1.make_rdm12s(ref, ncas, nelecas)
Drohf, Erohf = DE(d_tensor(F, d2r, 1.0))
# principal axes in the molecular frame (x long axis, y short, z normal)
axes = {'xyz'[int(np.argmax(np.abs(V[:, k])))]: k for k in range(3)}

# CASSCF(10,10) for classical context
mcs = mcscf.CASSCF(mf, ncas, nelecas)
mcs.fix_spin_(ss=2.0)
mcs.kernel(mc.mo_coeff)
Fs = ss_mo_ints(mol, mcs.mo_coeff[:, mcs.ncore:mcs.ncore + ncas])
_, d2s = mcs.fcisolver.make_rdm12s(mcs.ci, ncas, nelecas)
Ds = d_tensor(Fs, d2s, 1.0)
ws, Vs = np.linalg.eigh(Ds); Vs = Vs[:, np.argsort(np.abs(ws))]
Dcs = 1.5 * (Vs.T @ Ds @ Vs)[2, 2] * HARTREE2CM; Ecs = 0.5 * ((Vs.T @ Ds @ Vs)[0, 0] - (Vs.T @ Ds @ Vs)[1, 1]) * HARTREE2CM

out = {'E_ROHF': mf.e_tot, 'E_CASCI': mc.e_tot, 'E_CASSCF': mcs.e_tot, 'casscf_converged': bool(mcs.converged),
       'D_ROHF': Drohf, 'E_zfs_ROHF': Erohf, 'D_CASCI': Dex, 'E_zfs_CASCI': Eex, 'D_CASSCF': Dcs, 'E_zfs_CASSCF': Ecs,
       'Dtensor_CASCI_cm': (D * HARTREE2CM).tolist(), 'principal_axes_columns': V.tolist(),
       'n_det': int(na * nb), 'nqubits': 2 * ncas, 'active_rohf_indices': act, 'pi_weights': wact.tolist(),
       'setup_time_s': time.time() - t0}
json.dump(out, open('results/naph_reference.json', 'w'), indent=1)
np.savez('results/naph_setup.npz', h1=h1, eri=eri, ecore=ecore, F=F, V=V, ci=mc.ci, e_casci=mc.e_tot)
for k, v in out.items():
    if k not in ('Dtensor_CASCI_cm', 'principal_axes_columns'):
        print(k, v)
