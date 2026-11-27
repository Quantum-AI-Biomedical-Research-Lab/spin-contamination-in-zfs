"""Spin-spin D of O2 (X 3Sigma_g-) versus active space, orbitals and basis (classical CASCI/CASSCF)."""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pyscf import gto, scf, mcscf, fci
from zfs import ss_mo_ints, d_tensor, d_and_e, HARTREE2CM


def dss(mol, mc, S=1.0):
    ncas = mc.ncas
    F = ss_mo_ints(mol, mc.mo_coeff[:, mc.ncore:mc.ncore + ncas])
    _, d2 = mc.fcisolver.make_rdm12s(mc.ci, ncas, mc.nelecas)
    return d_and_e(d_tensor(F, d2, S))[0] * HARTREE2CM


rows = []
for basis in ['cc-pvdz', 'cc-pvtz']:
    mol = gto.M(atom='O 0 0 0; O 0 0 1.2075', basis=basis, spin=2, verbose=0)
    mf = scf.ROHF(mol).run(conv_tol=1e-11)
    dm = mf.make_rdm1()
    from pyscf.prop.zfs import uhf as zu
    dmu = mf.to_uhf().make_rdm1()
    d = zu.direct_spin_spin(None, mol, dmu); d -= np.trace(d) / 3 * np.eye(3)
    rows.append((basis, 'ROHF', '-', d_and_e(d)[0] * HARTREE2CM * (2.00231930436 / 2) ** 2, float(mf.e_tot), bool(mf.converged)))
    for ncas, ne in [(6, (5, 3)), (8, (7, 5)), (12, (7, 5))]:
        for kind in ['CASCI', 'CASSCF']:
            mc = (mcscf.CASCI if kind == 'CASCI' else mcscf.CASSCF)(mf, ncas, ne)
            mc.fix_spin_(ss=2.0)
            mc.fcisolver.max_cycle = 300
            if kind == 'CASSCF':
                mc.max_cycle_macro = 100
            mc.kernel()
            conv = bool(getattr(mc, 'converged', True))
            rows.append((basis, kind, f'({sum(ne)},{ncas})', dss(mol, mc), float(mc.e_tot), conv))
            print(rows[-1], flush=True)
import json
json.dump([{'basis': r[0], 'method': r[1], 'cas': r[2], 'D_SS_cm': float(r[3]),
            'E_tot': (r[4] if len(r) > 4 else None), 'converged': (r[5] if len(r) > 5 else True)} for r in rows],
          open('results/o2_convergence.json', 'w'), indent=1)
print('\n basis    method  CAS       D_SS (cm-1)  converged')
for r in rows:
    print(f" {r[0]:8s} {r[1]:7s} {r[2]:9s} {r[3]:.3f}   {r[5] if len(r) > 5 else '-'}")
