"""Electronic-structure convergence of the spin-spin D of naphthalene T1 (Section 3, Section S3).

CASSCF in natural orbitals for the full pi space CAS(10,10) and for sigma+pi spaces CAS(12,12) and CAS(14,14),
which add the highest occupied sigma and lowest virtual sigma* ROHF orbitals (one or two of each), in cc-pVDZ, and
CAS(10,10) in cc-pVTZ. D and E from the 2-RDM of CASCI in the CASSCF natural orbitals, as for the 20-qubit
reference (naph_setup_no.py). Usage (from the repository root): python scripts/naph_convergence.py <label> [<label> ...] with labels
dz_10, dz_12, dz_14, tz_10 (exact integrals) and dz_10_df, tz_10_df (density-fitted CASSCF orbitals;
the spin-dipole integrals stay exact). Appends to results/naph_convergence.json.
"""
import json, os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from pyscf import gto, scf, mcscf, fci, lib
from zfs import ss_mo_ints, d_tensor, HARTREE2CM

RUNS = {'dz_10': ('cc-pvdz', 0), 'dz_12': ('cc-pvdz', 1), 'dz_14': ('cc-pvdz', 2), 'tz_10': ('cc-pvtz', 0),
        'dz_10_df': ('cc-pvdz', 0), 'tz_10_df': ('cc-pvtz', 0)}
geom = json.load(open('results/naph_T1_geometry.json'))


def run(label):
    basis, nsig = RUNS[label]; t0 = time.time()
    mol = gto.M(atom=[[s, tuple(x)] for s, x in geom], basis=basis, spin=2, verbose=0, max_memory=10000)
    df_ = label.endswith('_df')
    mf = scf.ROHF(mol).density_fit().run(conv_tol=1e-10) if df_ else scf.ROHF(mol).run(conv_tol=1e-10)
    C = mf.mo_coeff
    xyz = mol.atom_coords(); assert np.abs(xyz[:, 2]).max() < 1e-6, 'molecule must lie in the z = 0 plane'
    # parity under the mirror plane z -> -z: pi orbitals are odd. wpi = (1 - parity)/2 is 1 for pi and 0 for sigma.
    pts = np.random.default_rng(1).uniform(-6, 6, (4000, 3)); pts[:, 2] = np.random.default_rng(2).uniform(0.2, 2.5, 4000)
    m1 = mol.eval_gto('GTOval_sph', pts) @ C; pr = pts.copy(); pr[:, 2] *= -1; m2 = mol.eval_gto('GTOval_sph', pr) @ C
    parity = np.einsum('pi,pi->i', m1, m2) / np.einsum('pi,pi->i', m1, m1); wpi = (1 - parity) / 2
    nocc_b, nocc_a = mol.nelec[1], mol.nelec[0]
    homo = nocc_a - 1
    pi = [i for i in range(C.shape[1]) if wpi[i] > 0.9]
    act_pi = sorted(sorted(pi, key=lambda i: abs(i - homo))[:10])
    sig_occ = [i for i in range(nocc_b) if wpi[i] < 0.1][-nsig:] if nsig else []
    sig_vir = [i for i in range(nocc_a, C.shape[1]) if wpi[i] < 0.1][:nsig] if nsig else []
    act = sorted(act_pi + sig_occ + sig_vir); ncas = len(act); nel = (6 + nsig, 4 + nsig)
    mcs = (mcscf.DFCASSCF(mf, ncas, nel) if df_ else mcscf.CASSCF(mf, ncas, nel)); mcs.natorb = True; mcs.fix_spin_(ss=2.0); mcs.conv_tol = 1e-8
    mcs.fcisolver.conv_tol = 1e-9; mcs.max_cycle_macro = 60
    mcs.kernel(mcs.sort_mo([i + 1 for i in act]))
    mc = (mcscf.DFCASCI(mf, ncas, nel) if df_ else mcscf.CASCI(mf, ncas, nel)); mc.fcisolver.conv_tol = 1e-10; mc.fix_spin_(ss=2.0); mc.kernel(mcs.mo_coeff)
    dm1a, dm1b = fci.direct_spin1.make_rdm1s(mc.ci, ncas, nel); occ = np.diag(dm1a + dm1b)
    Cact = mc.mo_coeff[:, mc.ncore:mc.ncore + ncas]
    t1 = time.time(); F = ss_mo_ints(mol, Cact); t_int = time.time() - t1
    _, d2 = fci.direct_spin1.make_rdm12s(mc.ci, ncas, nel)
    D = d_tensor(F, d2, 1.0); w, V = np.linalg.eigh(D); V = V[:, np.argsort(np.abs(w))]; Dp = V.T @ D @ V
    Dv, Ev = 1.5 * Dp[2, 2] * HARTREE2CM, 0.5 * (Dp[0, 0] - Dp[1, 1]) * HARTREE2CM
    s2 = fci.spin_op.spin_square(mc.ci, ncas, nel)[0]
    pi_w_act = [float(wpi[i]) for i in act]
    return dict(basis=basis, density_fitting=df_, cas=f'({sum(nel)},{ncas})', nao=mol.nao, act_rohf=act, sigma_occ=sig_occ, sigma_vir=sig_vir,
                pi_weights=pi_w_act, E_CASSCF=float(mcs.e_tot), converged=bool(mcs.converged), E_CASCI_NO=float(mc.e_tot),
                S2=float(s2), D_cm=float(Dv), E_cm=float(Ev), natural_occupations=occ.round(5).tolist(),
                ndet=int(fci.cistring.num_strings(ncas, nel[0]) * fci.cistring.num_strings(ncas, nel[1])),
                t_ints_s=t_int, t_total_s=time.time() - t0)


if __name__ == '__main__':
    fn = 'results/naph_convergence.json'
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    for lab in sys.argv[1:]:
        r = run(lab); out[lab] = r
        print(lab, {k: r[k] for k in ('basis', 'cas', 'nao', 'D_cm', 'E_cm', 'converged', 'S2', 'ndet', 't_ints_s', 't_total_s')},
              'occ', r['natural_occupations'], flush=True)
        json.dump(out, open(fn, 'w'), indent=1)
