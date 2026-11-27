"""Production driver: classical references + ADAPT-VQE with three pools for one system."""
import sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from pyscf import gto, scf, mcscf, fci, ao2mo
from zfs import ss_mo_ints, d_tensor, HARTREE2CM
from vqe import CISpace, pool_gsd, pool_sf, adapt_vqe


def tmm_geometry(rcc=1.40, rch=1.08):
    atoms = [['C', (0.0, 0.0, 0.0)]]
    for k in range(3):
        phi = 2 * np.pi * k / 3
        u = np.array([np.cos(phi), np.sin(phi), 0.0])
        c = rcc * u
        atoms.append(['C', tuple(c)])
        for sgn in (+1, -1):
            a = phi + sgn * np.pi / 3
            atoms.append(['H', tuple(c + rch * np.array([np.cos(a), np.sin(a), 0.0]))])
    return atoms


SYSTEMS = {
    # name: (atom, ncas, nelecas(alpha,beta), note)
    'CH2': ('C 0 0 0; H 0 0.98963 -0.42083; H 0 -0.98963 -0.42083', 6, (4, 2),
            'X 3B1, r(CH)=1.0753 A, HCH=133.93 deg'),
    'NH': ('N 0 0 0; H 0 0 1.0362', 6, (4, 2), 'X 3Sigma-, r=1.0362 A'),
    'NH_stretch': ('N 0 0 0; H 0 0 2.0724', 6, (4, 2), 'X 3Sigma-, r=2.0724 A (2 r_e)'),
    'O2': ('O 0 0 0; O 0 0 1.2075', 6, (5, 3), 'X 3Sigma_g-, r=1.2075 A'),
    'TMM': (tmm_geometry(), 4, (3, 1), "3A2' D3h, pi space, r(CC)=1.40 A"),
}


def select_pi_orbitals(mol, mf, ncas, nelecas):
    """Pick the ncas pi orbitals (odd under z -> -z) closest to the frontier for planar TMM."""
    labels = mol.ao_labels()
    pz = np.array([('pz' in l) or ('dxz' in l) or ('dyz' in l) for l in labels])
    S = mol.intor('int1e_ovlp')
    C = mf.mo_coeff
    w = np.einsum('mi,mn,ni->i', C[pz], S[np.ix_(pz, pz)], C[pz])  # weight on out-of-plane AOs
    pi_idx = [i for i in range(C.shape[1]) if w[i] > 0.9]
    nocc = int(round(sum(mf.mo_occ)))
    homo = max(i for i in range(len(mf.mo_occ)) if mf.mo_occ[i] > 0)
    pi_idx = sorted(pi_idx, key=lambda i: abs(i - homo))[:ncas]
    return sorted(pi_idx)


def axis_gauge(mol, mf, ncore, ncas, tol=1e-5):
    """Fix the gauge of degenerate active ROHF orbitals for a molecule on the z axis.

    Each exactly degenerate pair (orbital energies within tol) is rotated to pi_x / pi_y: the first orbital has
    no weight on y-type AOs (py, dyz), the second carries all of it; the largest coefficient of each orbital is
    made positive. Non-linear molecules are returned unchanged. Without this step the orientation returned by
    the eigensolver is arbitrary and can differ between machines and library builds.
    """
    C = mf.mo_coeff.copy()
    if not np.allclose(mol.atom_coords()[:, :2], 0.0):
        return C
    Y = np.array([('py' in l) or ('dyz' in l) for l in mol.ao_labels()])
    e = mf.mo_energy; act = list(range(ncore, ncore + ncas)); used = set()
    for i in act:
        for j in act:
            if j > i and i not in used and j not in used and abs(e[i] - e[j]) < tol:
                B = C[:, [i, j]]
                _, U = np.linalg.eigh(B[Y].T @ B[Y])          # ascending y-weight: pi_x, then pi_y
                B = B @ U
                for k in range(2):
                    B[:, k] *= np.sign(B[np.argmax(np.abs(B[:, k])), k])
                C[:, i], C[:, j] = B[:, 0], B[:, 1]
                used |= {i, j}
    return C


def run(name, max_ops=80):
    atom, ncas, nelecas, note = SYSTEMS[name]
    mol = gto.M(atom=atom, basis='cc-pvdz', spin=nelecas[0] - nelecas[1], verbose=0)
    mf = scf.ROHF(mol).run(conv_tol=1e-11)
    S = 0.5 * (nelecas[0] - nelecas[1])
    out = {'name': name, 'note': note, 'basis': 'cc-pVDZ', 'ncas': ncas, 'nelecas': nelecas,
           'nqubits': 2 * ncas, 'E_ROHF': mf.e_tot}

    mc = mcscf.CASCI(mf, ncas, nelecas)
    if name == 'TMM':
        idx = select_pi_orbitals(mol, mf, ncas, nelecas)
        mo = mc.sort_mo([i + 1 for i in idx])
        mc.mo_coeff = mo
    else:
        mc.mo_coeff = axis_gauge(mol, mf, mc.ncore, ncas)
    mc.fcisolver.conv_tol = 1e-12
    mc.fix_spin_(ss=S * (S + 1))
    mc.kernel()
    Cact = mc.mo_coeff[:, mc.ncore:mc.ncore + ncas]
    F = ss_mo_ints(mol, Cact)
    h1, ecore = mc.get_h1eff()
    eri = ao2mo.restore(1, mc.get_h2eff(), ncas)
    space = CISpace(ncas, nelecas)
    na, nb = len(space.sa), len(space.sb)

    # Hamiltonian (dense, small) from PySCF
    h2 = fci.direct_spin1.absorb_h1e(h1, eri, ncas, nelecas, 0.5)
    Hd = np.zeros((space.dim, space.dim))
    for i in range(space.dim):
        e = np.zeros(space.dim); e[i] = 1
        Hd[:, i] = fci.direct_spin1.contract_2e(h2, e.reshape(na, nb), ncas, nelecas).ravel()
    Hd = 0.5 * (Hd + Hd.T) + ecore * np.eye(space.dim)

    ci_ref = mc.ci.ravel()
    _, dm2s = fci.direct_spin1.make_rdm12s(mc.ci, ncas, nelecas)
    D_ref = d_tensor(F, dm2s, S)
    w, V = np.linalg.eigh(D_ref)
    order = np.argsort(np.abs(w)); V = V[:, order]       # x, y, z = principal axes of exact D

    def DE(D):
        Dp = V.T @ D @ V
        Dv, Ev = 1.5 * Dp[2, 2], 0.5 * (Dp[0, 0] - Dp[1, 1])
        return Dv * HARTREE2CM, Ev * HARTREE2CM
    Dref_v, Eref_v = DE(D_ref)
    if Eref_v * Dref_v < 0:  # fix x/y labels so that E/D >= 0
        V = V[:, [1, 0, 2]]; Dref_v, Eref_v = DE(D_ref)
    out.update({'E_CASCI': mc.e_tot,
                'S2_CASCI': float(fci.spin_op.spin_square(mc.ci, ncas, nelecas)[0]),
                'D_CASCI': Dref_v, 'E_zfs_CASCI': Eref_v,
                'Dtensor_CASCI_cm': (D_ref * HARTREE2CM).tolist()})

    # ROHF determinant value (in the same active space) for context
    ref = np.zeros(space.dim)
    ia = space.ia[(1 << nelecas[0]) - 1]; ib = space.ib[(1 << nelecas[1]) - 1]
    ref[ia * nb + ib] = 1.0

    # exact projector onto the target spin S within the M_S = S sector
    S2m = np.zeros((space.dim, space.dim))
    for i in range(space.dim):
        e = np.zeros(space.dim); e[i] = 1
        S2m[:, i] = fci.spin_op.contract_ss(e.reshape(na, nb), ncas, nelecas).ravel()
    ws, Vs = np.linalg.eigh(0.5 * (S2m + S2m.T))
    Vt = Vs[:, np.abs(ws - S * (S + 1)) < 1e-6]

    def observe(psi):
        ci = psi.reshape(na, nb)
        s2 = float(fci.spin_op.spin_square(ci, ncas, nelecas)[0])
        _, d2 = fci.direct_spin1.make_rdm12s(ci, ncas, nelecas)
        D = d_tensor(F, d2, S)
        Dv, Ev = DE(D)
        pp = Vt @ (Vt.T @ psi)
        wt = float(pp @ pp)                       # weight of the target spin component
        pp /= np.sqrt(wt)
        _, d2p = fci.direct_spin1.make_rdm12s(pp.reshape(na, nb), ncas, nelecas)
        Dp = d_tensor(F, d2p, S)
        Dpv, Epv = DE(Dp)
        return {'S2': s2, 'D': Dv, 'Ezfs': Ev, 'w_target': wt, 'D_proj': Dpv, 'Ezfs_proj': Epv,
                'dD_frob_cm': float(np.linalg.norm(D - D_ref) * HARTREE2CM),
                'dD_contam_frob_cm': float(np.linalg.norm(D - Dp) * HARTREE2CM),
                'fid': float((psi @ ci_ref) ** 2)}
    r0 = observe(ref)
    out['D_ROHF'] = r0['D']; out['E_zfs_ROHF'] = r0['Ezfs']

    # CASSCF (orbital-optimized) classical comparison
    mcs = mcscf.CASSCF(mf, ncas, nelecas)
    if name == 'TMM':
        mcs.fix_spin_(ss=S * (S + 1)); mcs.kernel(mc.mo_coeff)
    else:
        mcs.fix_spin_(ss=S * (S + 1)); mcs.kernel()
    Fs = ss_mo_ints(mol, mcs.mo_coeff[:, mcs.ncore:mcs.ncore + ncas])
    _, d2s = fci.direct_spin1.make_rdm12s(mcs.ci, ncas, nelecas)
    Ds = d_tensor(Fs, d2s, S)
    out['D_CASSCF'], out['E_zfs_CASSCF'] = DE(Ds)
    out['E_CASSCF'] = mcs.e_tot

    pools = {'GSD': pool_gsd(space), 'QEB': pool_gsd(space, qubit=True), 'sfGSD': pool_sf(space)}
    out['pool_sizes'] = {k: len(v) for k, v in pools.items()}
    out['runs'] = {}
    for pname, pool in pools.items():
        t0 = time.time()
        hist, chosen, thetas, psi = adapt_vqe(space, Hd, ref, pool, observe=observe,
                                              max_ops=max_ops, e_exact=mc.e_tot)
        for h in hist:
            h['dE'] = h['E'] - mc.e_tot
            h.pop('op', None)
        out['runs'][pname] = {'history': hist, 'time_s': time.time() - t0,
                              'ops': [list(map(lambda z: int(z) if not isinstance(z, str) else z, c)) for c in chosen]}
        last = hist[-1]
        print(f"{name:10s} {pname:6s} nops={last['n']:3d} dE={last['dE']:.2e} S2={last['S2']:.6f} "
              f"D={last['D']:.5f} (ref {Dref_v:.5f}) E={last['Ezfs']:.5f} (ref {Eref_v:.5f}) "
              f"t={time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(f'results/{name}.json', 'w'), indent=1, default=float)
    return out


if __name__ == '__main__':
    out = run(sys.argv[1], max_ops=int(sys.argv[2]) if len(sys.argv) > 2 else 80)
    print({k: v for k, v in out.items() if k not in ('runs', 'Dtensor_CASCI_cm')})
