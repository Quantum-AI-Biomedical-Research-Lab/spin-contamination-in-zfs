"""Exact shift-invert refinement of the ADAPT state at chemical accuracy: psi <- (H - sigma)^-1 psi, normalized,
with sigma = E(psi) - 3 mEh (below the exact energy for every state at chemical accuracy). One and two steps.
This is an idealized test of how fast D converges when the state is refined towards the exact eigenstate by
inverse iteration rather than by Krylov (Lanczos) steps. Writes results/shiftinvert_<tag>.json."""
import json, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse.linalg as sla

SHIFT = 3e-3


def refine(psi, Hm, dim, nsteps=2):
    out = []
    x = psi / np.linalg.norm(psi)
    for step in range(nsteps):
        E = x @ Hm(x)
        sigma = E - SHIFT
        A = sla.LinearOperator((dim, dim), matvec=lambda v: Hm(v) - sigma * v, dtype=float)
        y, info = sla.cg(A, x, x0=x / max(E - sigma, 1e-12), rtol=1e-11, maxiter=20000)
        x = y / np.linalg.norm(y)
        out.append((x.copy(), info))
    return out


if __name__ == '__main__':
    tag = sys.argv[1]
    res = {}
    if tag == 'small':
        import remedies, penalty
        from vqe import pool_gsd, pool_sf
        for name in ['CH2', 'NH', 'NH_stretch', 'O2']:
            sysd = remedies.build(name); space = sysd['space']; H, Xz, cD, E0, Dex = sysd['H'], sysd['Xz'], sysd['cD'], sysd['E0'], sysd['Dex']
            S2 = penalty.spin_matrix(space, space.norb, space.nelec)
            ws, Vs = np.linalg.eigh(S2); PT = Vs[:, np.abs(ws - 2.0) < 1e-6]
            for pname, mu in [('GSD', 0.0), ('QEB', 0.0), ('sfGSD', 0.0), ('QEB', 3.0)]:
                pool = pool_sf(space) if pname == 'sfGSD' else pool_gsd(space, qubit=(pname == 'QEB'))
                r = penalty.run(sysd, S2, PT, pool, pname, mu, 0.0, keep_states=True)
                ic = next(i for i, q in enumerate(r['rows']) if q['dE'] < 1.6e-3)
                psi = r['states'][ic]
                errs = [100 * (cD * psi @ Xz @ psi - Dex) / Dex]; dEs = [psi @ H @ psi - E0]
                for x, info in refine(psi, lambda v: H @ v, space.dim):
                    errs.append(100 * (cD * x @ Xz @ x - Dex) / Dex); dEs.append(x @ H @ x - E0)
                res[f'{name}_{pname}_mu{mu:g}'] = {'err': errs, 'dE': dEs}
                print(f"{name:10s} {pname:5s} mu={mu:g}: " + ' -> '.join(f"{e:+.3f}% ({1e3*d:.2g} mEh)" for e, d in zip(errs, dEs)), flush=True)
    else:
        import vqe_sparse as sps
        from xop import XOperator
        from fockmap import d_prefactor
        from zfs import HARTREE2CM
        dat = np.load('results/naph_setup_no.npz'); ncas, nelecas = 10, (6, 4)
        space = sps.SparseSpace(ncas, nelecas); shape = (space.na, space.nb)
        Hs = sps.Hamiltonian(dat['h1'], dat['eri'], float(dat['ecore']), ncas, nelecas)
        Hm = lambda v: Hs(v, shape)
        Fp = np.einsum('ai,bj,ab...->ij...', dat['V'], dat['V'], dat['F'])
        X = XOperator(ncas, nelecas, Fp[2, 2]); cD = 1.5 * d_prefactor(1.0) * HARTREE2CM
        T = dat['ci'].ravel(); Dex = cD * (T @ X(T)); E0 = float(dat['e_casci'])
        for p, n in [('GSD', 291), ('sfGSD', 191)]:
            psi = np.array(np.load(f'results/naph_no_{p}_states.npy', mmap_mode='r')[n])
            errs = [100 * (cD * psi @ X(psi) - Dex) / Dex]; dEs = [psi @ Hm(psi) - E0]
            for x, info in refine(psi, Hm, space.dim):
                errs.append(100 * (cD * x @ X(x) - Dex) / Dex); dEs.append(x @ Hm(x) - E0)
            res[f'naph_{p}'] = {'err': errs, 'dE': dEs}
            print(f"naph {p}: " + ' -> '.join(f"{e:+.3f}% ({1e3*d:.2g} mEh)" for e, d in zip(errs, dEs)), flush=True)
    json.dump(res, open(f'results/shiftinvert_{tag}.json', 'w'), indent=1, default=float)
