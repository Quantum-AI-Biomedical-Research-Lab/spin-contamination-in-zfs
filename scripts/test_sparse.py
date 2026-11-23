"""Validate vqe_sparse.py against the dense implementation (vqe.py) on CH2 CAS(6,6) and O2 CAS(8,6)."""
import json
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pyscf import gto, scf, mcscf, fci, ao2mo
import vqe as dense
import vqe_sparse as sps
from run import SYSTEMS

for name in ['CH2', 'O2']:
    atom, ncas, nelecas, _ = SYSTEMS[name]
    mol = gto.M(atom=atom, basis='cc-pvdz', spin=nelecas[0] - nelecas[1], verbose=0)
    mf = scf.ROHF(mol).run(conv_tol=1e-11)
    mc = mcscf.CASCI(mf, ncas, nelecas); mc.fcisolver.conv_tol = 1e-12; mc.fix_spin_(ss=2.0); mc.kernel()
    h1, ecore = mc.get_h1eff(); eri = ao2mo.restore(1, mc.get_h2eff(), ncas)
    ds = dense.CISpace(ncas, nelecas); ss = sps.SparseSpace(ncas, nelecas)
    assert np.allclose(ds.par, ss.P.ravel()), 'parity mismatch'
    H = sps.Hamiltonian(h1, eri, ecore, ncas, nelecas)
    rng = np.random.default_rng(3)
    x = rng.standard_normal(ss.dim); y = rng.standard_normal(ss.dim)
    for pname, dpool, spool in [('GSD', dense.pool_gsd(ds), sps.pool_gsd(ss)),
                                ('QEB', dense.pool_gsd(ds, qubit=True), sps.pool_gsd(ss, qubit=True)),
                                ('sfGSD', dense.pool_sf(ds), sps.pool_sf(ss))]:
        gd = np.array([y @ (A @ x) for A, _ in dpool])
        gs = np.array([sps.op_gradient(ss, o, y.reshape(ss.na, ss.nb), x.reshape(ss.na, ss.nb)) for o in spool])
        same_order = len(gd) == len(gs) and np.allclose(gd, gs)
        # generator matrices and exponentials for a sample of operators
        err_m = err_e = 0.0
        for k in range(0, len(spool), max(1, len(spool) // 25)):
            A = sps.op_matrix(ss, spool[k]); Ad = dpool[k][0] if same_order else None
            if Ad is not None:
                err_m = max(err_m, abs(A - Ad).max())
            G = sps.BlockGenerator(A)
            from scipy.linalg import expm
            err_e = max(err_e, np.linalg.norm(G.apply(0.37, x) - expm(0.37 * A.toarray()) @ x))
        print(f"{name:3s} {pname:5s} pool sizes dense {len(dpool)} sparse {len(spool)}; gradients identical "
              f"in order: {same_order} (max diff {np.abs(gd - gs).max() if len(gd) == len(gs) else 'n/a'}); "
              f"max |A_sparse - A_dense| {err_m:.1e}; max exp error {err_e:.1e}")
    # Hamiltonian
    Hd = sum(h1[p, q] * ds.E(p, q) for p in range(ncas) for q in range(ncas))
    for p in range(ncas):
        for q in range(ncas):
            for r in range(ncas):
                for s in range(ncas):
                    if abs(eri[p, q, r, s]) > 1e-14:
                        Hd = Hd + 0.5 * eri[p, q, r, s] * ds.e(p, q, r, s)
    print(f"{name:3s} H: max |H_sparse x - H_dense x| = {np.abs(H(x, (ss.na, ss.nb)) - (Hd @ x + ecore * x)).max():.1e}")

    # full ADAPT runs: sparse and dense codes on the same Hamiltonian (same orbitals) in this process.
    # (Comparing with stored trajectories is not meaningful for O2: the orientation of its degenerate pi
    # orbitals, and hence the ADAPT path, can change between ROHF runs.)
    ref = np.zeros(ss.dim); ref[ss.ia[(1 << nelecas[0]) - 1] * ss.nb + ss.ib[(1 << nelecas[1]) - 1]] = 1
    Hdd = (Hd + ecore * __import__('scipy.sparse', fromlist=['identity']).identity(ss.dim)).toarray()
    Hdd = 0.5 * (Hdd + Hdd.T)
    for pname, pool, dpool in [('GSD', sps.pool_gsd(ss), dense.pool_gsd(ds)),
                               ('QEB', sps.pool_gsd(ss, qubit=True), dense.pool_gsd(ds, qubit=True)),
                               ('sfGSD', sps.pool_sf(ss), dense.pool_sf(ds))]:
        hist, chosen, *_ = sps.adapt_vqe(ss, H, ref, pool, max_ops=80, e_exact=mc.e_tot)
        hd, chd, *_ = dense.adapt_vqe(ds, Hdd, ref, dpool, max_ops=80, e_exact=mc.e_tot)
        Es = np.array([h['E'] for h in hist]); Ed = np.array([h['E'] for h in hd])
        m = min(len(Es), len(Ed))
        ops_same = [tuple(c) for c in chosen[:m - 1]] == [tuple(c) for c in chd[:m - 1]]
        print(f"{name:3s} {pname:5s} ADAPT: n_ops sparse {len(Es)-1} dense {len(Ed)-1}; same operator sequence: {ops_same}; "
              f"max |E_sparse - E_dense| over common steps {np.abs(Es[:m] - Ed[:m]).max():.1e}; "
              f"final dE sparse {Es[-1]-mc.e_tot:.1e} dense {Ed[-1]-mc.e_tot:.1e}", flush=True)
