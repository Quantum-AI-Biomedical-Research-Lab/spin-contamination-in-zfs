"""20-qubit ADAPT-VQE for naphthalene T1, CAS(10,10).
Usage (from the repository root): python scripts/naph_run.py GSD|QEB|sfGSD [max_ops] [hours] [orbitals: rohf|no]
rohf: ROHF orbitals (results/naph_setup.npz); no: CASSCF natural orbitals (results/naph_setup_no.npz).
Every iterate is saved to results/naph_<tag>_<pool>_states.npy for later analysis."""
import json
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pyscf import fci
from zfs import d_tensor, HARTREE2CM
import vqe_sparse as sps

pname = sys.argv[1]
max_ops = int(sys.argv[2]) if len(sys.argv) > 2 else 300
hours = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
orb = sys.argv[4] if len(sys.argv) > 4 else 'rohf'
tag = '' if orb == 'rohf' else '_no'
dat = np.load(f'results/naph_setup{tag}.npz')
ncas, nelecas, S = 10, (6, 4), 1.0
space = sps.SparseSpace(ncas, nelecas)
H = sps.Hamiltonian(dat['h1'], dat['eri'], float(dat['ecore']), ncas, nelecas)
F, V, ci_ex, e_ex = dat['F'], dat['V'], dat['ci'].ravel(), float(dat['e_casci'])
shape = (space.na, space.nb)


def DE(Dt):
    Dp = V.T @ Dt @ V
    return 1.5 * Dp[2, 2] * HARTREE2CM, 0.5 * (Dp[0, 0] - Dp[1, 1]) * HARTREE2CM


def proj_triplet(x):
    """Lowdin projector onto S = 1 within M_S = 1 (possible S' = 1..5)."""
    y = x.copy()
    for s in (2, 3, 4, 5):
        c = s * (s + 1)
        y = (fci.spin_op.contract_ss(y.reshape(shape), ncas, nelecas).ravel() - c * y) / (2.0 - c)
    return y


states = []


def observe(psi):
    states.append(psi.copy())
    c = psi.reshape(shape)
    s2 = float(fci.spin_op.spin_square(c, ncas, nelecas)[0])
    _, d2 = fci.direct_spin1.make_rdm12s(c, ncas, nelecas)
    Dv, Ev = DE(d_tensor(F, d2, S))
    pt = proj_triplet(psi); wt = float(pt @ pt)
    if wt < 1 - 1e-12:
        _, d2p = fci.direct_spin1.make_rdm12s((pt / np.sqrt(wt)).reshape(shape), ncas, nelecas)
        Dp, Ep = DE(d_tensor(F, d2p, S))
    else:
        Dp, Ep = Dv, Ev
    return {'S2': s2, 'D': Dv, 'Ezfs': Ev, 'D_proj': Dp, 'Ezfs_proj': Ep, 'w_target': wt,
            'fid': float((psi @ ci_ex) ** 2)}


pool = {'GSD': lambda: sps.pool_gsd(space), 'QEB': lambda: sps.pool_gsd(space, qubit=True),
        'sfGSD': lambda: sps.pool_sf(space)}[pname]()
ref = np.zeros(space.dim); ref[0] = 1.0     # high-spin ROHF determinant (lowest alpha and beta strings)
print(f'{pname}: pool size {len(pool)}, determinants {space.dim}', flush=True)
out = {'pool': pname, 'pool_size': len(pool), 'E_exact': e_ex}


def log(rec):
    print(f"{pname} n={rec['n']:3d} dE={rec['E'] - e_ex:9.2e} S2-2={rec['S2'] - 2:9.2e} D={rec['D']:.5f} "
          f"Dproj={rec['D_proj']:.5f} E={rec['Ezfs']:.5f} fid={rec['fid']:.6f} t={rec['t']:.0f}s", flush=True)
    out['history'] = hist_live
    json.dump(out, open(f'results/naph{tag}_{pname}.json', 'w'), indent=1, default=float)
    np.save(f'results/naph{tag}_{pname}_states.npy', np.array(states))


hist_live = []
_orig = log


def log_wrap(rec):
    hist_live.append(rec)
    _orig(rec)


t0 = time.time()
hist, chosen, thetas, psi = sps.adapt_vqe(space, H, ref, pool, observe=observe, max_ops=max_ops, gtol=1e-5,
                                         etol=1e-8, e_exact=e_ex, time_budget=3600 * hours, log=log_wrap)
for h in hist:
    h['dE'] = h['E'] - e_ex
out.update({'history': hist, 'ops': [list(c) for c in chosen], 'thetas': thetas.tolist(), 'time_s': time.time() - t0})
json.dump(out, open(f'results/naph{tag}_{pname}.json', 'w'), indent=1, default=lambda z: int(z) if isinstance(z, np.integer) else float(z))
np.save(f'results/naph{tag}_{pname}_states.npy', np.array(states))
print(f'{pname} finished: {len(chosen)} operators, final dE = {hist[-1]["dE"]:.2e}, time {time.time() - t0:.0f} s')
