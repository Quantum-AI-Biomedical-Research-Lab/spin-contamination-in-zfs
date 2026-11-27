"""Post-run remedies (C1 Lanczos step, C2 subspace expansion with spin-free singles, D variance extrapolation)
applied to the saved 20-qubit naphthalene ADAPT states in CASSCF natural orbitals.
Writes results/naph_remedies.json and results/naph_remedies_<pool>.json."""
import json
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pyscf import fci
import vqe_sparse as sps
from xop import XOperator
from fockmap import d_prefactor
from zfs import HARTREE2CM
from remedies import M_EXTRAP, CHEM, metrics

dat = np.load('results/naph_setup_no.npz')
ncas, nelecas = 10, (6, 4)
space = sps.SparseSpace(ncas, nelecas); shape = (space.na, space.nb)
Hs = sps.Hamiltonian(dat['h1'], dat['eri'], float(dat['ecore']), ncas, nelecas)
H = lambda v: Hs(v, shape)
Fp = np.einsum('ai,bj,ab...->ij...', dat['V'], dat['V'], dat['F'])
X = XOperator(ncas, nelecas, Fp[2, 2])
cD = 1.5 * d_prefactor(1.0) * HARTREE2CM
E0 = float(dat['e_casci'])
Dex = cD * (dat['ci'].ravel() @ X(dat['ci'].ravel()))
# spin-free single excitations E_pq = E^a_pq + E^b_pq as (alpha-string, beta-string) maps
from xop import _one
Ea = [_one(ncas, 6, 6, [(p, 1), (q, 0)]) for p in range(ncas) for q in range(ncas)]
Eb = [_one(ncas, 4, 4, [(p, 1), (q, 0)]) for p in range(ncas) for q in range(ncas)]
sgn = (-1) ** 6   # beta operators pass N_alpha = 6 alpha electrons (even): no sign


def Esingle(k, v):
    C = v.reshape(shape)
    return ((Ea[k] @ C) + sgn * (C @ Eb[k].T)).ravel()


def lowest(vecs, tol=1e-10):
    B = np.array(vecs).T
    s, U = np.linalg.eigh(B.T @ B)
    keep = s > tol * s.max()
    Bo = B @ (U[:, keep] / np.sqrt(s[keep]))
    HB = np.array([H(Bo[:, i]) for i in range(Bo.shape[1])]).T
    e, c = np.linalg.eigh(Bo.T @ HB)
    v = Bo @ c[:, 0]
    return v / np.linalg.norm(v)


out = {'D_exact': Dex}
for p in ['GSD', 'QEB', 'sfGSD']:
    try:
        states = np.load(f'results/naph_no_{p}_states.npy')
        ops = json.load(open(f'results/naph_no_{p}.json'))['ops']
    except FileNotFoundError:
        continue
    from cnot_counts import cnots
    cn = np.cumsum([0] + [cnots(tuple(l), p == 'QEB')[0] for l in ops])
    rows = []
    for n, psi in enumerate(states):
        Hp = H(psi); E = psi @ Hp; r = Hp - E * psi
        row = {'n': n, 'cnot': int(cn[min(n, len(cn) - 1)]), 'dE': float(E - E0), 'sigma_H2': float(r @ r),
               'D': float(cD * psi @ X(psi))}
        v = lowest([psi, Hp]); row['D_lanczos'] = float(cD * v @ X(v))
        if n % 5 == 0 or E - E0 < 5e-3:      # subspace expansion only every 5th iterate early on (cost)
            v = lowest([psi] + [Esingle(k, psi) for k in range(ncas * ncas)]); row['D_qse'] = float(cD * v @ X(v))
        rows.append(row)
    for i, row in enumerate(rows):
        if i >= M_EXTRAP - 1:
            seg = rows[i - M_EXTRAP + 1:i + 1]
            x = np.array([s['sigma_H2'] for s in seg]); y = np.array([s['D'] for s in seg])
            row['D_extrap'] = float(np.polyfit(x, y, 1)[1]) if np.ptp(x) > 0 else row['D']
    res = {k: metrics(rows, key, Dex) for k, key in [('A', 'D'), ('C1', 'D_lanczos'), ('C2', 'D_qse'),
                                                      ('D', 'D_extrap')]}
    out[p] = res
    json.dump(rows, open(f'results/naph_remedies_{p}.json', 'w'))
    for k, v in res.items():
        print(p, k, v, flush=True)
json.dump(out, open('results/naph_remedies.json', 'w'), indent=1)
