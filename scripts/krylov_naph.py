"""Krylov refinement of D (krylov_test.py) for the stored 20-qubit naphthalene states in natural orbitals,
at the first iterate with chemical accuracy and 30 operators later. Writes results/krylov_naph.json."""
import json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import vqe_sparse as sps
from xop import XOperator
from fockmap import d_prefactor
from zfs import HARTREE2CM
from krylov_test import krylov_ritz
dat = np.load('results/naph_setup_no.npz'); ncas, nelecas = 10, (6, 4)
space = sps.SparseSpace(ncas, nelecas); shape = (space.na, space.nb)
Hs = sps.Hamiltonian(dat['h1'], dat['eri'], float(dat['ecore']), ncas, nelecas)
H = lambda v: Hs(v, shape)
Fp = np.einsum('ai,bj,ab...->ij...', dat['V'], dat['V'], dat['F'])
X = XOperator(ncas, nelecas, Fp[2, 2]); cD = 1.5 * d_prefactor(1.0) * HARTREE2CM
T = dat['ci'].ravel(); Dex = cD * (T @ X(T)); E0 = float(dat['e_casci'])
out = {'D_exact': Dex}
for p in ['GSD', 'sfGSD']:
    states = np.load(f'results/naph_no_{p}_states.npy', mmap_mode='r')
    dE = [float(s @ H(np.array(s)) - E0) for s in states[180:320]]
    ic = 180 + next(i for i, x in enumerate(dE) if x < 1.6e-3)
    res = {}
    for n in [ic, min(ic + 30, len(states) - 1)]:
        psi = np.array(states[n]); res[n] = {}
        for k in range(1, 7):
            x = psi if k == 1 else krylov_ritz(psi, H, k)
            res[n][k] = {'err': float(100 * (cD * x @ X(x) - Dex) / Dex), 'dE': float(x @ H(x) - E0)}
        print(p, 'n', n, ' '.join(f"k{k}:{res[n][k]['err']:+.2f}%({1e3*res[n][k]['dE']:.2f}mEh)" for k in res[n]), flush=True)
    out[p] = {'n_chem': ic, 'res': res}
json.dump(out, open('results/krylov_naph.json', 'w'), indent=1)
