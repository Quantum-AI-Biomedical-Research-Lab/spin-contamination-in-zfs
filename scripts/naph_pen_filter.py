"""Exact shift-invert step and Gaussian filter (T = 3/(E1 - E0)) applied to the final state of the spin-penalized
20-qubit naphthalene run (naph_penalty.py GSD 3 0), rebuilt from its saved operators and parameters.
(Only the final parameters are saved, so the state at the first iterate with chemical accuracy cannot be rebuilt.)
Writes results/naph_pen_filter.json."""
import json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import vqe_sparse as sps
from xop import XOperator
from fockmap import d_prefactor
from zfs import HARTREE2CM
from shiftinvert_test import refine
from naph_filter_lib import gauss_filter
dat = np.load('results/naph_setup_no.npz'); ncas, nelecas = 10, (6, 4)
space = sps.SparseSpace(ncas, nelecas); shape = (space.na, space.nb)
Hs = sps.Hamiltonian(dat['h1'], dat['eri'], float(dat['ecore']), ncas, nelecas); Hm = lambda v: Hs(v, shape)
Fp = np.einsum('ai,bj,ab...->ij...', dat['V'], dat['V'], dat['F'])
X = XOperator(ncas, nelecas, Fp[2, 2]); cD = 1.5 * d_prefactor(1.0) * HARTREE2CM
T = dat['ci'].ravel(); T /= np.linalg.norm(T); Dex = cD * (T @ X(T)); E0 = float(dat['e_casci']); E1 = float(dat['e1'])
P = json.load(open('results/naph_penalty_GSD_mu3_w0.json'))
pool = sps.pool_gsd(space); bylabel = {tuple(o.label): o for o in pool}
gens = [sps.BlockGenerator(sps.op_matrix(space, bylabel[tuple(tuple(x) if isinstance(x, list) else x for x in lab)])) for lab in P['ops']]
ref = np.zeros(space.dim); ref[0] = 1.0
psi = sps._state(gens, np.array(P['thetas']), ref); psi /= np.linalg.norm(psi)
err = lambda v: float(100 * (cD * v @ X(v) / (v @ v) - Dex) / Dex)
out = {'n': len(P['ops']), 'dE_mEh': 1e3 * float(psi @ Hm(psi) - E0), 'D': err(psi),
       'D_logged': 100 * (P['history'][-1]['D'] - Dex) / Dex}
x1 = refine(psi, Hm, space.dim, nsteps=1)[0][0]; out['shiftinvert'] = err(x1)
y, Ng = gauss_filter(psi, Hm, 3.0 / (E1 - E0)); out['gauss'] = err(y); out['Ng'] = Ng
print(out, flush=True)
json.dump(out, open('results/naph_pen_filter.json', 'w'), indent=1)
