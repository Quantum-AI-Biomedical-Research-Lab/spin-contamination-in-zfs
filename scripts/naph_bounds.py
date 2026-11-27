"""Fidelity bound (eq 9) and Gaussian-filter guarantee (eq 11) for 20-qubit naphthalene (natural orbitals).

Exact-state quantities: sigma_X(T) and its in-sector / out-of-sector parts, ||X - <X>_T|| (Lanczos), the first
excitation G_M of the M_S = 1 sector (lowest of the second triplet and the lowest state with S >= 2, i.e. the
ground state of the M_S = 2 sector), and a lower bound on the first excitation G' of H' = H + mu_f (S^2 - 2)
(S >= 2 states are shifted up by at least 4 mu_f). F, w and E(psi) - E0 at chemical accuracy are read from the
stored ADAPT histories (results/naph_no_<pool>.json), so the 20-qubit states are not needed.
Writes results/naph_bounds.json.
"""
import json, warnings; warnings.filterwarnings('ignore')
import numpy as np, scipy.sparse.linalg as sla
from pyscf import fci
import vqe_sparse as sps
from xop import XOperator
from fockmap import d_prefactor
from zfs import HARTREE2CM

MU_F, CHEM = 0.05, 1.6e-3
dat = np.load('results/naph_setup_no.npz')
ncas, nelecas, S = 10, (6, 4), 1.0
space = sps.SparseSpace(ncas, nelecas); shape = (space.na, space.nb)
Fp = np.einsum('ai,bj,ab...->ij...', dat['V'], dat['V'], dat['F'])
Xop = XOperator(ncas, nelecas, Fp[2, 2]); cD = 1.5 * d_prefactor(S) * HARTREE2CM
X = lambda v: cD * Xop(v)
S2 = lambda v: fci.spin_op.contract_ss(v.reshape(shape), ncas, nelecas).ravel()

def P1(x):                                   # Lowdin projector onto S = 1 within M_S = 1
    y = x.copy()
    for s in (2, 3, 4, 5):
        c = s * (s + 1); y = (S2(y) - c * y) / (2.0 - c)
    return y

T = dat['ci'].ravel(); T = T / np.linalg.norm(T); E0, E1 = float(dat['e_casci']), float(dat['e1'])
Dex = T @ X(T)
r = X(T) - Dex * T; sig = np.linalg.norm(r); rp = P1(r)
assert np.linalg.norm(P1(rp) - rp) < 1e-8 * max(1, np.linalg.norm(rp))
sig_par = np.linalg.norm(rp); sig_perp = np.sqrt(max(sig**2 - sig_par**2, 0.0))
L = sla.LinearOperator((space.dim, space.dim), matvec=lambda v: X(v) - Dex * v, dtype=float)
normX = abs(sla.eigsh(L, k=1, which='LM', tol=1e-8, return_eigenvectors=False)[0])
# sector spectrum
h1, eri, ecore = dat['h1'], dat['eri'], float(dat['ecore'])
e_ms1, _ = fci.direct_spin1.kernel(h1, eri, ncas, nelecas, ecore=ecore, nroots=1, conv_tol=1e-10)
e_ms2, _ = fci.direct_spin1.kernel(h1, eri, ncas, (7, 3), ecore=ecore, nroots=1, conv_tol=1e-10)
assert abs(e_ms1 - E0) < 1e-7, ('T is not the ground state of the M_S = 1 sector', e_ms1, E0)
G_M = min(E1, e_ms2) - E0
Gp = min(E1 - E0, e_ms2 + 4 * MU_F - E0)
Tr = 3.0 / (E1 - E0)
out = {'D': Dex, 'gap_mEh': 1e3 * (E1 - E0), 'E_lowest_Sge2_mEh': 1e3 * (e_ms2 - E0), 'G_M_mEh': 1e3 * G_M, 'Gp_lower_mEh': 1e3 * Gp,
       'T': Tr, 'sigmaX_over_D': sig / abs(Dex), 'sigma_par_over_D': sig_par / abs(Dex), 'sigma_perp_over_D': sig_perp / abs(Dex),
       'normX_over_D': normX / abs(Dex)}
filt = json.load(open('results/naph_filter.json'))
for p in ['GSD', 'QEB', 'sfGSD']:
    h = json.load(open(f'results/naph_no_{p}.json'))['history']
    ic = next((i for i, x in enumerate(h) if x['dE'] < CHEM), None)
    if ic is None:
        out[p] = None; continue
    x = h[ic]; Fid, w, eps = x['fid'], x['w_target'], x['dE']
    err = 100 * abs(x['D'] / Dex - 1)
    b_split = 2 * np.sqrt(Fid) * (sig_par * np.sqrt(max(w - Fid, 0)) + sig_perp * np.sqrt(max(1 - w, 0))) + (1 - Fid) * normX
    b_cs = 2 * np.sqrt(Fid * (1 - Fid)) * sig + (1 - Fid) * normX
    u = eps / G_M; b_prior = 2 * np.sqrt(u * (1 - u)) * sig + u * normX
    ratio = (1 - Fid) / Fid * np.exp(-Tr**2 * Gp * (Gp - 2 * eps)); Fg = 1 / (1 + ratio)
    bG = 2 * np.sqrt(Fg * (1 - Fg)) * sig + (1 - Fg) * normX
    fr = filt['runs'].get(p, {}) if isinstance(filt.get('runs'), dict) else {}
    out[p] = {'n': x['n'], 'eps_mEh': 1e3 * eps, 'F': Fid, 'w': w, 'infid': 1 - Fid, 'eckart_M': u, 'err_pct': err,
              'b11_split_pct': 100 * b_split / abs(Dex), 'b11_pct': 100 * b_cs / abs(Dex), 'b11_prior_pct': 100 * b_prior / abs(Dex),
              'infid_g_bound': 1 - Fg, 'filter_bound_pct': 100 * bG / abs(Dex), 'filter_stored': fr}
    o = out[p]
    print(f"naph NO {p:5s} n={o['n']} eps {o['eps_mEh']:.2f} mEh 1-F {o['infid']:.2e} (<= {u:.2e}) err {err:.2f}% "
          f"A-split {o['b11_split_pct']:.1f}% A {o['b11_pct']:.1f}% prior {o['b11_prior_pct']:.1f}% | filter: 1-F_g <= {1 - Fg:.1e}, D error <= {o['filter_bound_pct']:.2f}%", flush=True)
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items() if not isinstance(v, dict) and v is not None})
json.dump(out, open('results/naph_bounds.json', 'w'), indent=1, default=float)
