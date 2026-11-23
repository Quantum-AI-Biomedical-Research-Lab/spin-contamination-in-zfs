"""Variance-based error bars for D along ADAPT trajectories.

For psi = sqrt(w) T~ + sqrt(1-w) Q (M_S = S), with X the principal-axis operator (D = cD <X>):
  contamination:  |D(psi) - D(T~)| <= cD [2 sqrt(w(1-w)) sigma_X(T~) + (1-w) |<X>_Q - <X>_T~|]          (rigorous)
  target spin:    |D(T~) - D(T)|   <= cD [2 sigma_X(T) d + ||X - <X>_T|| d^2],  d <= sigma_H(T~)/(E1 - E(T~))  (rigorous)
  measurable estimate (first order, quantities of psi only, plus a gap estimate E1):
      b = 2 cD sigma_X(psi) [ sqrt(delta / (2(S+1))) + sigma_H(psi) / (sqrt(w) (E1 - E(psi))) ],  delta = <S^2> - S(S+1)
Usage (from the repository root): python scripts/errbar.py small            (12-qubit systems, dense, re-runs ADAPT)
       python errbar.py naph [no|rohf]   (20-qubit naphthalene, from saved states)
"""
import json
import sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pyscf import fci
from zfs import d_tensor, HARTREE2CM
from fockmap import d_prefactor


def analyse(states, X, H, S2, PT, E0, E1, Xex_T, Xnorm_shift, cD, S=1.0):
    rows = []
    for psi in states:
        Hpsi = H(psi); E = psi @ Hpsi; r = Hpsi - E * psi; sH = np.sqrt(r @ r)
        Xpsi = X(psi); xm = psi @ Xpsi; r = Xpsi - xm * psi; sX = np.sqrt(r @ r)
        delta = psi @ S2(psi) - S * (S + 1)
        t = PT(psi); w = t @ t; Tt = t / np.sqrt(w)
        HT = H(Tt); ET = Tt @ HT; r = HT - ET * Tt; sHT = np.sqrt(r @ r)
        XT = X(Tt); xT = Tt @ XT; r = XT - xT * Tt; sXT = np.sqrt(r @ r)
        if 1 - w > 1e-14:
            q = psi - np.sqrt(w) * Tt; q /= np.linalg.norm(q); xQ = q @ X(q)
        else:
            xQ = xT
        b_contam = abs(cD) * (2 * np.sqrt(max(w * (1 - w), 0)) * sXT + (1 - w) * abs(xQ - xT))
        d_b = sHT / (E1 - ET) if ET < E1 else np.inf
        b_target = abs(cD) * (2 * Xex_T['sigma'] * d_b + Xnorm_shift * d_b ** 2)
        b_est = (2 * abs(cD) * sX * (np.sqrt(max(delta, 0) / (2 * (S + 1))) + sH / (np.sqrt(w) * (E1 - E)))
                 if E < E1 else np.inf)          # no estimate above the next state of the same spin
        rows.append({'dE': E - E0, 'D': cD * xm, 'D_T': cD * xT, 'sigma_X': sX, 'err': abs(cD * xm - Xex_T['D']),
                     'err_contam': abs(cD * (xm - xT)), 'err_target': abs(cD * xT - Xex_T['D']),
                     'b_contam': b_contam, 'b_target': b_target, 'b_est': b_est, 'sigma_H': sH, 'delta': delta,
                     'w': w})
    return rows


def summarize(rows, Dex):
    ok_c = all(r['err_contam'] <= r['b_contam'] * (1 + 1e-6) + 1e-12 for r in rows)
    ok_t = all(r['err_target'] <= r['b_target'] * (1 + 1e-6) + 1e-12 for r in rows)
    fin = [r for r in rows[1:] if r['b_est'] < np.inf]
    cover = np.mean([r['err'] <= r['b_est'] for r in fin]) if fin else np.nan
    ratio = np.array([r['err'] / r['b_est'] for r in rows[1:] if 0 < r['b_est'] < np.inf])
    n_est = next((i for i, r in enumerate(rows) if r['b_est'] < 0.01 * abs(Dex)), None)
    n_true = next((i for i in range(len(rows)) if all(abs(g['D'] - Dex) < 0.01 * abs(Dex) for g in rows[i:])), None)
    n_chem = next((i for i, r in enumerate(rows) if r['dE'] < 1.6e-3), None)
    return {'rigorous_contam_holds': ok_c, 'rigorous_target_holds': ok_t, 'estimate_coverage': float(cover),
            'median_err_over_estimate': float(np.median(ratio)) if ratio.size else None,
            'max_err_over_estimate': float(ratio.max()) if ratio.size else None,
            'n_finite_estimate': int(ratio.size),
            'n_estimate_below_1pct': n_est, 'n_D_within_1pct_for_good': n_true, 'n_chem': n_chem,
            'estimate_at_chem_pct': (100 * rows[n_chem]['b_est'] / abs(Dex)) if n_chem is not None else None,
            'err_at_chem_pct': (100 * rows[n_chem]['err'] / abs(Dex)) if n_chem is not None else None}


def small():
    import scipy.sparse as sp
    import openfermion as of
    from theory_check import setup
    from fockmap import pyscf_to_fock, d_component_op
    from vqe import CISpace, pool_gsd, pool_sf, adapt_vqe
    out = {}
    for name in ['CH2', 'NH', 'NH_stretch', 'O2', 'TMM']:
        mol, mc, F, S, ncas, nelecas, h1, ecore, eri = setup(name)
        space = CISpace(ncas, nelecas); na, nb = len(space.sa), len(space.sb); dim = space.dim
        h2 = fci.direct_spin1.absorb_h1e(h1, eri, ncas, nelecas, 0.5)
        Hd = np.zeros((dim, dim)); S2m = np.zeros((dim, dim))
        for i in range(dim):
            e = np.zeros(dim); e[i] = 1
            Hd[:, i] = fci.direct_spin1.contract_2e(h2, e.reshape(na, nb), ncas, nelecas).ravel()
            S2m[:, i] = fci.spin_op.contract_ss(e.reshape(na, nb), ncas, nelecas).ravel()
        Hd = 0.5 * (Hd + Hd.T) + ecore * np.eye(dim); S2m = 0.5 * (S2m + S2m.T)
        _, d2 = fci.direct_spin1.make_rdm12s(mc.ci, ncas, nelecas)
        w, V = np.linalg.eigh(d_tensor(F, d2, S)); V = V[:, np.argsort(np.abs(w))]
        Fp = np.einsum('ai,bj,ab...->ij...', V, V, F)
        Xf = of.get_sparse_operator(d_component_op(Fp[2, 2], ncas, S), n_qubits=2 * ncas)
        A = np.array([pyscf_to_fock(np.eye(dim)[k].reshape(na, nb), ncas, nelecas) for k in range(dim)]).T
        Xd = (A.T @ (Xf @ A)).real; Xd = 0.5 * (Xd + Xd.T)
        ws, Vs = np.linalg.eigh(S2m); PTm = Vs[:, np.abs(ws - 2.0) < 1e-6]
        eT, cT = np.linalg.eigh(PTm.T @ Hd @ PTm); T = PTm @ cT[:, 0]; E0, E1 = eT[0], eT[1]
        cD = 1.5 * d_prefactor(S) * HARTREE2CM
        xT = T @ Xd @ T; sXT = np.sqrt(max(T @ Xd @ Xd @ T - xT ** 2, 0))
        Xshift = np.abs(np.linalg.eigvalsh(Xd - xT * np.eye(dim))).max()
        Xex = {'D': cD * xT, 'sigma': sXT}
        ref = np.zeros(dim); ref[space.ia[(1 << nelecas[0]) - 1] * nb + space.ib[(1 << nelecas[1]) - 1]] = 1
        out[name] = {'gap_mEh': 1e3 * (E1 - E0), 'D': cD * xT}
        for pname, pool in [('GSD', pool_gsd(space)), ('QEB', pool_gsd(space, qubit=True)), ('sfGSD', pool_sf(space))]:
            states = []
            adapt_vqe(space, Hd, ref, pool, observe=lambda psi: (states.append(psi.copy()), {})[1], max_ops=80,
                      e_exact=mc.e_tot)
            rows = analyse(states, lambda v: Xd @ v, lambda v: Hd @ v, lambda v: S2m @ v,
                           lambda v: PTm @ (PTm.T @ v), E0, E1, Xex, Xshift, cD)
            s = summarize(rows, cD * xT)
            out[name][pname] = s
            print(name, pname, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}, flush=True)
            json.dump(rows, open(f'results/errbar_{name}_{pname}.json', 'w'), default=float)
    json.dump(out, open('results/errbar_small.json', 'w'), indent=1, default=float)


def naph(orb):
    import scipy.sparse.linalg as sla
    import vqe_sparse as sps
    from xop import XOperator
    tag = '' if orb == 'rohf' else '_no'
    dat = np.load(f'results/naph_setup{tag}.npz')
    ncas, nelecas, S = 10, (6, 4), 1.0
    space = sps.SparseSpace(ncas, nelecas); shape = (space.na, space.nb)
    Hs = sps.Hamiltonian(dat['h1'], dat['eri'], float(dat['ecore']), ncas, nelecas)
    H = lambda v: Hs(v, shape)
    Fp = np.einsum('ai,bj,ab...->ij...', dat['V'], dat['V'], dat['F'])
    X = XOperator(ncas, nelecas, Fp[2, 2])
    S2 = lambda v: fci.spin_op.contract_ss(v.reshape(shape), ncas, nelecas).ravel()

    def PT(x):
        y = x.copy()
        for s in (2, 3, 4, 5):
            c = s * (s + 1); y = (S2(y) - c * y) / (2.0 - c)
        return y
    T = dat['ci'].ravel(); E0, E1 = float(dat['e_casci']), float(dat['e1'])
    cD = 1.5 * d_prefactor(S) * HARTREE2CM
    XT = X(T); xT = T @ XT; sXT = np.sqrt(max(XT @ XT - xT ** 2, 0))
    L = sla.LinearOperator((space.dim, space.dim), matvec=lambda v: X(v) - xT * v, dtype=float)
    Xshift = abs(sla.eigsh(L, k=1, which='LM', tol=1e-6, return_eigenvectors=False)[0])
    Xex = {'D': cD * xT, 'sigma': sXT}
    out = {'gap_mEh': 1e3 * (E1 - E0), 'D': cD * xT}
    for pname in ['GSD', 'QEB', 'sfGSD']:
        try:
            states = np.load(f'results/naph{tag}_{pname}_states.npy')
        except FileNotFoundError:
            continue
        rows = analyse(states, X, H, S2, PT, E0, E1, Xex, Xshift, cD)
        s = summarize(rows, cD * xT)
        out[pname] = s
        print('naph' + tag, pname, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}, flush=True)
        json.dump(rows, open(f'results/errbar_naph{tag}_{pname}.json', 'w'), default=float)
    json.dump(out, open(f'results/errbar_naph{tag}.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    small() if sys.argv[1] == 'small' else naph(sys.argv[2] if len(sys.argv) > 2 else 'no')
