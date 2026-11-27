"""Remedies for the target-spin error in D, tested on the 12-qubit systems (exact answer known).

  A  standard ADAPT (energy-gradient selection)                                      baseline
  B  D-aware ADAPT: select argmax_k |g_k|/max|g| + x_k/max x, with g_k = <[H,A_k]> and x_k the Frobenius norm of
     the gradient of the D tensor, x_k = || <[X_ab, A_k]> ||_F (frame invariant, uses no exact information);
     parameters are still optimized by minimizing the energy. Weight 1 fixed in advance.
  B2 post hoc variant of B (added after B failed on CH2): product score (|g_k|/max|g|)(x_k/max x), which
     never selects an operator with vanishing energy gradient
  C1 one Lanczos step: lowest state in span{psi, H psi}; D evaluated there
  C2 quantum subspace expansion with spin-free singles: lowest state in span{psi, E_pq psi}
  D  variance extrapolation: linear fit of D_k against sigma_H,k^2 over the last M iterates, D* = intercept

Pre-registered criterion: a remedy passes if D is within 1% at chemical accuracy in >= 3 of the 4 systems
(CH2, NH, stretched NH, O2) and it needs no more CNOTs than A to bring D within 1% for good.
Writes results/remedies.json and results/remedies_<system>_<pool>_<method>.json.
Usage (from the repository root): python scripts/remedies.py [systems...]
"""
import json
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import scipy.linalg as sla
import openfermion as of
from pyscf import fci
from theory_check import setup
from fockmap import pyscf_to_fock, d_component_op, d_prefactor
from zfs import d_tensor, HARTREE2CM
from vqe import CISpace, pool_gsd, pool_sf, adapt_vqe
from cnot_counts import cnots

CHEM = 1.6e-3
M_EXTRAP = 5
COMPS = [(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (0, 1, 2.0), (0, 2, 2.0), (1, 2, 2.0)]   # Frobenius weights


def build(name):
    mol, mc, F, S, ncas, nelecas, h1, ecore, eri = setup(name)
    space = CISpace(ncas, nelecas); na, nb = len(space.sa), len(space.sb); dim = space.dim
    h2 = fci.direct_spin1.absorb_h1e(h1, eri, ncas, nelecas, 0.5)
    Hd = np.zeros((dim, dim))
    for i in range(dim):
        e = np.zeros(dim); e[i] = 1
        Hd[:, i] = fci.direct_spin1.contract_2e(h2, e.reshape(na, nb), ncas, nelecas).ravel()
    Hd = 0.5 * (Hd + Hd.T) + ecore * np.eye(dim)
    A = np.array([pyscf_to_fock(np.eye(dim)[k].reshape(na, nb), ncas, nelecas) for k in range(dim)]).T

    def to_det(fop):
        Xf = of.get_sparse_operator(fop, n_qubits=2 * ncas)
        Xd = (A.T @ (Xf @ A)).real
        return 0.5 * (Xd + Xd.T)
    Xc = {(a, b): to_det(d_component_op(F[a, b], ncas, S)) for a, b, _ in COMPS}
    _, d2 = fci.direct_spin1.make_rdm12s(mc.ci, ncas, nelecas)
    w, V = np.linalg.eigh(d_tensor(F, d2, S)); V = V[:, np.argsort(np.abs(w))]
    Fp = np.einsum('ai,bj,ab...->ij...', V, V, F)
    Xz = to_det(d_component_op(Fp[2, 2], ncas, S))
    cD = 1.5 * d_prefactor(S) * HARTREE2CM
    T = mc.ci.ravel()
    Dex = cD * (T @ Xz @ T)
    # spin-free singles E_pq (p != q and p == q) for QSE
    Epq = [space.E(p, q).toarray() for p in range(ncas) for q in range(ncas)]
    ref = np.zeros(dim); ref[space.ia[(1 << nelecas[0]) - 1] * nb + space.ib[(1 << nelecas[1]) - 1]] = 1
    return dict(space=space, H=Hd, Xc=Xc, Xz=Xz, cD=cD, Dex=Dex, E0=mc.e_tot, Epq=Epq, ref=ref,
                mol=mol, mc=mc, h1=h1, ecore=ecore, eri=eri, S=S, Fp=Fp, ncas=ncas, nelecas=nelecas)


def lowest_in_span(vecs, H, tol=1e-10):
    B = np.array(vecs).T
    S = B.T @ B
    s, U = np.linalg.eigh(S)
    keep = s > tol * s.max()
    Bo = B @ (U[:, keep] / np.sqrt(s[keep]))
    e, c = np.linalg.eigh(Bo.T @ H @ Bo)
    v = Bo @ c[:, 0]
    return v / np.linalg.norm(v)


def run(name, sysd, pname, pool, method):
    H, Xz, cD = sysd['H'], sysd['Xz'], sysd['cD']
    Xlist = [(sysd['Xc'][(a, b)], wgt) for a, b, wgt in COMPS]
    states = []

    def score(psi, grads, mats):
        XP = [(X @ psi, wgt) for X, wgt in Xlist]
        x = np.zeros(len(mats))
        Ap = [Am @ psi for Am in mats]
        for xp, wgt in XP:
            x += wgt * (2 * np.array([xp @ a for a in Ap])) ** 2
        x = np.sqrt(x)
        g = np.abs(grads)
        if method == 'B2':
            return (g / g.max()) * (x / max(x.max(), 1e-300))
        return g / g.max() + x / max(x.max(), 1e-300)

    t0 = time.time()
    hist, chosen, thetas, psi = adapt_vqe(sysd['space'], H, sysd['ref'], pool,
                                          observe=lambda p: (states.append(p.copy()), {})[1], max_ops=80,
                                          e_exact=sysd['E0'], score=score if method in ('B', 'B2') else None)
    qubit = pname == 'QEB'
    cn = np.cumsum([0] + [cnots(tuple(lab), qubit)[0] for lab in chosen])
    gch = [None] + [h.get('g_chosen') for h in hist[1:]]
    gmx = [None] + [h.get('gmax') for h in hist[1:]]
    rows = []
    for n, psi in enumerate(states):
        Hp = H @ psi; E = psi @ Hp; r = Hp - E * psi
        row = {'n': n, 'cnot': int(cn[n]), 'dE': float(E - sysd['E0']), 'sigma_H2': float(r @ r),
               'D': float(cD * psi @ Xz @ psi), 'g_chosen': gch[n] if n < len(gch) else None,
               'gmax': gmx[n] if n < len(gmx) else None}
        if method == 'A':
            v = lowest_in_span([psi, Hp], H)
            row['D_lanczos'] = float(cD * v @ Xz @ v); row['dE_lanczos'] = float(v @ H @ v - sysd['E0'])
            v = lowest_in_span([psi] + [Ep @ psi for Ep in sysd['Epq']], H)
            row['D_qse'] = float(cD * v @ Xz @ v); row['dE_qse'] = float(v @ H @ v - sysd['E0'])
        rows.append(row)
    if method == 'A':
        for i, row in enumerate(rows):
            if i >= M_EXTRAP - 1:
                seg = rows[i - M_EXTRAP + 1:i + 1]
                x = np.array([s['sigma_H2'] for s in seg]); y = np.array([s['D'] for s in seg])
                row['D_extrap'] = float(np.polyfit(x, y, 1)[1]) if np.ptp(x) > 0 else row['D']
            else:
                row['D_extrap'] = None
    json.dump({'rows': rows, 'ops': [list(c) for c in chosen], 'time_s': time.time() - t0},
              open(f'results/remedies_{name}_{pname}_{method}.json', 'w'))
    return rows


def metrics(rows, key, Dex):
    i_chem = next((i for i, r in enumerate(rows) if r['dE'] < CHEM), None)
    vals = [r.get(key) for r in rows]
    err = lambda v: None if v is None else 100 * (v - Dex) / abs(Dex)
    i_d1 = next((i for i in range(len(rows))
                 if all(v is not None and abs(v - Dex) < 0.01 * abs(Dex) for v in vals[i:])), None)
    return {'n_chem': i_chem, 'cnot_chem': rows[i_chem]['cnot'] if i_chem is not None else None,
            'err_chem_pct': err(vals[i_chem]) if i_chem is not None else None,
            'n_D1': i_d1, 'cnot_D1': rows[i_d1]['cnot'] if i_d1 is not None else None,
            'n_final': len(rows) - 1, 'dE_final': rows[-1]['dE']}


if __name__ == '__main__':
    systems = sys.argv[1:] or ['CH2', 'NH', 'NH_stretch', 'O2']
    try:
        out = json.load(open('results/remedies.json'))
    except FileNotFoundError:
        out = {}
    for name in systems:
        sysd = build(name)
        out[name] = {'D_exact': sysd['Dex']}
        for pname in ['GSD', 'sfGSD']:
            pool = pool_gsd(sysd['space']) if pname == 'GSD' else pool_sf(sysd['space'])
            rA = run(name, sysd, pname, pool, 'A')
            rB = run(name, sysd, pname, pool, 'B')
            rB2 = run(name, sysd, pname, pool, 'B2')
            res = {'A': metrics(rA, 'D', sysd['Dex']), 'B': metrics(rB, 'D', sysd['Dex']),
                   'B2': metrics(rB2, 'D', sysd['Dex']),
                   'C1': metrics(rA, 'D_lanczos', sysd['Dex']), 'C2': metrics(rA, 'D_qse', sysd['Dex']),
                   'D': metrics(rA, 'D_extrap', sysd['Dex'])}
            out[name][pname] = res
            for k, v in res.items():
                print(f"{name:10s} {pname:5s} {k:2s} chem: n={v['n_chem']} cnot={v['cnot_chem']} "
                      f"err={v['err_chem_pct'] if v['err_chem_pct'] is None else round(v['err_chem_pct'], 3)}% | "
                      f"D within 1%: n={v['n_D1']} cnot={v['cnot_D1']} | final n={v['n_final']} dE={v['dE_final']:.1e}",
                      flush=True)
            json.dump(out, open('results/remedies.json', 'w'), indent=1)
