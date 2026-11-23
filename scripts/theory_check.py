"""Check every inequality of the theory section against ADAPT-VQE trajectories (12-qubit systems and TMM).

Notation: psi = sqrt(w) T~ + sqrt(1-w) Q (target-spin part T~, remainder Q with S' >= S+1), T the exact target
state, the ground state of the M_S = S sector (E0); E1 the next state of spin S; G_M the first excitation energy
of the whole sector (any spin); F = |<T|psi>|^2; X the D operator (cm^-1), D(phi) = <phi|X|phi>.
At every iterate:
  split   exact identity, eq 4 (residual reported)
  eq 5    contamination bound  2 ||P_{S+1,S+2} X T~|| (delta/2(S+1))^1/2 + delta Delta_X / 2(S+1)
  eq 6    target-state bound   2 s sigma_X(T) + s^2 ||X - <X>_T||, with the exact s and with s from eq 7
          (Eckart: s^2 <= (E(T~) - E0)/(E1 - E0); variance: s <= sigma_H(T~)/(E1 - E(T~)) if E(T~) < E1)
  s_w     s^2 <= (E(psi) - E0) / (w (E1 - E0))     [E(Q) >= E0 because E0 is the sector ground state]
  eq 9   fidelity bound (split form and Cauchy-Schwarz form); eq 10, its a-priori form with 1 - F <= eps/G_M
At the first iterate with chemical accuracy, the Gaussian-filter guarantee (eq 11) for
g = exp[-(H' - E(psi))^2 T^2 / 2], H' = H + mu_f (S^2 - S(S+1)), T = 3/(E1 - E0), mu_f = 0.05 Eh.
Writes results/theory_check.json.  Usage (from the repository root): python scripts/theory_check.py [systems...]
"""
import json
import sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pyscf import gto, scf, mcscf, fci, ao2mo
from zfs import ss_mo_ints, d_tensor, HARTREE2CM
from fockmap import pyscf_to_fock, d_component_op, d_prefactor
import openfermion as of
from vqe import CISpace, pool_gsd, pool_sf, adapt_vqe
from run import SYSTEMS, select_pi_orbitals, axis_gauge


def setup(name):
    atom, ncas, nelecas, _ = SYSTEMS[name]
    mol = gto.M(atom=atom, basis='cc-pvdz', spin=nelecas[0] - nelecas[1], verbose=0)
    mf = scf.ROHF(mol).run(conv_tol=1e-11)
    S = 0.5 * (nelecas[0] - nelecas[1])
    mc = mcscf.CASCI(mf, ncas, nelecas)
    if name == 'TMM':
        mc.mo_coeff = mc.sort_mo([i + 1 for i in select_pi_orbitals(mol, mf, ncas, nelecas)])
    else:
        mc.mo_coeff = axis_gauge(mol, mf, mc.ncore, ncas)
    mc.fcisolver.conv_tol = 1e-12
    mc.fix_spin_(ss=S * (S + 1)); mc.kernel()
    F = ss_mo_ints(mol, mc.mo_coeff[:, mc.ncore:mc.ncore + ncas])
    h1, ecore = mc.get_h1eff()
    eri = ao2mo.restore(1, mc.get_h2eff(), ncas)
    return mol, mc, F, S, ncas, nelecas, h1, ecore, eri


def run(name, pools=('GSD', 'QEB', 'sfGSD'), mu_f=0.05):
    mol, mc, F, S, ncas, nelecas, h1, ecore, eri = setup(name)
    space = CISpace(ncas, nelecas); na, nb = len(space.sa), len(space.sb); dim = space.dim
    h2 = fci.direct_spin1.absorb_h1e(h1, eri, ncas, nelecas, 0.5)
    Hd = np.zeros((dim, dim)); S2m = np.zeros((dim, dim))
    for i in range(dim):
        e = np.zeros(dim); e[i] = 1
        Hd[:, i] = fci.direct_spin1.contract_2e(h2, e.reshape(na, nb), ncas, nelecas).ravel()
        S2m[:, i] = fci.spin_op.contract_ss(e.reshape(na, nb), ncas, nelecas).ravel()
    Hd = 0.5 * (Hd + Hd.T) + ecore * np.eye(dim); S2m = 0.5 * (S2m + S2m.T)
    # D operator (principal frame of exact D) in the determinant basis, in cm^-1
    _, d2 = fci.direct_spin1.make_rdm12s(mc.ci, ncas, nelecas)
    w, V = np.linalg.eigh(d_tensor(F, d2, S)); V = V[:, np.argsort(np.abs(w))]
    Fp = np.einsum('ai,bj,ab...->ij...', V, V, F)
    Xf = of.get_sparse_operator(d_component_op(Fp[2, 2], ncas, S), n_qubits=2 * ncas)
    A = np.array([pyscf_to_fock(np.eye(dim)[k].reshape(na, nb), ncas, nelecas) for k in range(dim)]).T
    X = (A.T @ (Xf @ A)).real * 1.5 * d_prefactor(S) * HARTREE2CM
    X = 0.5 * (X + X.T)
    # spin sectors, exact spectrum of the M_S = S sector
    ws, Vs = np.linalg.eigh(S2m)
    ss = lambda s: Vs[:, np.abs(ws - s * (s + 1)) < 1e-6]
    PT = ss(S); P12 = np.hstack([ss(S + 1), ss(S + 2)])
    E, U = np.linalg.eigh(Hd); s2E = np.einsum('ij,ik,kj->j', U, S2m, U)
    assert abs(s2E[0] - S * (S + 1)) < 1e-6, 'target is not the ground state of the sector'
    T = U[:, 0]; E0 = E[0]; G_M = E[1] - E0
    trip = np.abs(s2E - S * (S + 1)) < 1e-6; E1 = E[trip][1]
    Ep = E + mu_f * (s2E - S * (S + 1)); Gp = np.sort(Ep)[1] - Ep[0]
    Dex = T @ X @ T
    I = np.eye(dim); r = (X - Dex * I) @ T
    sigX = np.linalg.norm(r); sig_par = np.linalg.norm(PT @ (PT.T @ r)); sig_perp = np.sqrt(max(sigX**2 - sig_par**2, 0.0))
    normX = np.max(np.abs(np.linalg.eigvalsh(X - Dex * I)))          # ||X - <X>_T||
    out = {'name': name, 'D': Dex, 'gap_mEh': 1e3 * (E1 - E0), 'G_M_mEh': 1e3 * G_M, 'Gp_mEh': 1e3 * Gp,
           'sigmaX_over_D': sigX / abs(Dex), 'sigma_par_over_D': sig_par / abs(Dex), 'sigma_perp_over_D': sig_perp / abs(Dex),
           'normX_over_D': normX / abs(Dex)}
    ref = np.zeros(dim); ref[space.ia[(1 << nelecas[0]) - 1] * nb + space.ib[(1 << nelecas[1]) - 1]] = 1
    pools_d = {'GSD': pool_gsd(space), 'QEB': pool_gsd(space, qubit=True), 'sfGSD': pool_sf(space)}
    chem_states = {}
    for p in pools:
        st = {}

        def obs(psi, st=st):
            psi = psi / np.linalg.norm(psi)
            pt = PT @ (PT.T @ psi); wt = float(pt @ pt); Tt = pt / np.sqrt(wt)
            q = psi - pt; nq = np.linalg.norm(q)
            Dn, Dt = psi @ X @ psi, Tt @ X @ Tt
            delta = max(psi @ S2m @ psi - S * (S + 1), 0.0); u = delta / (2 * (S + 1))
            eps = psi @ Hd @ psi - E0; Fid = float((T @ psi) ** 2); ET = Tt @ Hd @ Tt
            if 'psi' not in st and eps < 1.6e-3:
                st['psi'] = psi.copy()
            rec = {'eps_mEh': 1e3 * eps, 'w': wt, 'F': Fid, 'err': abs(Dn - Dex)}
            # eq 4 (exact split) and eq 5 (contamination bound)
            if nq > 1e-12:
                Q = q / nq
                rec['split_residual'] = abs((Dn - Dt) - (2 * np.sqrt(wt * (1 - wt)) * (Tt @ X @ Q) + (1 - wt) * (Q @ X @ Q - Dt)))
                rec['contam'] = abs(Dn - Dt)
                rec['b5'] = 2 * np.linalg.norm(P12.T @ (X @ Tt)) * np.sqrt(u) + u * abs(Q @ X @ Q - Dt)
            else:
                rec['split_residual'] = 0.0; rec['contam'] = abs(Dn - Dt); rec['b5'] = 0.0
            # eq 6 with exact s and with eq 7 (Eckart, variance)
            c = min(abs(T @ Tt), 1.0); s = np.sqrt(max(1 - c * c, 0.0))
            b6 = lambda sv: 2 * sv * sigX + sv * sv * normX
            s_eck = np.sqrt(max(ET - E0, 0.0) / (E1 - E0))
            sigH = np.sqrt(max(Tt @ Hd @ Hd @ Tt - ET**2, 0.0))
            s_var = sigH / (E1 - ET) if ET < E1 else np.inf
            rec.update(target=abs(Dt - Dex), b6_exact=b6(s), b6_eckart=b6(min(s_eck, 1.0)), b6_var=b6(min(s_var, 1.0)),
                       s=s, s_w_bound=np.sqrt(max(eps, 0.0) / (wt * (E1 - E0))))
            # eq 9: fidelity bound (split and Cauchy-Schwarz forms); eq 10: a-priori form (sector Eckart)
            rec['b11_split'] = 2 * np.sqrt(Fid) * (sig_par * np.sqrt(max(wt - Fid, 0.0)) + sig_perp * np.sqrt(max(1 - wt, 0.0))) + (1 - Fid) * normX
            rec['b11'] = 2 * np.sqrt(Fid * (1 - Fid)) * sigX + (1 - Fid) * normX
            ua = min(max(eps, 0.0) / G_M, 1.0); rec['eckart_M'] = ua
            rec['b11_prior'] = 2 * np.sqrt(ua * (1 - ua)) * sigX + ua * normX
            return rec
        hist, *_ = adapt_vqe(space, Hd, ref, pools_d[p], observe=obs, max_ops=80, e_exact=E0)
        H = hist[1:]
        def maxratio(a, b):
            v = [h[a] / h[b] for h in H if h[b] > 1e-14]
            return max(v) if v else 0.0
        # violations, with tolerances at the level of floating-point round-off: 1e-9 |D| for D differences and
        # 1e-13 for squared overlaps (s and 1 - F come from 1 - |overlap|^2 and carry ~1e-16 round-off)
        # Bounds built from energy differences (eq 7, the s_w bound, the a-priori form) are meaningless once the
        # energy error is below the round-off of the energies themselves (~1e-12 Eh); such converged iterates are
        # excluded from those checks and counted in 'n_converged'.
        # Result keys keep an earlier equation numbering: 'eq11_split'/'eq11' = fidelity bound (manuscript eq 9),
        # 'eq11_prior' = its a-priori form (eq 10); 'eq5', 'eq6_*' match eqs 5 and 6.
        tol = 1e-9 * max(abs(Dex), 1.0)
        EN = {'eq6_eckart', 'eq6_var', 'eq11_prior'}
        Hc = [h for h in H if h['eps_mEh'] > 1e-9]
        pairs_D = {'eq5': ('contam', 'b5'), 'eq6_exact': ('target', 'b6_exact'), 'eq6_eckart': ('target', 'b6_eckart'),
                   'eq6_var': ('target', 'b6_var'), 'eq11_split': ('err', 'b11_split'), 'eq11': ('err', 'b11'),
                   'eq11_prior': ('err', 'b11_prior')}
        viol, worst = {}, {}
        for k, (a, b) in pairs_D.items():
            bad = [h for h in (Hc if k in EN else H) if h[a] > h[b] + tol]
            viol[k] = len(bad)
            if bad:
                worst[k] = {'eps_mEh': bad[0]['eps_mEh'], 'excess': bad[0][a] - bad[0][b]}
        viol['s_w'] = sum(1 for h in Hc if h['s'] ** 2 > h['s_w_bound'] ** 2 + 1e-13)
        viol['eckart_M'] = sum(1 for h in H if 1 - h['F'] > h['eckart_M'] + 1e-13)
        res = {'n': len(H), 'n_converged': len(H) - len(Hc), 'violations': viol, 'worst_violation': worst, 'max_split_residual': max(h['split_residual'] for h in H),
               'max_ratio': {k: maxratio(a, b) for k, (a, b) in {'eq5': ('contam', 'b5'), 'eq6_exact': ('target', 'b6_exact'),
                             'eq6_var': ('target', 'b6_var'), 'eq11_split': ('err', 'b11_split'), 'eq11': ('err', 'b11')}.items()}}
        # first iterate with chemical accuracy: bounds there and the Gaussian-filter guarantee (eq 11)
        ic = next((i for i, h in enumerate(hist) if h['eps_mEh'] < 1.6), None)
        if ic is not None:
            h = hist[ic]
            res['chem'] = {k: h[k] for k in ('eps_mEh', 'F', 'w', 'err', 'b11_split', 'b11', 'b11_prior')}
            res['chem'].update(n=ic)
        out[p] = res
        chem_states[p] = st.get('psi')
        print(f"{name:10s} {p:5s} n={len(H):2d} (converged {len(H) - len(Hc)}) violations {viol}  max split residual {res['max_split_residual']:.1e}  "
              f"max ratios " + ' '.join(f"{k} {v:.2f}" for k, v in res['max_ratio'].items()), flush=True)
    # Gaussian-filter guarantee on the chemical-accuracy states of the three pools
    Tr = 3.0 / (E1 - E0)
    for p in pools:
        psi = chem_states.get(p)
        if psi is None:
            continue
        eps = psi @ Hd @ psi - E0; Fid = (T @ psi) ** 2
        cpsi = U.T @ psi; g = np.exp(-0.5 * ((Ep - (psi @ Hd @ psi)) * Tr) ** 2); y = U @ (g * cpsi); y /= np.linalg.norm(y)
        Fg = (T @ y) ** 2
        ratio_bound = (1 - Fid) / Fid * np.exp(-Tr**2 * Gp * (Gp - 2 * eps)); Fg_b = 1 / (1 + ratio_bound)
        bG = 2 * np.sqrt(Fg_b * (1 - Fg_b)) * sigX + (1 - Fg_b) * normX
        out[p]['filter'] = {'T': Tr, 'infid': 1 - Fid, 'infid_g': 1 - Fg, 'infid_g_bound': 1 - Fg_b,
                            'err_pct': 100 * abs(y @ X @ y - Dex) / abs(Dex), 'bound_pct': 100 * bG / abs(Dex),
                            'ok': bool(1 - Fg <= (1 - Fg_b) + 1e-13 and abs(y @ X @ y - Dex) <= bG + 1e-9 * abs(Dex))}
        f = out[p]['filter']
        print(f"{name:10s} {p:5s} filter: 1-F {f['infid']:.1e} -> {f['infid_g']:.1e} (bound {f['infid_g_bound']:.1e}); "
              f"error {f['err_pct']:.3f}% <= {f['bound_pct']:.2f}%  {'OK' if f['ok'] else 'VIOLATION'}", flush=True)
    print(f"{name:10s} E1-E0 {out['gap_mEh']:.1f}  G_M {out['G_M_mEh']:.1f}  G' {out['Gp_mEh']:.1f} mEh  sigma_X/|D| {out['sigmaX_over_D']:.2f} "
          f"(par {out['sigma_par_over_D']:.2f}, perp {out['sigma_perp_over_D']:.2f})  ||X-D||/|D| {out['normX_over_D']:.1f}", flush=True)
    return out


if __name__ == '__main__':
    res = [run(s) for s in (sys.argv[1:] or ['CH2', 'NH', 'NH_stretch', 'O2', 'TMM'])]
    json.dump(res, open('results/theory_check.json', 'w'), indent=1, default=float)
