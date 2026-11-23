"""Adversarial stress test of every statement in Section 2 of the manuscript.

Part A  General-S check of eq 2 (D from the M_S = S component): the full first-order H_SS in the (2S+1)-fold
        manifold, built from explicit spin matrices, versus S.D.S, for S = 3/2 (CH2+) and S = 2 (quintet CH2).
        The paper's systems are all triplets, so the S-dependence of 3/(S(2S-1)) is otherwise untested.
Part B  Eqs 4-13 on molecular Hamiltonians (CH2, NH, NH_stretch, O2) with adversarial and random states,
        complex phases, large and tiny errors, arbitrary filter centres, gap bounds below the true gap,
        the pre-filter composition, and the simulation-error item (i) of the error budget.
Part C  The same checks on synthetic spin-adapted Hamiltonians with hostile spectra (tiny gaps, a wrong-spin
        state just above E0, clustered excited states), keeping the molecular S^2 and X operators.
Writes results/sec2_stress.json.  Run from the repository root (see README).
"""
import json, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import openfermion as of
from pyscf import gto, scf, mcscf, fci, ao2mo
from zfs import d_tensor, ss_mo_ints, HARTREE2CM
from fockmap import pyscf_to_fock, d_component_op, d_prefactor
from theory_check import setup
from vqe import CISpace
from stress_test import hss_op
from filter_guarantee_lib import cert_inputs, cert_bound_pct

rng = np.random.default_rng(2026)


# ----------------------------------------------------------------------------------------------- Part A
def spin_mats(S):
    m = np.arange(S, -S - 1, -1)
    Sz = np.diag(m)
    Sp = np.zeros((len(m), len(m)))
    for k in range(1, len(m)):
        Sp[k - 1, k] = np.sqrt(S * (S + 1) - m[k] * (m[k] + 1))
    Sm = Sp.T
    return [(Sp + Sm) / 2, (Sp - Sm) / 2j, Sz]


def part_a(label, charge, nelecas, ncas=6):
    atom = 'C 0 0 0; H 0 0.98963 -0.42083; H 0 -0.98963 -0.42083'
    S = 0.5 * (nelecas[0] - nelecas[1])
    mol = gto.M(atom=atom, basis='cc-pvdz', charge=charge, spin=int(2 * S), verbose=0)
    mf = scf.ROHF(mol).run(conv_tol=1e-11)
    mc = mcscf.CASCI(mf, ncas, nelecas); mc.fcisolver.conv_tol = 1e-12
    mc.fix_spin_(ss=S * (S + 1)); mc.kernel()
    F = ss_mo_ints(mol, mc.mo_coeff[:, mc.ncore:mc.ncore + ncas])
    _, d2 = fci.direct_spin1.make_rdm12s(mc.ci, ncas, nelecas)
    D = d_tensor(F, d2, S)
    nq = 2 * ncas
    Smin = of.get_sparse_operator(of.hermitian_conjugated(of.s_plus_operator(ncas)), n_qubits=nq)
    S2 = of.get_sparse_operator(of.s_squared_operator(ncas), n_qubits=nq)
    v = pyscf_to_fock(mc.ci, ncas, nelecas).astype(complex)
    basis = [v]
    for _ in range(int(round(2 * S))):
        v = Smin @ v; v = v / np.linalg.norm(v); basis.append(v)
    B = np.array(basis).T
    s2 = np.real(np.vdot(B[:, 0], S2 @ B[:, 0]))
    Hop, scale = hss_op(F, ncas)
    Hs = of.get_sparse_operator(Hop, n_qubits=nq) * scale
    Hm = B.conj().T @ (Hs @ B)
    n = B.shape[1]
    Hm_tl = Hm - np.trace(Hm) / n * np.eye(n)
    Sv = spin_mats(S)
    Heff = sum(D[a, b] * Sv[a] @ Sv[b] for a in range(3) for b in range(3))
    Heff = Heff - np.trace(Heff) / n * np.eye(n)
    err = np.abs(Hm_tl - Heff).max() / np.abs(Heff).max()
    ev_full = np.linalg.eigvalsh(Hm_tl) * HARTREE2CM; ev_eff = np.linalg.eigvalsh(Heff) * HARTREE2CM
    w = np.linalg.eigvalsh(D); Dax = 1.5 * w[np.argmax(np.abs(w))] * HARTREE2CM
    r = dict(S=S, S2=s2, D_cm=Dax, rel_dev=float(err), sublevels_full=ev_full.round(6).tolist(),
             sublevels_SDS=ev_eff.round(6).tolist())
    print(f"A {label}: S={S} <S2>={s2:.8f} D={Dax:+.5f} cm-1  max|H_SS - S.D.S|/max = {err:.1e}")
    return r


# ----------------------------------------------------------------------------------------------- Part B / C
def molecular(name, mu_f=0.05):
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
    Xop = d_component_op(Fp[2, 2], ncas, S)
    Xf = of.get_sparse_operator(Xop, n_qubits=2 * ncas)
    A = np.array([pyscf_to_fock(np.eye(dim)[k].reshape(na, nb), ncas, nelecas) for k in range(dim)]).T
    cD = 1.5 * d_prefactor(S) * HARTREE2CM                      # D = cD <X>, in cm-1
    X = (A.T @ (Xf @ A)).real * cD; X = 0.5 * (X + X.T)
    q = of.jordan_wigner(Xop)
    W1 = 2 * cD * sum(abs(c) for t, c in q.terms.items() if t != ())
    return dict(name=name, S=S, H=Hd, S2=S2m, X=X, W1=W1, mu_f=mu_f)


def synthetic(base, kind, seed):
    """Spin-adapted H with a hostile spectrum: [H, S^2] = 0, target = lowest spin-S state = sector ground state."""
    r = np.random.default_rng(seed)
    S = base['S']; ws, Vs = np.linalg.eigh(base['S2'])
    H = np.zeros_like(base['H'])
    spins = sorted(set(np.round((-1 + np.sqrt(1 + 4 * np.maximum(ws, 0))) / 2 * 2) / 2))
    for sp in spins:
        P = Vs[:, np.abs(ws - sp * (sp + 1)) < 1e-6]
        k = P.shape[1]
        if k == 0:
            continue
        U, _ = np.linalg.qr(r.standard_normal((k, k)))
        if kind == 'tiny_gap':           # spin-S gap 0.5 mEh, other spins far
            ev = np.sort(r.uniform(0.02, 1.0, k)) if sp != S else np.r_[0.0, 5e-4, np.sort(r.uniform(5e-4, 1.0, k - 2))]
        elif kind == 'wrong_spin_close':  # a spin S+1 state 0.2 mEh above E0, spin-S gap 50 mEh
            ev = (np.r_[2e-4, np.sort(r.uniform(2e-4, 1.0, k - 1))] if sp == S + 1 else
                  np.r_[0.0, 0.05, np.sort(r.uniform(0.05, 1.0, k - 2))] if sp == S else np.sort(r.uniform(0.01, 1.0, k)))
        else:                             # clustered: many states just above the gap
            ev = (np.r_[0.0, 0.01 + 1e-5 * np.arange(k - 1)] if sp == S else 0.01 + r.uniform(0, 1e-3, k))
        H += P @ U @ np.diag(ev) @ U.T @ P.T
    H = 0.5 * (H + H.T)
    out = dict(base); out.update(name=f"{base['name']}_{kind}", H=H)
    return out


def g_apply(Ep, U, c, T, v):
    return U @ (np.exp(-T ** 2 * (Ep - c) ** 2 / 2) * (U.conj().T @ v))


def run_system(sysd, n_states=400, label=''):
    H, S2, X, S = sysd['H'], sysd['S2'], sysd['X'], sysd['S']
    dim = H.shape[0]; mu_f = sysd['mu_f']; Ss = S * (S + 1)
    ws, Vs = np.linalg.eigh(S2)
    PS = Vs[:, np.abs(ws - Ss) < 1e-6]
    P12 = np.hstack([Vs[:, np.abs(ws - (S + 1) * (S + 2)) < 1e-6], Vs[:, np.abs(ws - (S + 2) * (S + 3)) < 1e-6]])
    H = H - np.linalg.eigvalsh(H)[0] * np.eye(H.shape[0])     # energies relative to E0: no cancellation in sigma
    E, U = np.linalg.eigh(H); s2E = np.einsum('ij,ik,kj->j', U, S2, U)
    assert abs(s2E[0] - Ss) < 1e-6
    T = U[:, 0]; E0 = E[0]; GM = E[1] - E0
    E1 = E[np.abs(s2E - Ss) < 1e-6][1]
    Hp = H + mu_f * (S2 - Ss * np.eye(dim)); Ep, Up = np.linalg.eigh(Hp)
    assert abs(Ep[0] - E0) < 1e-10 and abs(abs(Up[:, 0] @ T) - 1) < 1e-8
    Gp = Ep[1] - Ep[0]
    xev = np.linalg.eigvalsh(X); W = xev[-1] - xev[0]; W1 = sysd['W1']
    Dex = T @ X @ T; I = np.eye(dim); r = (X - Dex * I) @ T
    sigX = np.linalg.norm(r); sig_par = np.linalg.norm(PS.T @ r); sig_perp = np.linalg.norm(r - PS @ (PS.T @ r))
    normX = np.max(np.abs(np.linalg.eigvalsh(X - Dex * I)))
    tolD = 1e-9 * max(abs(Dex), W); tolp = 1e-12
    worst = {}; nchk = {}; viol = {}; vdet = {}; ctx = {}

    def chk(key, actual, bound, tol):
        nchk[key] = nchk.get(key, 0) + 1
        if bound > 0 and np.isfinite(bound):
            worst[key] = max(worst.get(key, 0.0), actual / bound)
        if actual > bound + tol:
            viol[key] = viol.get(key, 0) + 1
            d = vdet.setdefault(key, [0.0, 0.0, None])
            if actual - bound > d[0]:
                vdet[key] = [float(actual - bound), float(actual / bound) if bound > 0 else np.inf, dict(ctx)]

    # adversarial directions
    dR = PS @ (PS.T @ r); dR -= (T @ dR) * T; dR /= np.linalg.norm(dR)
    dQ = r - PS @ (PS.T @ r); dQ /= np.linalg.norm(dQ)
    Xs = [Up[:, k] for k in range(1, min(dim, 12))]
    prefilter_sigma_up = 0; prefilter_cases = 0; e_increase = 0; e_cases = 0
    for it in range(n_states):
        fam = it % 5
        a = 10 ** rng.uniform(-4, 0.3)
        if fam == 0:      # random complex perturbation
            z = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        elif fam == 1:    # adversarial: in-spin along (X - <X>)T and contaminant along (1-P_S) X T, random phases
            z = np.exp(1j * rng.uniform(0, 2 * np.pi)) * dR * rng.uniform(0.2, 1) + \
                np.exp(1j * rng.uniform(0, 2 * np.pi)) * dQ * rng.uniform(0.2, 1)
        elif fam == 2:    # low-lying H' eigenstates (filter-relevant)
            z = sum((rng.standard_normal() + 1j * rng.standard_normal()) * v for v in Xs[:rng.integers(1, len(Xs))])
        elif fam == 3:    # pure contamination
            z = (Vs[:, np.abs(ws - Ss) > 1e-6] @ rng.standard_normal(dim - PS.shape[1])).astype(complex)
        else:             # extreme eigenvectors of X (worst for D) mixed in
            xe, xv = np.linalg.eigh(X); z = (xv[:, 0] * rng.standard_normal() + xv[:, -1] * rng.standard_normal()).astype(complex)
        z = z / np.linalg.norm(z)
        ctx = {'family': int(fam), 'a': float(a)}
        psi = T * np.exp(1j * rng.uniform(0, 2 * np.pi)) + a * z; psi /= np.linalg.norm(psi)
        ex = lambda O, v: np.real(np.vdot(v, O @ v))
        sd = lambda O, v: np.linalg.norm(O @ v - ex(O, v) * v)
        pt = PS @ (PS.T @ psi); w = np.real(np.vdot(pt, pt)); Tt = pt / np.sqrt(w)
        qv = psi - pt; nq = np.linalg.norm(qv)
        Dn, Dt = ex(X, psi), ex(X, Tt)
        delta = ex(S2, psi) - Ss; Eps = ex(H, psi); eps = Eps - E0
        F = abs(np.vdot(T, psi)) ** 2
        ctx.update(eps=float(eps), F=float(F), w=float(w))
        # eq 4 identity (with the phase convention: sqrt(w)Tt + sqrt(1-w)Q exactly)
        if nq > 1e-12:
            Q = qv / nq
            rhs = 2 * np.sqrt(w * (1 - w)) * np.real(np.vdot(Tt, X @ Q)) + (1 - w) * (ex(X, Q) - Dt)
            chk('eq4_identity', abs((Dn - Dt) - rhs), 0.0, tolD)
            chk('delta_ge_2(S+1)(1-w)', 2 * (S + 1) * (1 - w), delta, 1e-12)
            u = delta / (2 * (S + 1))
            cross = np.linalg.norm(P12.T @ (X @ Tt))
            chk('eq5_bound', abs(Dn - Dt), 2 * cross * np.sqrt(u) + u * abs(ex(X, Q) - Dt), tolD)
            chk('eq5_bound_2normX', abs(Dn - Dt), 2 * cross * np.sqrt(u) + u * 2 * np.max(np.abs(xev)), tolD)
            sTt = sd(X, Tt)
            chk('eq5_factor_le_sigma(Tt)', cross, sTt, 1e-12 * W)
            chk('energy_no_cross_term', abs((Eps - ex(H, Tt)) - (1 - w) * (ex(H, Q) - ex(H, Tt))), 0.0, 1e-10)
            chk('E(Tt)-E0<=eps/w', ex(H, Tt) - E0, eps / w, 1e-12)
        # eq 6, eq 7
        c = abs(np.vdot(T, Tt)); s = np.sqrt(max(1 - c * c, 0))
        chk('eq6', abs(Dt - Dex), 2 * s * sigX + s * s * normX, tolD)
        ETt = ex(H, Tt)
        if ETt - E0 > 1e-11:
            chk('eq7_eckart', s * s, (ETt - E0) / (E1 - E0), tolp)
            sH = sd(H, Tt)
            if ETt < E1:
                chk('eq7_variance', s, sH / (E1 - ETt), 1e-7)
        # eq 9 (both the split and the Cauchy-Schwarz/W forms)
        b9 = 2 * np.sqrt(F) * (sig_par * np.sqrt(max(w - F, 0)) + sig_perp * np.sqrt(max(1 - w, 0))) + (1 - F) * normX
        chk('eq9', abs(Dn - Dex), b9, tolD)
        chk('eq9_W_form', abs(Dn - Dex), W * (np.sqrt(F * (1 - F)) + (1 - F)), tolD)
        chk('sigmaX(T)<=W/2', sigX, W / 2, 1e-12 * W)
        # eq 10
        sigH = sd(H, psi)
        if eps > 1e-11:
            chk('eq10_eckart', 1 - F, eps / GM, tolp)
            if Eps < E0 + GM:
                chk('eq10_variance', 1 - F, sigH ** 2 / (E0 + GM - Eps) ** 2, 1e-7)
        # eq 11: filter at the VQE energy and at arbitrary centres
        for cen in (Eps, ex(Hp, psi), E0 + rng.uniform(-0.45, 0.45) * Gp):
            ec = abs(cen - E0)
            if not ec < Gp / 2 or F < 1e-12:
                continue
            for Tg in (0.5 / Gp, 1.5 / Gp, 3 / Gp, 6 / Gp):
                ph = g_apply(Ep, Up, cen, Tg, psi); Z = np.real(np.vdot(ph, ph)); ph = ph / np.sqrt(Z)
                Fg = abs(np.vdot(T, ph)) ** 2
                if Fg < 1 - 1e-13 and F < 1 - 1e-13:
                    chk('eq11', (1 - Fg) / Fg, (1 - F) / F * np.exp(-Tg ** 2 * Gp * (Gp - 2 * ec)), 1e-12)
        # eq 12-13: measured-input certificate, H or H' moments, true gap and gap bounds below it
        for Hm, gap_true in ((H, GM), (Hp, Gp)):
            Em = ex(Hm, psi); sm = sd(Hm, psi); epsm = Em - E0
            for sc in (1.0, 0.9, 0.5):
                Dl, Dlp = sc * gap_true, sc * Gp
                if not sm < Dl / 2:
                    continue
                eb = Dl / 2 - np.sqrt(Dl ** 2 / 4 - sm ** 2); ub = min(eb / Dl, sm ** 2 / (Dl - eb) ** 2)
                ci = cert_inputs(sm, Dl)
                if ci is not None:
                    chk('lib_matches_independent', abs(ci[0] - eb) + abs(ci[1] - ub), 0.0, 1e-15)
                if epsm < Dl / 2:          # hypothesis (H2)
                    chk('eq12_temple', epsm, eb, 1e-12)
                    chk('eq12_fidelity', 1 - F, ub, 1e-10)
                    if ub < 1 and eb < Dlp / 2:
                        for cen in (Eps, Em):
                            if abs(cen - E0) > eb + 1e-12:
                                continue
                            for Tg in (1 / Gp, 3 / Gp, 6 / Gp):
                                ph = g_apply(Ep, Up, cen, Tg, psi); ph /= np.linalg.norm(ph)
                                Dg = ex(X, ph)
                                qq = ub / (1 - ub) * np.exp(-Tg ** 2 * Dlp * (Dlp - 2 * eb)); fb = qq / (1 + qq)
                                for Wx, key in ((W, 'eq13_exactW'), (W1, 'eq13_W1')):
                                    chk(key, abs(Dg - Dex), Wx * (np.sqrt(fb * (1 - fb)) + fb), tolD)
                                chk('lib_bound_matches', abs(cert_bound_pct(Tg, eb, ub, Dlp, W / abs(Dex)) / 100 * abs(Dex)
                                                            - W * (np.sqrt(fb * (1 - fb)) + fb)), 0.0, 1e-12 * W)
        # pre-filter: composition, sigma and energy after a short pre-filter, certificate from pre-filtered moments
        if eps < Gp / 2 and F > 1e-6:
            T1, T2 = 0.25 / Gp, 3.0 / Gp
            a1 = g_apply(Ep, Up, Eps, T1, psi); a12 = g_apply(Ep, Up, Eps, T2, a1)
            a_tot = g_apply(Ep, Up, Eps, np.hypot(T1, T2), psi)
            chk('prefilter_composition', np.linalg.norm(a12 - a_tot), 0.0, 1e-12)
            ph = a1 / np.linalg.norm(a1)
            for Hm, gap_true in ((H, GM), (Hp, Gp)):
                Em0 = ex(Hm, psi); s0 = sd(Hm, psi)
                Em1 = ex(Hm, ph); s1 = sd(Hm, ph)
                prefilter_cases += 1; prefilter_sigma_up += int(s1 > s0 * (1 + 1e-9))
                if Em0 - E0 < gap_true / 2:
                    e_cases += 1; e_increase += int(Em1 - E0 > (Em0 - E0) * (1 + 1e-9) + 1e-13)
                if s1 < gap_true / 2 and Em1 - E0 < gap_true / 2:
                    eb = gap_true / 2 - np.sqrt(gap_true ** 2 / 4 - s1 ** 2); ub = min(eb / gap_true, s1 ** 2 / (gap_true - eb) ** 2)
                    off = abs(Eps - Em1) + eb
                    chk('prefilter_offset', abs(Eps - E0), off, 1e-12)
                    if off < Gp / 2 and ub < 1:
                        fin = a12 / np.linalg.norm(a12); Dg = ex(X, fin)
                        qq = ub / (1 - ub) * np.exp(-T2 ** 2 * Gp * (Gp - 2 * off)); fb = qq / (1 + qq)
                        chk('eq13_prefiltered', abs(Dg - Dex), W * (np.sqrt(fb * (1 - fb)) + fb), tolD)
        # error budget (i): any sector-preserving perturbation of the filter operator with norm eps_sim
        if eps < Gp / 2:
            Tg = 3 / Gp; gpsi = g_apply(Ep, Up, Eps, Tg, psi); Z = np.real(np.vdot(gpsi, gpsi))
            Dg = ex(X, gpsi / np.sqrt(Z))
            for es in (1e-6, 1e-3, 0.05 * Z):
                if not Z > 2 * es:
                    continue
                for mode in ('random', 'adversarial'):
                    if mode == 'random':
                        M = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
                    else:                       # push g psi towards an extreme eigenvector of X
                        xe, xv = np.linalg.eigh(X); tgt = xv[:, -1] if Dg < 0.5 * (xe[0] + xe[-1]) else xv[:, 0]
                        M = np.outer(tgt, psi.conj())
                    M *= es / np.linalg.norm(M, 2)
                    v = gpsi + M @ psi; Dgt = ex(X, v / np.linalg.norm(v))
                    chk('budget_i', abs(Dgt - Dg), 2 * es * W / (Z - 2 * es), tolD)
    res = dict(n_states=n_states, checks=nchk, violations=viol, violation_detail=vdet, worst_ratio={k: float(v) for k, v in worst.items()},
               prefilter_sigma_increased=f"{prefilter_sigma_up}/{prefilter_cases}",
               prefilter_energy_increased=f"{e_increase}/{e_cases}",
               gaps_mEh=dict(E1_E0=1e3 * (E1 - E0), G_M=1e3 * GM, Gp=1e3 * Gp), W_over_D=W / abs(Dex), W1_over_W=W1 / W)
    nv = sum(viol.values()); nc = sum(nchk.values())
    print(f"{label or sysd['name']:28s} checks {nc:6d}  violations {nv}  "
          f"sigma up after pre-filter {res['prefilter_sigma_increased']}  E up {res['prefilter_energy_increased']}")
    if viol:
        print('   VIOLATIONS', {k: (viol[k], '%.1e' % vdet[k][0], '%.3g' % vdet[k][1]) for k in viol})
    return res


if __name__ == '__main__':
    out = {'A': {}, 'B': {}, 'C': {}}
    out['A']['CH2_triplet'] = part_a('CH2 triplet', 0, (4, 2))
    out['A']['CH2+_quartet'] = part_a('CH2+ quartet', 1, (4, 1))
    out['A']['CH2_quintet'] = part_a('CH2 quintet', 0, (5, 1))
    for name in ('CH2', 'NH', 'NH_stretch', 'O2'):
        base = molecular(name)
        out['B'][name] = run_system(base)
        for k, kind in enumerate(('tiny_gap', 'wrong_spin_close', 'clustered')):
            sy = synthetic(base, kind, seed=100 + k)
            out['C'][sy['name']] = run_system(sy, n_states=250)
    json.dump(out, open('results/sec2_stress.json', 'w'), indent=1, default=float)
