"""Cost of a proven accuracy for the Gaussian filter, and the role of the VQE input state (Section 4.5, Table 2, Figure 3).

For every system and pool at the first ADAPT iterate with chemical accuracy:
  - the filter length T needed to prove |D_g - D_exact| <= 1% and 0.1% of |D| from eqs 9 and 11, with (a) the exact
    fidelity and (b) a measurable fidelity bound, the variance form of eq 10 applied to H' = H + mu_f (S^2 - S(S+1)):
    1 - F <= sigma_H'(psi)^2 / (E0 + G' - E'(psi))^2;
  - the certificate from measured inputs only (eqs 12 and 13 of the manuscript): measured E and sigma of H (sector gap
    G_M) or of H' (gap G'), gap bound Delta (the true gap; also 0.9 of it), hypothesis E - E0 < Delta/2, spectral width
    W of X; T for 1% and 0.1% ('cert_T1', 'cert_T01', 'cert_T01_gap90') and the simulation error per evolution that
    keeps the implementation error below 0.01% ('eps_sim_for_1e-4'). The 'meas' entries (variance bound with the exact
    E0 and exact-state constants) are kept for comparison only;
  - the same for the reference determinant as filter input, its energy error relative to G'/2 (the guarantee and a
    filter centred at E(psi) need E(psi) - E0 < G'/2) and its actual filtered error;
  - for the figure, the actual error and the proven bound versus T (stretched NH: with and without the spin shift).
Naphthalene (natural orbitals) uses the stored histories, results/naph_bounds.json and results/naph_filter.json, and a
300-vector Lanczos filter for the reference determinant. Writes results/filter_guarantee.json.
"""
import json, sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, remedies, penalty
from vqe import pool_gsd, pool_sf

MU_F, CHEM = 0.05, 1.6e-3
TGRID = np.linspace(0.0, 6.0, 61)                 # T (E1 - E0)

def bound_pct(Tg, r, eps, Gp, sig, nX):
    q = r * np.exp(-Tg ** 2 * Gp * (Gp - 2 * eps)); fg = q / (1 + q)
    return 100 * (2 * np.sqrt(fg * (1 - fg)) * sig + fg * nX)

def T_for(target, r, eps, Gp, sig, nX, gap):
    if not (eps < Gp / 2) or not np.isfinite(r):
        return None
    lo, hi = 0.0, 50.0 / gap
    if bound_pct(hi, r, eps, Gp, sig, nX) > target:
        return None
    for _ in range(200):
        m = 0.5 * (lo + hi); lo, hi = (m, hi) if bound_pct(m, r, eps, Gp, sig, nX) > target else (lo, m)
    return hi * gap                                # in units of 1/(E1 - E0)

from filter_guarantee_lib import cert_inputs, cert_bound_pct

def cert_T(target, routes, W, gap):
    """smallest T (E1 - E0) whose certificate is below target over the available routes (sig, Delta, Delta_filter)"""
    best = None
    for sig, Dl, Df in routes:
        ci = cert_inputs(sig, Dl)
        if ci is None or not (ci[0] < Df / 2):
            continue
        eb, u = ci; lo, hi = 0.0, 50.0 / gap
        if cert_bound_pct(hi, eb, u, Df, W) > target:
            continue
        for _ in range(200):
            m = 0.5 * (lo + hi); lo, hi = (m, hi) if cert_bound_pct(m, eb, u, Df, W) > target else (lo, m)
        best = hi * gap if best is None else min(best, hi * gap)
    return best

def cert_curve(routes, W, gap):
    out = []
    for x in TGRID:
        vals = [cert_bound_pct(x / gap, *cert_inputs(s, Dl), Df, W) for s, Dl, Df in routes
                if cert_inputs(s, Dl) is not None and cert_inputs(s, Dl)[0] < Df / 2]
        out.append(float(min(vals)) if vals else None)
    return out

def cert_record(routes, W, gap, Z=1.0):
    rec = {'cert_T1': cert_T(1, routes, W, gap), 'cert_T01': cert_T(0.1, routes, W, gap),
           'cert_curve': cert_curve(routes, W, gap),
           'cert_routes': [{'sigma_mEh': 1e3 * s, 'Delta_mEh': 1e3 * Dl, 'Delta_filter_mEh': 1e3 * Df,
                            'eps_bound_mEh': (1e3 * cert_inputs(s, Dl)[0]) if cert_inputs(s, Dl) else None,
                            'infid_bound': cert_inputs(s, Dl)[1] if cert_inputs(s, Dl) else None} for s, Dl, Df in routes],
           # gap lower bound 10% below the true gap
           'cert_T01_gap90': cert_T(0.1, [(s, 0.9 * Dl, 0.9 * Df) for s, Dl, Df in routes], W, gap),
           # simulation error per evolution that keeps the implementation error below 0.01% of |D|: 2 eps W / (Z - 2 eps)
           'eps_sim_for_1e-4': 1e-4 * Z / (2 * W + 2e-4)}
    return rec

def analyse(psi, H, Hp_diag_basis, V, E0, T, X, Dex, gap, Gp, sig, nX, Ep, GM=None, W=None):
    psi = psi / np.linalg.norm(psi)
    E = psi @ H @ psi; eps = E - E0; F = (T @ psi) ** 2
    c = V.T @ psi; Epsi_p = (c ** 2) @ Ep; sHp = np.sqrt(max((c ** 2) @ Ep ** 2 - Epsi_p ** 2, 0.0))
    u = sHp ** 2 / (E0 + Gp - Epsi_p) ** 2 if Epsi_p < E0 + Gp else np.inf
    r_ex = (1 - F) / F; r_ms = u / (1 - u) if u < 1 else np.inf
    rec = {'eps_mEh': 1e3 * eps, 'F': F, 'sigmaHp_mEh': 1e3 * sHp, 'infid_bound_var': u,
           'T1_exact': T_for(1, r_ex, eps, Gp, sig, nX, gap), 'T01_exact': T_for(0.1, r_ex, eps, Gp, sig, nX, gap),
           'T1_meas': T_for(1, r_ms, eps, Gp, sig, nX, gap), 'T01_meas': T_for(0.1, r_ms, eps, Gp, sig, nX, gap),
           'eps_below_half_gap': bool(eps < Gp / 2)}
    sH = np.sqrt(max((c ** 2) @ Hp_diag_basis ** 2 - ((c ** 2) @ Hp_diag_basis) ** 2, 0.0)); rec['sigmaH_mEh'] = 1e3 * sH
    if W is not None:
        rec.update(cert_record([(sH, GM, Gp), (sHp, Gp, Gp)], W, gap))
    for tag, Es in [('shift', Ep), ('noshift', None)]:
        curve = []
        for x in TGRID:
            Tt = x / gap; Ec = Es if Es is not None else Hp_diag_basis
            g = np.exp(-0.5 * ((Ec - E) * Tt) ** 2); y = V @ (g * c); Ng = y @ y; y /= np.sqrt(Ng)
            curve.append((float(x), float(100 * abs(y @ X @ y - Dex) / abs(Dex)), float(Ng)))
        rec['curve_' + tag] = curve
    rec['bound_curve_exact'] = [float(bound_pct(x / gap, r_ex, eps, Gp, sig, nX)) for x in TGRID]
    rec['bound_curve_meas'] = [float(bound_pct(x / gap, r_ms, eps, Gp, sig, nX)) if np.isfinite(r_ms) else None for x in TGRID]
    return rec

out = {'TGRID': TGRID.tolist(), 'MU_F': MU_F}
for name in (sys.argv[1:] or ['CH2', 'NH', 'NH_stretch', 'O2']):
    sysd = remedies.build(name); space = sysd['space']
    H, Xz, cD, Dex0 = sysd['H'], sysd['Xz'], sysd['cD'], sysd['Dex']; X = cD * Xz
    S2 = penalty.spin_matrix(space, sysd['ncas'], sysd['nelecas']); ws, Vs = np.linalg.eigh(S2); PT = Vs[:, np.abs(ws - 2) < 1e-6]
    Ev, V = np.linalg.eigh(H); s2E = np.einsum('ij,ik,kj->j', V, S2, V)
    T = V[:, 0]; E0 = Ev[0]; Dex = T @ X @ T
    trip = np.abs(s2E - 2) < 1e-4; gap = Ev[trip][1] - E0
    Ep = Ev + MU_F * (s2E - 2); Gp = np.sort(Ep)[1] - Ep[0]; G0 = Ev[1] - E0
    r = (X - Dex * np.eye(len(T))) @ T; sig = np.linalg.norm(r); nX = np.max(np.abs(np.linalg.eigvalsh(X - Dex * np.eye(len(T)))))
    res = {'gap_mEh': 1e3 * gap, 'Gp_mEh': 1e3 * Gp, 'G_noshift_mEh': 1e3 * G0, 'sigmaX_over_D': sig / abs(Dex), 'normX_over_D': nX / abs(Dex)}
    sigD, nXD = sig / abs(Dex), nX / abs(Dex)
    wX = np.linalg.eigvalsh(X); WD = (wX[-1] - wX[0]) / abs(Dex); res['W_over_D'] = WD; res['G_M_mEh'] = 1e3 * G0
    for pname in ['GSD', 'QEB', 'sfGSD']:
        pool = pool_sf(space) if pname == 'sfGSD' else pool_gsd(space, qubit=(pname == 'QEB'))
        rr = penalty.run(sysd, S2, PT, pool, pname, 0.0, 0.0, keep_states=True)
        ic = next(i for i, q in enumerate(rr['rows']) if q['dE'] < CHEM)
        res[pname] = analyse(rr['states'][ic], H, Ev, V, E0, T, X, Dex, gap, Gp, sigD, nXD, Ep, G0, WD); res[pname]['n'] = ic
    res['refdet'] = analyse(sysd['ref'], H, Ev, V, E0, T, X, Dex, gap, Gp, sigD, nXD, Ep, G0, WD)
    out[name] = res
    for k in ['GSD', 'QEB', 'sfGSD', 'refdet']:
        a = res[k]; f = lambda v: '-' if v is None else f'{v:.2f}'
        print(f"{name:10s} {k:6s} eps {a['eps_mEh']:8.2f} mEh  1-F {1 - a['F']:.2e}  T(1%) exact {f(a['T1_exact'])} meas {f(a['T1_meas'])}  "
              f"T(0.1%) exact {f(a['T01_exact'])} meas {f(a['T01_meas'])}  err@T=3 {a['curve_shift'][30][1]:.3f}%  eps<G'/2 {a['eps_below_half_gap']}  "
              f"CERT T(1%) {f(a['cert_T1'])} T(0.1%) {f(a['cert_T01'])} gap90 {f(a['cert_T01_gap90'])} eps_sim {a['eps_sim_for_1e-4']:.1e}", flush=True)
    print(f"{name:10s} gap {res['gap_mEh']:.1f}  G' {res['Gp_mEh']:.1f}  G(no shift) {res['G_noshift_mEh']:.1f} mEh", flush=True)

# ---------------- naphthalene (natural orbitals)
import vqe_sparse as sps
from xop import XOperator
from fockmap import d_prefactor
from zfs import HARTREE2CM
from naph_filter_lib import gauss_filter
nb = json.load(open('results/naph_bounds.json')); nf = json.load(open('results/naph_filter.json'))
dat = np.load('results/naph_setup_no.npz'); ncas, nel = 10, (6, 4)
spc = sps.SparseSpace(ncas, nel); shape = (spc.na, spc.nb)
Hs = sps.Hamiltonian(dat['h1'], dat['eri'], float(dat['ecore']), ncas, nel); Hm = lambda v: Hs(v, shape)
Fp = np.einsum('ai,bj,ab...->ij...', dat['V'], dat['V'], dat['F']); cD = 1.5 * d_prefactor(1.0) * HARTREE2CM
Xop = XOperator(ncas, nel, Fp[2, 2]); Tn = dat['ci'].ravel(); Tn /= np.linalg.norm(Tn); E0n = float(dat['e_casci'])
Dn = cD * (Tn @ Xop(Tn)); gapn = nb['gap_mEh'] * 1e-3; Gpn = nb['Gp_lower_mEh'] * 1e-3; sgn, nXn = nb['sigmaX_over_D'], nb['normX_over_D']
resn = {'gap_mEh': nb['gap_mEh'], 'Gp_mEh': nb['Gp_lower_mEh'], 'sigmaX_over_D': sgn, 'normX_over_D': nXn}
import scipy.sparse.linalg as sla
XL = sla.LinearOperator((spc.dim, spc.dim), matvec=lambda v: cD * Xop(np.ascontiguousarray(v.ravel())), dtype=float)
xmax = sla.eigsh(XL, k=1, which='LA', tol=1e-8, return_eigenvectors=False)[0]
xmin = sla.eigsh(XL, k=1, which='SA', tol=1e-8, return_eigenvectors=False)[0]
Wn = (xmax - xmin) / abs(Dn); GMn = nb['G_M_mEh'] * 1e-3; resn['W_over_D'] = float(Wn); resn['G_M_mEh'] = nb['G_M_mEh']
print(f'naph W/|D| {Wn:.2f}  G_M {nb["G_M_mEh"]:.1f} mEh', flush=True)
errb = {p: json.load(open(f'results/errbar_naph_no_{p}.json')) for p in ['GSD', 'sfGSD']}
for p in ['GSD', 'sfGSD', 'refdet']:
    hp = json.load(open(f'results/naph_no_{"GSD" if p == "refdet" else p}.json'))['history']
    ic = 0 if p == 'refdet' else next(i for i, x in enumerate(hp) if x['dE'] < CHEM)
    x = hp[ic]; F, eps = x['fid'], x['dE']; sH = errb['GSD' if p == 'refdet' else p][ic]['sigma_H']
    u = sH ** 2 / (gapn - eps) ** 2 if eps < gapn else np.inf
    r_ex = (1 - F) / F; r_ms = u / (1 - u) if u < 1 else np.inf
    a = {'n': ic, 'eps_mEh': 1e3 * eps, 'F': F, 'sigmaHp_mEh': 1e3 * sH, 'infid_bound_var': u, 'eps_below_half_gap': bool(eps < Gpn / 2),
         'T1_exact': T_for(1, r_ex, eps, Gpn, sgn, nXn, gapn), 'T01_exact': T_for(0.1, r_ex, eps, Gpn, sgn, nXn, gapn),
         'T1_meas': T_for(1, r_ms, eps, Gpn, sgn, nXn, gapn), 'T01_meas': T_for(0.1, r_ms, eps, Gpn, sgn, nXn, gapn),
         'bound_curve_exact': [float(bound_pct(t / gapn, r_ex, eps, Gpn, sgn, nXn)) for t in TGRID],
         'bound_curve_meas': [float(bound_pct(t / gapn, r_ms, eps, Gpn, sgn, nXn)) if np.isfinite(r_ms) else None for t in TGRID]}
    if p != 'refdet':
        g = nf['runs'][p]['gauss']
        a['curve_shift'] = [(float(int(k) * gapn), abs(float(g[k]['D'])), float(g[k]['Ng'])) for k in g if k.isdigit()]
        a['curve_shift'].append((3.0, abs(float(g['rule_spin']['D'])), float(g['rule_spin']['Ng'])))
        a['curve_shift'].insert(0, (0.0, abs(float(nf['runs'][p]['D'])), 1.0))
        a['curve_shift'].sort()
    else:
        ref = np.zeros(spc.dim); ref[0] = 1.0
        Eref = float(ref @ Hm(ref)); a['curve_shift'] = []; a['centred_E0'] = []
        for xg in [1.0, 2.0, 3.0, 4.0]:                       # filter centred at E(ref), as for the ADAPT states
            y, Ng = gauss_filter(ref, Hm, xg / gapn, m=300)
            a['curve_shift'].append((xg, float(100 * abs(cD * (y @ Xop(y)) / Dn - 1)), float(Ng)))
        # centred at the exact E0 (requires an energy estimate within the gap): reuse the Lanczos basis
        m = 300; q = ref.copy(); Q = np.zeros((m, spc.dim)); al = np.zeros(m); be = np.zeros(m)
        for j in range(m):
            Q[j] = q; w = Hm(q); al[j] = q @ w
            w -= Q[:j + 1].T @ (Q[:j + 1] @ w); w -= Q[:j + 1].T @ (Q[:j + 1] @ w)
            if j + 1 < m:
                be[j] = np.linalg.norm(w); q = w / be[j]
        th, Uk = np.linalg.eigh(np.diag(al) + np.diag(be[:m - 1], 1) + np.diag(be[:m - 1], -1))
        for xg in [1.0, 2.0, 3.0, 4.0]:
            gg = np.exp(-0.5 * ((th - E0n) * xg / gapn) ** 2); y = Q.T @ (Uk @ (gg * Uk[0])); Ng = float(y @ y); y /= np.sqrt(Ng)
            a['centred_E0'].append((xg, float(100 * abs(cD * (y @ Xop(y)) / Dn - 1)), Ng))
    a.update(cert_record([(sH, GMn, Gpn)], Wn, gapn, Z=0.99))      # sigma_H of H with the sector gap G_M (S >= 2 states lie 174 mEh up)
    resn[p] = a
    f = lambda v: '-' if v is None else f'{v:.2f}'
    print(f"naph NO   {p:6s} eps {a['eps_mEh']:8.2f} mEh  1-F {1 - F:.2e}  T(1%) exact {f(a['T1_exact'])} meas {f(a['T1_meas'])}  "
          f"T(0.1%) exact {f(a['T01_exact'])} meas {f(a['T01_meas'])}  eps<G'/2 {a['eps_below_half_gap']}  "
          f"CERT T(1%) {f(a['cert_T1'])} T(0.1%) {f(a['cert_T01'])} gap90 {f(a['cert_T01_gap90'])} eps_sim {a['eps_sim_for_1e-4']:.1e}", flush=True)
    if p == 'refdet':
        print('   reference determinant, filter centred at E(ref):', a['curve_shift'], '\n   centred at exact E0:', a['centred_E0'], flush=True)
out['naph_no'] = resn
json.dump(out, open('results/filter_guarantee.json', 'w'), indent=1, default=float)
