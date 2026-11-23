"""Measured-input accuracy certificate for 20-qubit naphthalene (natural orbitals) with a pre-filter (Section 2.3).

At chemical accuracy the ADAPT states have sigma_H > Delta/2 (38.1 and 39.3 mEh against Delta/2 = 35 mEh), so
Temple's inequality cannot bound E(psi) - E0 from the moments of psi. The variance is dominated by components far
above the target, which a short Gaussian pre-filter removes. Gaussian filters with a common centre c = E(psi)
compose, g_T2 g_T1 = g_T with T^2 = T1^2 + T2^2, so the moments of the pre-filtered state phi = g_T1 psi (measurable
with Hadamard tests) can serve as the certificate inputs for the remaining filter g_T2:
  eps(phi) <= eb from sigma(phi) and Delta (Temple, hypothesis eps(phi) < Delta/2), 1 - F(phi) <= u,
  |c - E0| <= |c - E(phi)| + eb,  and eq 13 with T2 and W.
The ADAPT states at the first iterate with chemical accuracy (GSD n = 291, sfGSD n = 191) are rebuilt from the stored
operator sequence by re-optimizing the angles from the stored final angles (BFGS); the rebuilt energy and D are
compared with the stored history. Writes results/naph_certificate.json and results/naph_no_<pool>_chem_state.npy.
"""
import json, os, time, warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import minimize
from pyscf import fci
import vqe_sparse as sps
from xop import XOperator
from fockmap import d_prefactor
from zfs import HARTREE2CM
from naph_filter_lib import gauss_filter
from filter_guarantee_lib import cert_inputs, cert_bound_pct

MU_F, CHEM = 0.05, 1.6e-3
X1 = 0.25         # reported pre-filter T1 (E1 - E0): sigma of the pre-filtered state (~5 mEh) stays cheap to measure (qpe_compare.py)
dat = np.load('results/naph_setup_no.npz'); ncas, nel = 10, (6, 4)
spc = sps.SparseSpace(ncas, nel); shape = (spc.na, spc.nb)
Hs = sps.Hamiltonian(dat['h1'], dat['eri'], float(dat['ecore']), ncas, nel); Hm = lambda v: Hs(v, shape)
S2 = lambda v: fci.spin_op.contract_ss(v.reshape(shape), ncas, nel).ravel()
Fp = np.einsum('ai,bj,ab...->ij...', dat['V'], dat['V'], dat['F']); cD = 1.5 * d_prefactor(1.0) * HARTREE2CM
Xop = XOperator(ncas, nel, Fp[2, 2]); Tn = dat['ci'].ravel(); Tn /= np.linalg.norm(Tn); E0 = float(dat['e_casci'])
Dex = cD * (Tn @ Xop(Tn))
fg = json.load(open('results/filter_guarantee.json'))['naph_no']; nb = json.load(open('results/naph_bounds.json'))
gap = nb['gap_mEh'] * 1e-3; GM = nb['G_M_mEh'] * 1e-3; Gp = nb['Gp_lower_mEh'] * 1e-3; W = fg['W_over_D']

def moments(v, Hp=False):
    hv = Hm(v)
    if Hp:
        hv = hv + MU_F * (S2(v) - 2 * v)
    E = float(v @ hv); return E, float(np.sqrt(max(hv @ hv - E ** 2, 0.0)))

SCALE = 1.0

def cert_T_total(sig, Dl, Df, dc, target):
    """smallest total T (E1 - E0) = sqrt(T1^2 + T2^2) for a given pre-filter; dc = |c - E(phi)|"""
    Dl, Df = SCALE * Dl, SCALE * Df
    ci = cert_inputs(sig, Dl)
    if ci is None:
        return None, None
    eb, u = ci; ec = dc + eb
    if not ec < Df / 2:
        return None, None
    lo, hi = 0.0, 50.0 / gap
    q = lambda T2: 100 * W * (lambda f: np.sqrt(f * (1 - f)) + f)((lambda qq: qq / (1 + qq))(u / (1 - u) * np.exp(-T2 ** 2 * Df * (Df - 2 * ec))))
    if q(hi) > target:
        return None, None
    for _ in range(200):
        m = 0.5 * (lo + hi); lo, hi = (m, hi) if q(m) > target else (lo, m)
    return hi * gap, {'eps_bound_mEh': 1e3 * eb, 'infid_bound': u, 'eps_centre_mEh': 1e3 * ec}

out = {'W_over_D': W, 'gap_mEh': 1e3 * gap, 'G_M_mEh': 1e3 * GM, 'Gp_mEh': 1e3 * Gp}
for p in ['sfGSD', 'GSD']:
    t0 = time.time()
    o = json.load(open(f'results/naph_no_{p}.json')); h = o['history']
    ic = next(i for i, x in enumerate(h) if x['dE'] < CHEM); n = h[ic]['n']
    pool = sps.pool_sf(spc) if p == 'sfGSD' else sps.pool_gsd(spc)
    lab = {tuple(op.label) if isinstance(op.label, (list, tuple)) else op.label: op for op in pool}
    sf = f'results/naph_no_{p}_chem_state.npy'
    if os.path.exists(sf):                                   # rebuilt state from an earlier run of this script
        psi = np.load(sf)
    else:
        gens = [sps.BlockGenerator(sps.op_matrix(spc, lab[tuple(l)])) for l in o['ops'][:n]]
        ref = np.zeros(spc.dim); ref[0] = 1.0
        res = minimize(sps._energy_grad, np.array(o['thetas'][:n]), args=(gens, ref, Hs, shape), jac=True,
                       method='BFGS', options={'gtol': 1e-9, 'maxiter': 4000})
        psi = sps._state(gens, res.x, ref); psi /= np.linalg.norm(psi)
        np.save(sf, psi)
    Epsi, sH = moments(psi); _, sHp = moments(psi, Hp=True)
    Dpsi = cD * (psi @ Xop(psi)); F = float((psi @ Tn) ** 2)
    rec = {'n': n, 'dE_rebuilt_mEh': 1e3 * (Epsi - E0), 'dE_history_mEh': 1e3 * h[ic]['dE'],
           'D_err_rebuilt_pct': 100 * (Dpsi - Dex) / Dex, 'D_err_history_pct': 100 * (h[ic]['D'] - Dex) / Dex,
           'F_rebuilt': F, 'F_history': h[ic]['fid'], 'sigmaH_mEh': 1e3 * sH, 'sigmaHp_mEh': 1e3 * sHp, 'prefilter': []}
    print(f"{p} n={n}: dE {rec['dE_rebuilt_mEh']:.4f} (history {rec['dE_history_mEh']:.4f}) mEh, D err {rec['D_err_rebuilt_pct']:+.3f} "
          f"(history {rec['D_err_history_pct']:+.3f})%, F {F:.5f} ({h[ic]['fid']:.5f}), sigma_H {1e3*sH:.1f}, sigma_H' {1e3*sHp:.1f} mEh, "
          f"t={time.time()-t0:.0f}s", flush=True)
    best = {}
    for x1 in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        if x1 == 0.0:
            phi, Ng = psi, 1.0
        else:
            phi, Ng = gauss_filter(psi, Hm, x1 / gap, m=300)
        Ephi, sphi = moments(phi); _, sphip = moments(phi, Hp=True)
        dc = abs(Epsi - Ephi)
        r = {'x1': x1, 'Ng': Ng, 'dE_mEh': 1e3 * (Ephi - E0), 'sigmaH_mEh': 1e3 * sphi, 'sigmaHp_mEh': 1e3 * sphip,
             'centre_offset_mEh': 1e3 * dc}
        for tgt, key in [(1.0, 'T1'), (0.1, 'T01')]:
            vals = []
            for s, Dl, Df in [(sphi, GM, Gp), (sphip, Gp, Gp)]:
                x2, info = cert_T_total(s, Dl, Df, dc, tgt)
                if x2 is not None:
                    vals.append((np.hypot(x1, x2), x2, info))
            if vals:
                v = min(vals, key=lambda z: z[0]); r[key] = {'total': v[0], 'x2': v[1], **v[2]}
                if key not in best or v[0] < best[key]['total']:
                    best[key] = {'x1': x1, 'total': v[0]}
            else:
                r[key] = None
        rec['prefilter'].append(r)
        f2 = lambda d: '-' if d is None else f"{d['total']:.2f}"
        print(f"   x1={x1:.2f} Ng={Ng:.4f} dE={1e3*(Ephi-E0):.4f} sigma_H={1e3*sphi:.2f} sigma_H'={1e3*sphip:.2f} mEh  "
              f"T_total(1%) {f2(r['T1'])}  T_total(0.1%) {f2(r['T01'])}", flush=True)
    rec['best'] = best
    SCALE = 0.9; b90 = {}
    for r in rec['prefilter']:
        phi = psi if r['x1'] == 0.0 else gauss_filter(psi, Hm, r['x1'] / gap, m=300)[0]
        for tgt, key in [(1.0, 'T1'), (0.1, 'T01')]:
            vals = [np.hypot(r['x1'], x2) for s, Dl, Df in [(r['sigmaH_mEh'] * 1e-3, GM, Gp), (r['sigmaHp_mEh'] * 1e-3, Gp, Gp)]
                    for x2 in [cert_T_total(s, Dl, Df, r['centre_offset_mEh'] * 1e-3, tgt)[0]] if x2 is not None]
            if vals and (key not in b90 or min(vals) < b90[key]['total']):
                b90[key] = {'x1': r['x1'], 'total': float(min(vals))}
        if r['x1'] == X1:
            rec['chosen_gap90_T01'] = min([np.hypot(X1, x2) for s, Dl, Df in [(r['sigmaH_mEh'] * 1e-3, GM, Gp), (r['sigmaHp_mEh'] * 1e-3, Gp, Gp)]
                                           for x2 in [cert_T_total(s, Dl, Df, r['centre_offset_mEh'] * 1e-3, 0.1)[0]] if x2 is not None])
    SCALE = 1.0
    rc = next(q for q in rec['prefilter'] if q['x1'] == X1)
    rec['chosen'] = {'x1': X1, 'T1': rc['T1']['total'], 'T01': rc['T01']['total'], 'T01_gap90': rec['chosen_gap90_T01'],
                     'sigmaH_mEh': rc['sigmaH_mEh'], 'sigmaHp_mEh': rc['sigmaHp_mEh']}
    rec['best_gap90'] = b90; out[p] = rec
    print(p, 'best with gap bound 0.9 G', b90, flush=True)
    print(p, 'best', best, flush=True)
TGRID = json.load(open('results/filter_guarantee.json'))['TGRID']
for p in ['sfGSD', 'GSD']:
    r = next(q for q in out[p]['prefilter'] if q['x1'] == X1)
    curve = []
    for x in TGRID:
        vals = []
        for s, Dl, Df in [(r['sigmaH_mEh'] * 1e-3, GM, Gp), (r['sigmaHp_mEh'] * 1e-3, Gp, Gp)]:
            ci = cert_inputs(s, Dl)
            if ci is None or x < r['x1']:
                continue
            ec = r['centre_offset_mEh'] * 1e-3 + ci[0]
            if ec < Df / 2:
                vals.append(cert_bound_pct(np.sqrt(x ** 2 - r['x1'] ** 2) / gap, ec, ci[1], Df, W))
        curve.append(float(min(vals)) if vals else None)
    out[p]['cert_curve'] = curve
json.dump(out, open('results/naph_certificate.json', 'w'), indent=1, default=float)
