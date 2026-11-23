"""End-to-end finite-shot simulation of the certified filter protocol (Section 4.5, Section S7).

Each replica runs the protocol of the paper with statistical inputs:
  1. the filter center is the measured energy c = E(psi) + s_E xi  (s_E = 0.5 mEh);
  2. sigma of H (or H') is measured by amplitude estimation to +-50% (95%), sigma_hat = sigma (1 + 0.25 xi'),
     and enters Temple's inequality through its upper confidence limit 2 sigma_hat;
  3. if 2 sigma_hat >= Delta/2, a first filter stage tau_1 = 0.25, 0.5, 1 or 2 /(E1 - E0) is applied and its
     energy and sigma (measured the same way) are the certificate inputs (two-stage filter);
  4. the center offset is bounded by |c - E0| <= eps_bar + 2 s_E (+ |c - E_hat(phi)| + 2 s_E for two stages);
  5. the filter length tau is the smallest one at which eq 13 proves 0.1% of |D|;
  6. every controlled evolution is implemented with a coherent error, U(t) -> U(t) exp(-i eps_sim G), ||G|| = 1;
  7. D_g is estimated from N Hadamard tests on a block encoding of (X - c_I)/lambda (lambda: Pauli 1-norm of X
     without the identity) and N Hadamard tests of the overlap, each with freshly sampled times and one +-1
     outcome. Averaged over the times, each outcome is Bernoulli with mean <g~psi|B|g~psi>, so the shot statistics
     are simulated exactly by binomial draws (checked against explicit time sampling for CH2). N is chosen for a
     standard error of 1% of |D|.
The replica is covered if |D_hat - D_exact| <= cert bound + implementation bound + 1.96 SE_hat.
Naphthalene (natural orbitals) uses a 400-vector Lanczos representation of the ADAPT state with H'.
Writes results/finite_shot.json. Run from the repository root (see README).
"""
import json, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import openfermion as of
import remedies, penalty
from vqe import pool_gsd, pool_sf
from fockmap import d_component_op, d_prefactor
from filter_guarantee_lib import cert_inputs

MU_F, CHEM, S_E = 0.05, 1.6e-3, 0.5e-3
rng = np.random.default_rng(20261123)


class Model:
    """Everything in an eigenbasis of H' (12 qubits: exact; naphthalene: Ritz basis of a Lanczos space)."""
    def __init__(self, Ep, He, Xe, a, cI, lam, E0, Dex, gaps, W, W1):
        self.Ep, self.He, self.Xe, self.a = Ep, He, Xe, a.astype(complex)
        self.cI, self.lam, self.E0, self.Dex = cI, lam, E0, Dex
        self.GM, self.Gp, self.gap = gaps
        self.W, self.W1 = W, W1
        self.dim = len(Ep)
        Hp = np.diag(Ep)
        self.ops = {'H': He, "H'": Hp}

    def g(self, v, c, tau):
        return np.exp(-tau ** 2 * (self.Ep - c) ** 2 / 2) * v

    def mom(self, v, op):
        v = v / np.linalg.norm(v); M = self.ops[op]
        hv = M @ v; E = np.real(np.vdot(v, hv)); return E, float(np.linalg.norm(hv - E * v))

    def dval(self, v):
        v = v / np.linalg.norm(v); return float(np.real(np.vdot(v, self.Xe @ v)))


def certify(model, Ehat, sig_up, Delta, Dprime, ec_extra, W, target=1e-3):
    ci = cert_inputs(sig_up, Delta)
    if ci is None:
        return None
    eb, u = ci; ec = eb + ec_extra
    if not ec < Dprime / 2:
        return None
    def bnd(t2):                                                       # in units of |D|
        q = u / (1 - u) * np.exp(-t2 ** 2 * Dprime * (Dprime - 2 * ec)); f = q / (1 + q)
        return W * (np.sqrt(f * (1 - f)) + f)
    lo, hi = 0.0, 60.0 / model.gap
    if bnd(hi) > target:
        return None
    for _ in range(100):
        m = 0.5 * (lo + hi); lo, hi = (m, hi) if bnd(m) > target else (lo, m)
    return hi, bnd(hi)


def replica(model, eps_sim=0.0, dc=0.0, use_W1=False, sigma_exact=False, gap_scale=1.0, rs=rng, G=None):
    a = model.a / np.linalg.norm(model.a)
    E_psi = np.real(np.vdot(a, model.He @ a))
    c = E_psi + S_E * rs.standard_normal() + dc                       # measured center (+ deliberate offset dc)
    W = model.W1 if use_W1 else model.W
    best = None; noise = (c - dc - E_psi)
    for op, Delta in (('H', model.GM), ("H'", model.Gp)):
        Delta *= gap_scale; Dp = model.Gp * gap_scale
        for x1 in (0.0, 0.25, 0.5, 1.0, 2.0):
            t1 = x1 / model.gap
            phi = model.g(a, c, t1) if x1 > 0 else a
            Ephi, sphi = model.mom(phi, op)
            shat = sphi if sigma_exact else sphi * (1 + 0.25 * rs.standard_normal())
            sig_up = sphi if sigma_exact else 2 * abs(shat)
            if x1 == 0:
                extra = 2 * S_E                                        # a deliberate offset dc is NOT known to the certificate
                ci_ok = bool(sig_up >= sphi and abs(noise) <= 2 * S_E)
            else:
                nphi = S_E * rs.standard_normal(); Ehat_phi = Ephi + nphi
                extra = abs(c - Ehat_phi) + 2 * S_E
                ci_ok = bool(sig_up >= sphi and abs(nphi) <= 2 * S_E)
            r = certify(model, c, sig_up, Delta, Dp, extra, W)
            if r is not None:
                tau = np.hypot(t1, r[0])
                if best is None or tau < best[0]:
                    best = (tau, r[1], op, x1, ci_ok)
                break                                                  # shortest first stage that works
    if best is None:
        return None
    tau, Bcert, op, x1, ci_ok = best
    # does hypothesis (H2) hold for the route used?  single stage: energy of the measured operator;
    # two stages: energy of H' (Section 2.3)
    eH = E_psi - model.E0; eHp = float(np.real(np.vdot(a, model.Ep * a))) - model.E0
    Dl = (model.GM if op == 'H' else model.Gp) * gap_scale
    h2 = bool((eH if (op == 'H' and x1 == 0) else eHp) < Dl / 2)
    # implemented filter: g~ = g V,  V = exp(-i eps_sim G)
    v = a
    if eps_sim > 0:
        w_, U_ = np.linalg.eigh(G); v = U_ @ (np.exp(-1j * eps_sim * w_) * (U_.conj().T @ a))
    gv = model.g(v, c, tau)
    Z = float(np.real(np.vdot(gv, gv)))
    mX = float(np.real(np.vdot(gv, (model.Xe - model.cI * np.eye(model.dim)) @ gv))) / model.lam
    Dg_impl = model.cI + model.lam * mX / Z
    # number of circuits for a 1% standard error (delta method)
    var1 = (model.lam / Z) ** 2 * ((1 - mX ** 2) + (mX / Z) ** 2 * (1 - Z ** 2))
    N = int(np.ceil(var1 / (0.01 * abs(model.Dex)) ** 2))
    kX = rs.binomial(N, (1 + mX) / 2); kZ = rs.binomial(N, (1 + Z) / 2)
    mXh, Zh = 2 * kX / N - 1, 2 * kZ / N - 1
    Dhat = model.cI + model.lam * mXh / Zh
    var_h = (model.lam / Zh) ** 2 * ((1 - mXh ** 2) + (mXh / Zh) ** 2 * (1 - Zh ** 2)) / N
    SE = float(np.sqrt(max(var_h, 0)))
    Bimpl = 2 * eps_sim * W * abs(model.Dex) / (Z - 2 * eps_sim) if eps_sim > 0 else 0.0
    D = abs(model.Dex); Bcert = Bcert * D
    return dict(tau=tau * model.gap, x1=x1, route=op, N=N, Z=Z, err=100 * abs(Dhat - model.Dex) / D,
                sys=100 * abs(Dg_impl - model.Dex) / D, Bcert=100 * Bcert / D, Bimpl=100 * Bimpl / D, SE=100 * SE / D,
                covered=bool(abs(Dhat - model.Dex) <= Bcert + Bimpl + 1.96 * SE),
                sys_ok=bool(abs(Dg_impl - model.Dex) <= Bcert + Bimpl + 1e-12 * D), h2=h2, ci_ok=ci_ok,
                z=float((Dhat - Dg_impl) / SE) if SE > 0 else 0.0)


def summarize(rows):
    ok = [r for r in rows if r is not None]
    if not ok:
        return {'n': 0}
    f = lambda k: np.array([r[k] for r in ok])
    return dict(n=len(ok), n_fail=len(rows) - len(ok), tau_med=float(np.median(f('tau'))), tau_max=float(f('tau').max()),
                x1=sorted(set(float(x) for x in f('x1'))), N_med=float(np.median(f('N'))), Z_med=float(np.median(f('Z'))),
                err_med=float(np.median(f('err'))), err_95=float(np.percentile(f('err'), 95)), sys_max=float(f('sys').max()),
                Bcert=float(np.median(f('Bcert'))), Bimpl_max=float(f('Bimpl').max()), SE_med=float(np.median(f('SE'))),
                coverage=float(np.mean(f('covered'))), sys_ok=bool(np.all(f('sys_ok')[f('h2')])) if f('h2').any() else None,
                n_h2_violated=int((~f('h2')).sum()), n_fail_when_h2_violated=int((~f('sys_ok') & ~f('h2')).sum()),
                n_fail_when_h2_holds=int((~f('sys_ok') & f('h2')).sum()), n_ci_failed=int((~f('ci_ok')).sum()),
                n_fail_when_all_hold=int((~f('sys_ok') & f('h2') & f('ci_ok')).sum()),
                z_mean=float(np.mean(f('z'))), z_std=float(np.std(f('z'))))


def pauli_norm(Fzz, ncas, S, cD):
    q = of.jordan_wigner(d_component_op(Fzz, ncas, S))
    cI = cD * float(np.real(q.terms.get((), 0.0)))
    lam = cD * sum(abs(v) for t, v in q.terms.items() if t != ())
    return cI, lam


def small_models(name):
    sysd = remedies.build(name); space = sysd['space']
    H, X, cD = sysd['H'], sysd['cD'] * sysd['Xz'], sysd['cD']
    S2 = penalty.spin_matrix(space, sysd['ncas'], sysd['nelecas']); ws, Vs = np.linalg.eigh(S2); PT = Vs[:, np.abs(ws - 2) < 1e-6]
    Ev, V = np.linalg.eigh(H); s2E = np.einsum('ij,ik,kj->j', V, S2, V)
    E0 = Ev[0]; T = V[:, 0]; Dex = T @ X @ T
    gap = Ev[np.abs(s2E - 2) < 1e-4][1] - E0; GM = Ev[1] - E0
    Hp = H + MU_F * (S2 - 2 * np.eye(len(H))); Ep, Up = np.linalg.eigh(Hp); Gp = Ep[1] - Ep[0]
    wX = np.linalg.eigvalsh(X); W = (wX[-1] - wX[0]) / abs(Dex)
    cI, lam = pauli_norm(sysd['Fp'][2, 2], sysd['ncas'], sysd['S'], cD)
    assert np.max(np.abs(wX - cI)) <= lam * (1 + 1e-12)
    W1 = 2 * lam / abs(Dex)
    He, Xe = Up.T @ H @ Up, Up.T @ X @ Up
    states = {}
    for pname in ['GSD', 'QEB', 'sfGSD']:
        pool = pool_sf(space) if pname == 'sfGSD' else pool_gsd(space, qubit=(pname == 'QEB'))
        rr = penalty.run(sysd, S2, PT, pool, pname, 0.0, 0.0, keep_states=True)
        for lab_, thr in (('chem', CHEM), ('5mEh', 5e-3), ('15mEh', 15e-3)):
            ic = next((i for i, q in enumerate(rr['rows']) if q['dE'] < thr), None)
            if ic is not None:
                states[(pname, lab_)] = (rr['states'][ic], rr['rows'][ic]['dE'])
    mk = lambda psi: Model(Ep, He, Xe, Up.T @ psi, cI, lam, E0, Dex, (GM, Gp, gap), W, W1)
    return mk, states, dict(gap=gap, GM=GM, Gp=Gp, W=W, W1=W1, lam_over_D=lam / abs(Dex), Dex=Dex, Ep=Ep, Up=Up, X=X, H=H)


def naph_models(m=400):
    import vqe_sparse as sps
    from pyscf import fci
    from xop import XOperator
    from zfs import HARTREE2CM
    dat = np.load('results/naph_setup_no.npz'); ncas, nel = 10, (6, 4)
    spc = sps.SparseSpace(ncas, nel); shape = (spc.na, spc.nb)
    Hs = sps.Hamiltonian(dat['h1'], dat['eri'], float(dat['ecore']), ncas, nel); Hm = lambda v: Hs(v, shape)
    S2 = lambda v: fci.spin_op.contract_ss(v.reshape(shape), ncas, nel).ravel()
    Hpm = lambda v: Hm(v) + MU_F * (S2(v) - 2 * v)
    Fp = np.einsum('ai,bj,ab...->ij...', dat['V'], dat['V'], dat['F']); cD = 1.5 * d_prefactor(1.0) * HARTREE2CM
    Xop = XOperator(ncas, nel, Fp[2, 2]); Tn = dat['ci'].ravel(); Tn /= np.linalg.norm(Tn); E0 = float(dat['e_casci'])
    Dex = cD * (Tn @ Xop(Tn))
    fg = json.load(open('results/filter_guarantee.json'))['naph_no']; nb = json.load(open('results/naph_bounds.json'))
    gap = nb['gap_mEh'] * 1e-3; GM = nb['G_M_mEh'] * 1e-3; Gp = nb['Gp_lower_mEh'] * 1e-3; W = fg['W_over_D']
    cI, lam = pauli_norm(Fp[2, 2], ncas, 1.0, cD); W1 = 2 * lam / abs(Dex)
    out = {}
    for p in ['GSD', 'sfGSD']:
        psi = np.load(f'results/naph_no_{p}_chem_state.npy'); psi = psi / np.linalg.norm(psi)
        Q = np.zeros((m, spc.dim)); al = np.zeros(m); be = np.zeros(m); q = psi.copy()
        for j in range(m):
            Q[j] = q; w = Hpm(q); al[j] = q @ w
            w -= Q[:j + 1].T @ (Q[:j + 1] @ w); w -= Q[:j + 1].T @ (Q[:j + 1] @ w)
            be[j] = np.linalg.norm(w); q = w / be[j]
        Tm = np.diag(al) + np.diag(be[:-1], 1) + np.diag(be[:-1], -1)
        th, Y = np.linalg.eigh(Tm); R = (Q.T @ Y)                      # Ritz vectors (dim x m)
        HR = np.array([Hm(np.ascontiguousarray(R[:, k])) for k in range(m)]).T
        XR = np.array([cD * Xop(np.ascontiguousarray(R[:, k])) for k in range(m)]).T
        He = R.T @ HR; He = 0.5 * (He + He.T); Xe = R.T @ XR; Xe = 0.5 * (Xe + Xe.T)
        a = R.T @ psi
        info = dict(ritz_E0_err_mEh=1e3 * (th[0] - E0), ritz_overlap_T=float(abs(R[:, 0] @ Tn)), psi_in_span=float(np.linalg.norm(a)),
                    dE_mEh=1e3 * (psi @ Hm(psi) - E0))
        out[p] = (Model(th, He, Xe, a, cI, lam, E0, Dex, (GM, Gp, gap), W, W1), info)
    return out, dict(gap=gap, GM=GM, Gp=Gp, W=W, W1=W1, lam_over_D=lam / abs(Dex), Dex=Dex)


def check_time_sampling(mk, psi, gap, ncirc=100000, reps=300):
    """CH2: explicit time sampling (fresh t1, t2 ~ N(0, tau^2) per circuit) versus the binomial model."""
    model = mk(psi); a = model.a / np.linalg.norm(model.a); c = np.real(np.vdot(a, model.He @ a)); tau = 3 / gap
    B = (model.Xe - model.cI * np.eye(model.dim)) / model.lam; x = model.Ep - c
    gv = model.g(a, c, tau); Z = np.real(np.vdot(gv, gv)); mX = np.real(np.vdot(gv, B @ gv))
    ests_t, ests_b = [], []
    for _ in range(reps):
        sX = sZ = 0
        for b0 in range(0, ncirc, 20000):
            nb_ = min(20000, ncirc - b0)
            t1, t2, t3, t4 = (rng.normal(0, tau, nb_) for _ in range(4))
            A1 = np.exp(-1j * np.outer(t1, x)) * a; A2 = np.exp(-1j * np.outer(t2, x)) * a
            mxt = np.real(np.einsum('ij,ij->i', (A1.conj() @ B), A2))
            A3 = np.exp(-1j * np.outer(t3, x)) * a; A4 = np.exp(-1j * np.outer(t4, x)) * a
            mzt = np.real(np.einsum('ij,ij->i', A3.conj(), A4))
            sX += np.where(rng.random(nb_) < (1 + mxt) / 2, 1, -1).sum(); sZ += np.where(rng.random(nb_) < (1 + mzt) / 2, 1, -1).sum()
        ests_t.append(model.cI + model.lam * (sX / ncirc) / (sZ / ncirc))
        kX = rng.binomial(ncirc, (1 + mX) / 2); kZ = rng.binomial(ncirc, (1 + Z) / 2)
        ests_b.append(model.cI + model.lam * (2 * kX / ncirc - 1) / (2 * kZ / ncirc - 1))
    D = abs(model.Dex); Dg = model.cI + model.lam * mX / Z
    return dict(ncirc=ncirc, reps=reps, Dg=Dg, mean_time=float(np.mean(ests_t)), sd_time=float(np.std(ests_t)),
                mean_binom=float(np.mean(ests_b)), sd_binom=float(np.std(ests_b)), sd_pct_time=100 * float(np.std(ests_t)) / D,
                sd_pct_binom=100 * float(np.std(ests_b)) / D)


def scenarios(model, R, Gmat):
    res = {}
    res['baseline'] = summarize([replica(model, G=Gmat) for _ in range(R)])
    res['exact_sigma'] = summarize([replica(model, sigma_exact=True, G=Gmat) for _ in range(R // 4)])
    res['W1'] = summarize([replica(model, use_W1=True, G=Gmat) for _ in range(R // 4)])
    res['gap90'] = summarize([replica(model, gap_scale=0.9, G=Gmat) for _ in range(R // 4)])
    for e in (1e-6, 1e-3, 1e-2):
        res[f'eps_sim={e:g}'] = summarize([replica(model, eps_sim=e, G=Gmat) for _ in range(R // 4)])
    for dc in (5e-3, 10e-3, 20e-3):
        res[f'dc={1e3*dc:g}mEh'] = summarize([replica(model, dc=dc, G=Gmat) for _ in range(R // 4)])
    return res


def random_G(dim, rs):
    A = rs.standard_normal((dim, dim)) + 1j * rs.standard_normal((dim, dim)); A = A + A.conj().T
    return A / np.max(np.abs(np.linalg.eigvalsh(A)))


if __name__ == '__main__':
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    out = {'settings': dict(s_E_mEh=1e3 * S_E, sigma_precision='+-50% (95%), upper limit 2 sigma_hat', target_cert='0.1% of |D|',
                            target_SE='1% of |D|', replicas=R, mu_f=MU_F)}
    for name in ['CH2', 'NH', 'NH_stretch', 'O2']:
        t0 = time.time(); mk, states, info = small_models(name)
        o = {'gap_mEh': 1e3 * info['gap'], 'W_over_D': info['W'], 'W1_over_D': info['W1'], 'lambda_over_D': info['lam_over_D']}
        if name == 'CH2':
            o['time_sampling_check'] = check_time_sampling(mk, states[('GSD', 'chem')][0], info['gap'])
            print('time-sampling check', o['time_sampling_check'], flush=True)
        for (p, lab_), (psi, dE) in states.items():
            model = mk(psi); Gm = random_G(model.dim, rng)
            key = f'{p}_{lab_}'
            o[key] = {'dE_mEh': 1e3 * dE}
            if lab_ == 'chem':
                o[key].update(scenarios(model, R, Gm))
            else:
                o[key]['baseline'] = summarize([replica(model, G=Gm) for _ in range(R // 4)])
            b = o[key]['baseline']
            print(f"{name:10s} {key:12s} dE {1e3*dE:6.2f} mEh  tau {b.get('tau_med', np.nan):.2f}  N {b.get('N_med', np.nan):.2e}  "
                  f"err med {b.get('err_med', np.nan):.2f}%  cov {b.get('coverage', np.nan):.3f}  sys_ok|H2 {b.get('sys_ok')}  H2 viol {b.get('n_h2_violated')} "
                  f"(fail {b.get('n_fail_when_h2_violated')})  CI fail {b.get('n_ci_failed')}  fail|all hold {b.get('n_fail_when_all_hold')}  x1 {b.get('x1')}", flush=True)
        out[name] = o; print(name, f'{time.time()-t0:.0f} s', flush=True)
        json.dump(out, open('results/finite_shot.json', 'w'), indent=1, default=float)
    t0 = time.time(); nm, info = naph_models()
    o = {'gap_mEh': 1e3 * info['gap'], 'W_over_D': info['W'], 'W1_over_D': info['W1'], 'lambda_over_D': info['lam_over_D']}
    for p, (model, inf) in nm.items():
        Gm = random_G(model.dim, rng)
        o[f'{p}_chem'] = {'krylov': inf, 'dE_mEh': inf['dE_mEh']}
        o[f'{p}_chem'].update(scenarios(model, R, Gm))
        b = o[f'{p}_chem']['baseline']
        print(f"naph {p}: krylov {inf}  tau {b.get('tau_med'):.2f} N {b.get('N_med'):.2e} err med {b.get('err_med'):.2f}% cov {b.get('coverage'):.3f} "
              f"sys_ok {b.get('sys_ok')} x1 {b.get('x1')}", flush=True)
    out['naph_no'] = o; print('naph', f'{time.time()-t0:.0f} s', flush=True)
    json.dump(out, open('results/finite_shot.json', 'w'), indent=1, default=float)
