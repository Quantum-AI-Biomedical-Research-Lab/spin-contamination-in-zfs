"""Test (not used in the manuscript): can hypothesis (H2), E(psi) - E0 < Delta/2, be checked with a rigorous classical
lower bound on E0?  E_LB = min E[D] over two-electron RDMs that satisfy the D, Q and G positivity conditions (DQG),
fixed N_alpha, N_beta and <S^2> = S(S+1).  Every true M_S = S, spin-S state is feasible, so E_LB <= E0, and
E(psi) - E_LB < Delta/2 proves (H2).  The margin that matters is (E0 - E_LB) versus Delta/2 - (E(psi) - E0).

Spin-orbital index i = 2p + sigma (sigma = 0 alpha, 1 beta), D[ij,kl] = <a+_i a+_j a_l a_k>.
The maps for Q and G and the energy and S^2 functionals are checked against the exact CASCI 2-RDM first.
Writes results/v2rdm_bound.json.  Run from the repository root (see README).
"""
import json, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import scipy.sparse as sps
import cvxpy as cp
from pyscf import fci, ao2mo
from theory_check import setup

SM = [np.array([[0, .5], [.5, 0]], complex), np.array([[0, -.5j], [.5j, 0]], complex), np.array([[.5, 0], [0, -.5]], complex)]
MSS = sum(np.einsum('ab,cd->abcd', s, s) for s in SM).real      # (s.s)[sig sig', tau tau']


class Model:
    def __init__(self, n, na, nb):
        self.n, self.na, self.nb, self.N = n, na, nb, na + nb
        ns = 2 * n; self.ns = ns
        sp = lambda i: i % 2
        pairs = [(i, j) for i in range(ns) for j in range(i + 1, ns)]
        self.cls = {I: sp(I[0]) + sp(I[1]) for I in pairs}      # 0 aa, 1 ab, 2 bb
        self.blocks = {c: [I for I in pairs if self.cls[I] == c] for c in (0, 1, 2)}
        self.pos = {I: (c, k) for c, L in self.blocks.items() for k, I in enumerate(L)}
        self.var = {}
        for c, L in self.blocks.items():
            for a in range(len(L)):
                for b in range(a, len(L)):
                    self.var[(c, a, b)] = len(self.var)
        self.nvar = len(self.var)

    def Dlin(self, i, j, k, l):
        """D[ij,kl] as {var: coef}."""
        if i == j or k == l:
            return {}
        s = 1.0
        if i > j: i, j, s = j, i, -s
        if k > l: k, l, s = l, k, -s
        c1, a = self.pos[(i, j)]; c2, b = self.pos[(k, l)]
        if c1 != c2:
            return {}
        return {self.var[(c1, min(a, b), max(a, b))]: s}

    def glin(self, i, k):
        out = {}
        for j in range(self.ns):
            for v, c in self.Dlin(i, j, k, j).items():
                out[v] = out.get(v, 0.0) + c / (self.N - 1)
        return out

    def vec_from_D(self, Dt):
        x = np.zeros(self.nvar)
        for (c, a, b), v in self.var.items():
            I, J = self.blocks[c][a], self.blocks[c][b]
            x[v] = Dt[I[0], I[1], J[0], J[1]]
        return x

    @staticmethod
    def _add(d, lin, f=1.0):
        for v, c in lin.items():
            d[v] = d.get(v, 0.0) + f * c

    def matrices(self):
        """Return list of (A, b, m): matrix = reshape(A x + b, (m, m)), for every PSD block of D, Q, G and gamma, 1-gamma."""
        ns, mats = self.ns, []
        g = {(i, k): self.glin(i, k) for i in range(ns) for k in range(ns)}

        def build(rows, entry):
            m = len(rows); R, C, V = [], [], []; b = np.zeros(m * m)
            for a, ra in enumerate(rows):
                for bb, rb in enumerate(rows):
                    lin, const = entry(ra, rb)
                    idx = a * m + bb; b[idx] = const
                    for v, c in lin.items():
                        if c != 0:
                            R.append(idx); C.append(v); V.append(c)
            return sps.csr_matrix((V, (R, C)), shape=(m * m, self.nvar)), b, m

        for c in (0, 1, 2):                      # D blocks
            L = self.blocks[c]
            mats.append(('D%d' % c,) + build(L, lambda I, J: (self.Dlin(I[0], I[1], J[0], J[1]), 0.0)))

        def qentry(I, J):
            p, q = I; r, s = J; d = {}; d0 = float(p == r and q == s) - float(p == s and q == r)
            if p == r: self._add(d, g[(s, q)], -1)
            if p == s: self._add(d, g[(r, q)], +1)
            if q == r: self._add(d, g[(s, p)], +1)
            if q == s: self._add(d, g[(r, p)], -1)
            self._add(d, self.Dlin(r, s, p, q))
            return d, d0
        for c in (0, 1, 2):                      # Q blocks
            mats.append(('Q%d' % c,) + build(self.blocks[c], qentry))

        def gentry(I, J):
            p, q = I; r, s = J; d = {}
            if p == r: self._add(d, g[(q, s)])
            self._add(d, self.Dlin(q, r, p, s))
            return d, 0.0
        allp = [(p, q) for p in range(ns) for q in range(ns)]
        for dsz in (0, 1, -1):                   # G blocks by the S_z change of a+_r a_s
            rows = [(p, q) for (p, q) in allp if (p % 2) - (q % 2) == dsz]
            mats.append(('G%+d' % dsz,) + build(rows, gentry))
        for sg in (0, 1):                        # gamma and 1 - gamma, per spin
            orb = [i for i in range(ns) if i % 2 == sg]
            mats.append(('g%d' % sg,) + build(orb, lambda i, k: (g[(i, k)], 0.0)))
            mats.append(('h%d' % sg,) + build(orb, lambda i, k: ({v: -c for v, c in g[(i, k)].items()}, float(i == k))))
        return mats

    def functional(self, h1, eri, ecore):
        """Energy as (vector over variables, constant)."""
        n, ns = self.n, self.ns; w = np.zeros(self.nvar)
        g = lambda i, k: self.glin(i, k)
        for p in range(n):
            for q in range(n):
                if abs(h1[p, q]) > 1e-14:
                    for s in range(2):
                        for v, c in g(2 * p + s, 2 * q + s).items():
                            w[v] += h1[p, q] * c
        for p in range(n):
            for q in range(n):
                for r in range(n):
                    for t in range(n):
                        val = eri[p, q, r, t]
                        if abs(val) < 1e-14:
                            continue
                        for s1 in range(2):
                            for s2 in range(2):
                                for v, c in self.Dlin(2 * p + s1, 2 * r + s2, 2 * q + s1, 2 * t + s2).items():
                                    w[v] += 0.5 * val * c
        return w, ecore

    def s2_functional(self):
        w = np.zeros(self.nvar); n = self.n
        for p in range(n):
            for q in range(n):
                for a in range(2):
                    for a2 in range(2):
                        for b in range(2):
                            for b2 in range(2):
                                m = MSS[a, a2, b, b2]
                                if m == 0:
                                    continue
                                for v, c in self.Dlin(2 * p + a, 2 * q + b, 2 * p + a2, 2 * q + b2).items():
                                    w[v] += m * c
        return w, 0.75 * self.N


def spin_orbital_D(ci, n, nelec):
    _, (aa, ab, bb) = fci.direct_spin1.make_rdm12s(ci, n, nelec)
    ns = 2 * n; Dt = np.zeros((ns,) * 4)
    # pyscf dm2[p,q,r,s] = <p+ r+ s q>  ->  D[(p s1)(r s2),(q s1)(s s2)]
    for (blk, s1, s2) in ((aa, 0, 0), (bb, 1, 1), (ab, 0, 1)):
        for p in range(n):
            for q in range(n):
                for r in range(n):
                    for s in range(n):
                        v = blk[p, q, r, s]
                        if v == 0:
                            continue
                        i, j, k, l = 2 * p + s1, 2 * r + s2, 2 * q + s1, 2 * s + s2
                        if i == j or k == l:
                            continue
                        Dt[i, j, k, l] = v; Dt[j, i, l, k] = v      # full antisymmetric fill
                        Dt[j, i, k, l] = -v; Dt[i, j, l, k] = -v
    return Dt


def run(label, h1, eri, ecore, n, nelec, ci, e0, gaps):
    t0 = time.time(); S = 0.5 * (nelec[0] - nelec[1])
    M = Model(n, *nelec); mats = M.matrices()
    wE, cE = M.functional(h1, eri, ecore); wS, cS = M.s2_functional()
    # ---- validation on the exact CASCI 2-RDM
    Dt = spin_orbital_D(ci, n, nelec); x0 = M.vec_from_D(Dt)
    eE = wE @ x0 + cE; eS = wS @ x0 + cS
    mineig = min(np.linalg.eigvalsh((A @ x0 + b).reshape(m, m)).min() for _, A, b, m in mats)
    print(f"{label}: nvar {M.nvar}; check E {eE - e0:+.1e} Eh, <S2> {eS - S*(S+1):+.1e}, min eig of exact D/Q/G {mineig:+.1e}", flush=True)
    assert abs(eE - e0) < 1e-8 and abs(eS - S * (S + 1)) < 1e-8 and mineig > -1e-8
    # ---- SDP
    x = cp.Variable(M.nvar); cons = []
    for name, A, b, m in mats:
        Y = cp.reshape(A @ x + b, (m, m), order='C')
        cons.append(0.5 * (Y + Y.T) >> 0)
    for c, L in M.blocks.items():
        tr = sum(M.var[(c, a, a)] for a in range(len(L)))
        ids = [M.var[(c, a, a)] for a in range(len(L))]
        target = {0: nelec[0] * (nelec[0] - 1) / 2, 1: nelec[0] * nelec[1], 2: nelec[1] * (nelec[1] - 1) / 2}[c]
        cons.append(cp.sum(x[ids]) == target)
    cons.append(wS @ x + cS == S * (S + 1))
    prob = cp.Problem(cp.Minimize(wE @ x + cE - e0), cons)       # objective relative to E0 for precision
    if n < 10:     # interior point (Clarabel)
        prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-8, tol_gap_rel=1e-8, tol_feas=1e-8, max_iter=500)
        gapLB = -prob.value; extra = {}
    else:          # 20 qubits: interior point needs a dense 20100^2 system per iteration; use first-order SCS
        prob.solve(solver=cp.SCS, eps_abs=1e-7, eps_rel=1e-7, max_iters=400000, verbose=False)
        info = prob.solver_stats.extra_stats.get('info', {}) if prob.solver_stats.extra_stats else {}
        gapLB = -prob.value
        extra = {k: info.get(k) for k in ('pobj', 'dobj', 'res_pri', 'res_dual', 'gap', 'iter')}
        if info.get('dobj') is not None:
            extra['E0_minus_dobj_mEh'] = -1e3 * info['dobj']
    st = prob.solver_stats
    res = dict(status=prob.status, solver=st.solver_name, E0_minus_ELB_mEh=1e3 * gapLB, time_s=time.time() - t0, nvar=M.nvar,
               iters=st.num_iters, **extra)
    for k, G in gaps.items():                                      # H2 check at chemical accuracy (1.6 mEh)
        res[f'margin_{k}_mEh'] = 1e3 * (G / 2 - 1.6e-3 - gapLB)
        res[f'half_{k}_mEh'] = 1e3 * G / 2
    print(f"   {prob.status}: E0 - E_LB = {1e3*gapLB:.2f} mEh; " +
          '; '.join(f"Delta/2 ({k}) = {1e3*G/2:.1f} mEh -> margin {res[f'margin_{k}_mEh']:+.1f}" for k, G in gaps.items()) +
          f"  [{res['time_s']:.0f} s]", flush=True)
    return res


def sector_gaps(h1, eri, ecore, n, nelec, mu_f=0.05, nroots=8):
    S = 0.5 * (nelec[0] - nelec[1])
    e, c = fci.direct_spin1.kernel(h1, eri, n, nelec, ecore=ecore, nroots=nroots, conv_tol=1e-11, max_space=40)
    s2 = np.array([fci.spin_op.spin_square0(v, n, nelec)[0] for v in c])
    E0 = e[0]; assert abs(s2[0] - S * (S + 1)) < 1e-5
    GM = e[1] - E0
    ep = e + mu_f * (s2 - S * (S + 1)); Gp = np.sort(ep)[1] - E0
    return E0, c[0], {'G_M (H route)': GM, "G' (H' route)": Gp}


if __name__ == '__main__':
    import os
    out = json.load(open('results/v2rdm_bound.json')) if os.path.exists('results/v2rdm_bound.json') else {}
    names = sys.argv[1:] or ['CH2', 'NH', 'NH_stretch', 'O2', 'TMM', 'naph_no', 'naph_rohf']
    for name in names:
        if name.startswith('naph'):
            d = np.load('results/naph_setup_no.npz' if name == 'naph_no' else 'results/naph_setup.npz')
            h1, eri, ecore, n, nelec = d['h1'], d['eri'], float(d['ecore']), 10, (6, 4)
        else:
            mol, mc, F, S, n, nelec, h1, ecore, eri = setup(name)
        E0, ci, gaps = sector_gaps(h1, eri, ecore, n, nelec)
        out[name] = run(name, h1, eri, ecore, n, nelec, ci, E0, gaps)
        json.dump(out, open('results/v2rdm_bound.json', 'w'), indent=1)
