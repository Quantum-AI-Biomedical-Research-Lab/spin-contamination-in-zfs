"""Gaussian spectral filter g(H) = exp[-(H - mu)^2 T^2 / 2], mu = E(psi), on the stored 20-qubit naphthalene ADAPT
states in natural orbitals at the first iterate with chemical accuracy (GSD n = 291, sfGSD n = 191).
g(H) psi is evaluated in a Lanczos basis with full reorthogonalization (m = 300; checked against m = 200).
Cost ingredients: Pauli 1-norm lam_H of the JW Hamiltonian, and for the D operator X_zz the QWC groups with
sum_g sigma_g and sum_g sqrt(<X_g^2>) on the exact triplet (statevector Pauli application, 20 qubits).
Writes results/naph_filter.json."""
import json
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import openfermion as of
from pyscf import fci
import vqe_sparse as sps
from xop import XOperator
from fockmap import d_prefactor, hamiltonian_op, d_component_op, pyscf_to_fock
from zfs import HARTREE2CM

TS = [10, 20, 30, 40, 50, 60, 80, 'rule']
MU_F = 0.05                                    # spin shift (Eh) for the 'rule_spin' filter
t0 = time.time()
dat = np.load('results/naph_setup_no.npz'); ncas, nelecas = 10, (6, 4)
space = sps.SparseSpace(ncas, nelecas); shape = (space.na, space.nb)
Hs = sps.Hamiltonian(dat['h1'], dat['eri'], float(dat['ecore']), ncas, nelecas)
Hm = lambda v: Hs(v, shape)
Fp = np.einsum('ai,bj,ab...->ij...', dat['V'], dat['V'], dat['F'])
X = XOperator(ncas, nelecas, Fp[2, 2]); cD = 1.5 * d_prefactor(1.0) * HARTREE2CM
T = dat['ci'].ravel(); T = T / np.linalg.norm(T)
Dex = cD * (T @ X(T)); E0 = float(dat['e_casci']); E1 = float(dat['e1'])
S2 = lambda v: fci.spin_op.contract_ss(v.reshape(shape), ncas, nelecas).ravel()


def lanczos(psi, m, Hop=None):
    Hop = Hop or Hm
    n = psi.size; Q = np.zeros((m, n)); a = np.zeros(m); b = np.zeros(m)
    q = psi / np.linalg.norm(psi)
    for j in range(m):
        Q[j] = q; w = Hop(q); a[j] = q @ w
        w -= Q[:j + 1].T @ (Q[:j + 1] @ w); w -= Q[:j + 1].T @ (Q[:j + 1] @ w)
        if j + 1 < m:
            b[j] = np.linalg.norm(w); q = w / b[j]
    Tm = np.diag(a) + np.diag(b[:m - 1], 1) + np.diag(b[:m - 1], -1)
    return Q, Tm


out = {'D_exact': Dex, 'gap_E1': E1 - E0, 'runs': {}}
for p, n in [('GSD', 291), ('sfGSD', 191)]:
    psi = np.array(np.load(f'results/naph_no_{p}_states.npy', mmap_mode='r')[n]); psi /= np.linalg.norm(psi)
    Epsi = float(psi @ Hm(psi))
    res = {'dE_mEh': 1e3 * (Epsi - E0), 'D': 100 * (cD * psi @ X(psi) - Dex) / Dex, 'gauss': {}}
    Qb, Tm = lanczos(psi, 300)
    th, U = np.linalg.eigh(Tm)
    th2, U2 = np.linalg.eigh(Tm[:200, :200])
    for TT in TS:
        Tv = 3.0 / (E1 - E0) if TT == 'rule' else TT
        y = Qb.T @ (U @ (np.exp(-0.5 * ((th - Epsi) * Tv) ** 2) * U[0]))
        y2 = Qb[:200].T @ (U2 @ (np.exp(-0.5 * ((th2 - Epsi) * Tv) ** 2) * U2[0]))
        Ng = float(y @ y); y /= np.sqrt(Ng); y2 /= np.linalg.norm(y2)
        res['gauss'][TT] = {'T': Tv, 'D': float(100 * (cD * y @ X(y) - Dex) / Dex), 'Ng': Ng,
                            'dE_mEh': float(1e3 * (y @ Hm(y) - E0)), 'S2dev': float(y @ S2(y) - 2),
                            'lanczos_check': float(1 - abs(y @ y2))}
    Qs, Ts = lanczos(psi, 300, lambda v: Hm(v) + MU_F * (S2(v) - 2 * v))
    ths, Us = np.linalg.eigh(Ts); Tv = 3.0 / (E1 - E0)
    y = Qs.T @ (Us @ (np.exp(-0.5 * ((ths - Epsi) * Tv) ** 2) * Us[0])); Ng = float(y @ y); y /= np.sqrt(Ng)
    res['gauss']['rule_spin'] = {'T': Tv, 'D': float(100 * (cD * y @ X(y) - Dex) / Dex), 'Ng': Ng,
                                 'dE_mEh': float(1e3 * (y @ Hm(y) - E0)), 'S2dev': float(y @ S2(y) - 2)}
    del Qs
    out['runs'][p] = res
    print(p, f"D {res['D']:+.3f}% dE {res['dE_mEh']:.2f} mEh |", ' '.join(
        f"T{TT}:{res['gauss'][TT]['D']:+.3f}%({res['gauss'][TT]['dE_mEh']:.3f}mEh,Ng={res['gauss'][TT]['Ng']:.4f},chk={res['gauss'][TT]['lanczos_check']:.0e})"
        for TT in TS), f"t={time.time()-t0:.0f}s", flush=True)
json.dump(out, open('results/naph_filter.json', 'w'), indent=1)

# ---- cost ingredients
Hq = of.jordan_wigner(hamiltonian_op(dat['h1'], dat['eri'], float(dat['ecore']), ncas)); Hq.compress(1e-12)
Hq.terms.pop((), None); lamH = float(sum(abs(v) for v in Hq.terms.values()))
print(f"lam_H = {lamH:.2f} Eh ({len(Hq.terms)} Pauli terms)  t={time.time()-t0:.0f}s", flush=True)
Xq = of.jordan_wigner(d_component_op(Fp[2, 2], ncas, 1.0)); Xq.compress(1e-14)
X0 = float(np.real(Xq.terms.pop((), 0.0)))
nq = 2 * ncas
psiF = np.asarray(pyscf_to_fock(dat['ci'], ncas, nelecas)).real; psiF /= np.linalg.norm(psiF)
idx = np.arange(2 ** nq, dtype=np.int64)


def apply_pauli(term, v):
    xm = zm = 0; ny = 0
    for q, s in term:
        bit = 1 << (nq - 1 - q)
        if s in 'XY': xm |= bit
        if s in 'ZY': zm |= bit
        if s == 'Y': ny += 1
    j = idx ^ xm                                      # (P v)[i] = phase(j) v[j], j = i ^ xm, P|j> = phase(j)|i>
    sign = 1 - 2 * (np.bitwise_count(j & zm).astype(np.int64) & 1)
    return (1j ** ny) * sign * v[j]


groups = of.group_into_tensor_product_basis_sets(Xq, seed=0)
Xmean = X0; sig = sec = 0.0
for _, gop in groups.items():
    Gv = np.zeros(psiF.size, dtype=complex)
    for term, c in gop.terms.items():
        Gv += c * apply_pauli(term, psiF)
    m1 = float(np.real(psiF @ Gv)); m2 = float(np.real(np.vdot(Gv, Gv)))
    Xmean += m1; sig += np.sqrt(max(m2 - m1 * m1, 0)); sec += np.sqrt(m2)
out['cost'] = {'lam_H': lamH, 'H_terms': len(Hq.terms), 'X_mean': Xmean, 'X_identity': X0,
               'X_l1': float(sum(abs(v) for v in Xq.terms.values())), 'X_terms': len(Xq.terms), 'X_groups': len(groups),
               'sum_sigma_g': sig, 'sum_sqrt_m2_g': sec, 'D_from_X': cD * Xmean}
print(f"X: <X>={Xmean:.5g} (D={cD*Xmean:.5f}, exact {Dex:.5f}) terms={len(Xq.terms)} groups={len(groups)} "
      f"l1={out['cost']['X_l1']:.4g} sum sigma_g={sig:.4g} sum sqrt<Xg^2>={sec:.4g}  t={time.time()-t0:.0f}s", flush=True)
json.dump(out, open('results/naph_filter.json', 'w'), indent=1)
