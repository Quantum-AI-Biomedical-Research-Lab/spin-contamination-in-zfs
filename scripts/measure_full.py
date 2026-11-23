"""Shot cost of the full D tensor (five independent components) versus one principal-axis component.

Target: every Cartesian component D_ab (molecular frame) to 1% of |D|. Two schemes, both with QWC grouping and
optimal allocation, evaluated on the exact active-space triplet:
  separate  each independent component (xx, yy, xy, xz, yz; zz = -xx - yy) measured on its own;
  shared    one QWC grouping of the union of all Pauli strings; group g gets N a_g shots with a_g ∝ max_c sigma_gc,
            and N is the smallest total for which every component reaches the target.
"""
import json
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import openfermion as of
from fockmap import pyscf_to_fock, d_component_op, d_prefactor
from zfs import d_tensor, HARTREE2CM
from pyscf import fci
from theory_check import setup as _setup


def setup(name):
    return _setup(name)[:6]


def group_sigmas(qops, psi, nq):
    """QWC-group the union of strings; return sigma[g, c] for each group g and component c."""
    union = of.QubitOperator()
    for q in qops:
        for t in q.terms:
            if t != ():
                union += of.QubitOperator(t, 1.0)
    groups = of.group_into_tensor_product_basis_sets(union, seed=0)
    sig = np.zeros((len(groups), len(qops)))
    for gi, (_, g) in enumerate(groups.items()):
        for ci, q in enumerate(qops):
            part = of.QubitOperator()
            for t in g.terms:
                if t in q.terms:
                    part += of.QubitOperator(t, q.terms[t])
            if not part.terms:
                continue
            G = of.get_sparse_operator(part, n_qubits=nq)
            Gp = G @ psi
            sig[gi, ci] = np.sqrt(max((np.vdot(Gp, Gp) - np.vdot(psi, Gp) ** 2).real, 0))
    return sig, len(groups), len(union.terms)


out = {}
for name in ['CH2', 'NH', 'NH_stretch', 'O2', 'TMM']:
    mol, mc, F, S, ncas, nelecas = setup(name)
    nq = 2 * ncas
    psi = pyscf_to_fock(mc.ci, ncas, nelecas)
    pref = d_prefactor(S) * HARTREE2CM                       # cm^-1 per unit <X_ab>
    _, d2 = fci.direct_spin1.make_rdm12s(mc.ci, ncas, nelecas)
    w, V = np.linalg.eigh(d_tensor(F, d2, S)); V = V[:, np.argsort(np.abs(w))]
    Dval = abs(1.5 * (V.T @ d_tensor(F, d2, S) @ V)[2, 2] * HARTREE2CM)
    eps = 0.01 * Dval / pref                                 # target in units of <X_ab>
    comps = [(0, 0), (1, 1), (0, 1), (0, 2), (1, 2)]
    qops = [of.jordan_wigner(d_component_op(F[a, b], ncas, S)) for a, b in comps]
    for q in qops:
        q.terms.pop((), None)
    sig, ngroups, nterms = group_sigmas(qops, psi, nq)
    # separate: each component grouped on its own
    sep = 0.0
    for q in qops:
        s1, _, _ = group_sigmas([q], psi, nq)
        sep += (s1[:, 0].sum() / eps) ** 2
    # shared
    a = sig.max(axis=1); a = a / a.sum()
    mask = a > 0
    N_shared = max(((sig[mask, c] ** 2) / a[mask]).sum() / eps ** 2 for c in range(len(qops)))
    # single principal-axis component (known principal axes)
    Fp = np.einsum('ai,bj,ab...->ij...', V, V, F)
    qz = of.jordan_wigner(d_component_op(Fp[2, 2], ncas, S)); qz.terms.pop((), None)
    sz, _, _ = group_sigmas([qz], psi, nq)
    eps_z = 0.01 * Dval / (1.5 * pref)
    N_axis = (sz[:, 0].sum() / eps_z) ** 2
    out[name] = {'D_cm': Dval, 'N_principal_axis': N_axis, 'N_full_separate': sep, 'N_full_shared': N_shared,
                 'union_terms': nterms, 'union_groups': ngroups,
                 'ratio_shared_to_axis': N_shared / N_axis, 'ratio_separate_to_axis': sep / N_axis}
    print(f"{name:10s} principal-axis {N_axis:.2e}  full tensor separate {sep:.2e}  shared {N_shared:.2e} "
          f"({nterms} strings, {ngroups} groups)  shared/axis {N_shared / N_axis:.1f}", flush=True)
json.dump(out, open('results/measure_full.json', 'w'), indent=1, default=float)
