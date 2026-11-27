"""(i) Controlled contamination test of the first-order (sqrt) sensitivity of D.
(ii) Shot-cost estimates for measuring D vs the energy on the exact active-space triplet."""
import json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import openfermion as of
from pyscf import gto, scf, mcscf, fci, ao2mo
from zfs import ss_mo_ints, d_tensor, HARTREE2CM
from fockmap import pyscf_to_fock, hamiltonian_op, d_component_op, d_prefactor
from run import SYSTEMS, select_pi_orbitals, axis_gauge
from vqe import CISpace


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
    return mol, mc, F, S, ncas, nelecas


def principal(F, mc, ncas, nelecas, S):
    _, d2 = fci.direct_spin1.make_rdm12s(mc.ci, ncas, nelecas)
    D = d_tensor(F, d2, S)
    w, V = np.linalg.eigh(D)
    V = V[:, np.argsort(np.abs(w))]
    Fp = np.einsum('ai,bj,ab...->ij...', V, V, F)   # integrals in the principal frame
    return D, V, Fp


def shots(op_sparse_terms, psi, eps, nq):
    """Return (no-grouping, QWC-grouped) optimal-allocation shot counts for <O> to precision eps."""
    qop = of.jordan_wigner(op_sparse_terms)
    qop.compress(1e-14)
    const = qop.terms.pop((), 0.0)
    tot = 0.0
    for term, c in qop.terms.items():
        P = of.get_sparse_operator(of.QubitOperator(term), n_qubits=nq)
        ev = (psi @ (P @ psi)).real
        tot += abs(c) * np.sqrt(max(1 - ev * ev, 0.0))
    n_nogroup = (tot / eps) ** 2
    groups = of.group_into_tensor_product_basis_sets(qop, seed=0)
    tg = 0.0
    for _, g in groups.items():
        G = of.get_sparse_operator(g, n_qubits=nq)
        Gp = G @ psi
        var = (np.vdot(Gp, Gp) - np.vdot(psi, Gp) ** 2).real
        tg += np.sqrt(max(var, 0.0))
    n_group = (tg / eps) ** 2
    return n_nogroup, n_group, len(qop.terms), len(groups)


if __name__ == '__main__':
    out = {}
    for name in ['CH2', 'NH', 'NH_stretch', 'O2', 'TMM']:
        mol, mc, F, S, ncas, nelecas = setup(name)
        D, V, Fp = principal(F, mc, ncas, nelecas, S)
        Dval = 1.5 * (V.T @ D @ V)[2, 2]
        nq = 2 * ncas
        psi = pyscf_to_fock(mc.ci, ncas, nelecas)
        res = {'D_cm': Dval * HARTREE2CM}

        # ---- (i) controlled contamination: mix in the nearest wrong-spin state along a GSD direction
        space = CISpace(ncas, nelecas)
        na, nb = len(space.sa), len(space.sb)
        S2m = np.zeros((space.dim, space.dim))
        for i in range(space.dim):
            e = np.zeros(space.dim); e[i] = 1
            S2m[:, i] = fci.spin_op.contract_ss(e.reshape(na, nb), ncas, nelecas).ravel()
        ws, Vs = np.linalg.eigh(0.5 * (S2m + S2m.T))
        Vq = Vs[:, np.abs(ws - 6.0) < 1e-6]                  # quintet subspace (S=2)
        if Vq.shape[1] > 0:
            # the quintet direction that maximally couples to D through the exact triplet
            Xop = d_component_op(Fp[2, 2], ncas, S)
            Xs = of.get_sparse_operator(Xop, n_qubits=nq)
            # build quintet basis vectors in Fock space
            QF = np.array([pyscf_to_fock(Vq[:, k].reshape(na, nb), ncas, nelecas) for k in range(Vq.shape[1])]).T
            cpl = QF.T @ (Xs @ psi).real                         # <Q_k | X_zz | T>
            q = QF @ (cpl / np.linalg.norm(cpl))
            eps_list = np.logspace(-4, -1, 7)
            scan = []
            for e_ in eps_list:
                ph = np.sqrt(1 - e_ ** 2) * psi + e_ * q
                Dz = 1.5 * d_prefactor(S) * (ph @ (Xs @ ph)).real * HARTREE2CM
                s2dev = 4 * e_ ** 2                               # <S^2> - 2 for triplet+quintet mix
                scan.append((s2dev, abs(Dz - Dval * HARTREE2CM) / abs(Dval * HARTREE2CM)))
            s = np.array(scan)
            slope = np.polyfit(np.log10(s[:, 0]), np.log10(s[:, 1]), 1)[0]
            res['controlled_scan'] = scan
            res['controlled_slope'] = slope
            res['rel_shift_at_S2dev_1e-3'] = float(np.interp(-3, np.log10(s[:, 0]), np.log10(s[:, 1])))
            res['rel_shift_at_S2dev_1e-3'] = 10 ** res['rel_shift_at_S2dev_1e-3']
            print(f"{name}: controlled slope = {slope:.4f}; |dD|/D at <S2>-2=1e-3: {100*res['rel_shift_at_S2dev_1e-3']:.2f}%")

        # ---- (ii) shot counts
        h1, ecore = mc.get_h1eff()
        eri = ao2mo.restore(1, mc.get_h2eff(), ncas)
        Hop = hamiltonian_op(h1, eri, ecore, ncas)
        nH = shots(Hop, psi, 1.6e-3, nq)
        # D operator in the principal frame: D = 3/2 * pref * X_zz ; target 1% of |D|
        XD = d_component_op(Fp[2, 2], ncas, S)
        pref = 1.5 * d_prefactor(S) * HARTREE2CM                # cm^-1 per unit <X_zz>
        eps_X = 0.01 * abs(Dval * HARTREE2CM) / abs(pref)
        nD = shots(XD, psi, eps_X, nq)
        eps_X2 = 0.001 / abs(pref)                               # absolute 0.001 cm^-1
        nD2 = shots(XD, psi, eps_X2, nq)
        res.update({'shots_H_chem': nH[:2], 'nterms_H': nH[2], 'ngroups_H': nH[3],
                    'shots_D_1pct': nD[:2], 'shots_D_1e-3cm': nD2[:2], 'nterms_D': nD[2], 'ngroups_D': nD[3]})
        print(f"   H: terms={nH[2]} groups={nH[3]} shots(1.6 mEh) nogroup={nH[0]:.2e} QWC={nH[1]:.2e}")
        print(f"   D: terms={nD[2]} groups={nD[3]} shots(1% D) nogroup={nD[0]:.2e} QWC={nD[1]:.2e};"
              f" shots(0.001 cm-1) QWC={nD2[1]:.2e}", flush=True)
        out[name] = res
    json.dump(out, open('results/measure.json', 'w'), indent=1, default=float)
