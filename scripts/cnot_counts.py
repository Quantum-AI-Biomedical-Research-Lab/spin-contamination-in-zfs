"""CNOT counts of the ADAPT ansätze with a standard Pauli-gadget compilation (Jordan-Wigner, interleaved spin
orbitals): each Pauli string of weight w in a generator costs 2(w-1) CNOTs.

GSD and QEB generators consist of mutually commuting Pauli strings, so this compiles their exponentials exactly.
sfGSD generators are sums of up to four spin-orbital excitations that need not commute; the count corresponds to one
product of their exponentials (a first-order Trotter step, which breaks S^2 slightly) and is a lower estimate for an
exact, S^2-preserving implementation.
"""
import json
import sys
import numpy as np
import openfermion as of


def qraise(i):
    return of.QubitOperator(f'X{i}', 0.5) + of.QubitOperator(f'Y{i}', -0.5j)


def qlower(i):
    return of.QubitOperator(f'X{i}', 0.5) + of.QubitOperator(f'Y{i}', 0.5j)


def fermion_T(label):
    k = label[0]
    if k == 's':
        _, sp, p, q = label
        return of.FermionOperator(((2 * p + sp, 1), (2 * q + sp, 0)))
    if k == 'dss':
        _, sp, p, r, q, s = label
        return of.FermionOperator(((2 * p + sp, 1), (2 * r + sp, 1), (2 * s + sp, 0), (2 * q + sp, 0)))
    if k == 'dab':
        _, p, r, q, s = label
        return of.FermionOperator(((2 * p, 1), (2 * q, 0), (2 * r + 1, 1), (2 * s + 1, 0)))
    if k == 'E':
        _, p, q = label
        return sum(of.FermionOperator(((2 * p + sp, 1), (2 * q + sp, 0))) for sp in (0, 1))
    if k == 'e':
        _, p, q, r, s = label
        T = of.FermionOperator()
        if p != r and q != s:
            for sp in (0, 1):
                T += of.FermionOperator(((2 * p + sp, 1), (2 * r + sp, 1), (2 * s + sp, 0), (2 * q + sp, 0)))
        T += of.FermionOperator(((2 * p, 1), (2 * q, 0), (2 * r + 1, 1), (2 * s + 1, 0)))
        T += of.FermionOperator(((2 * r, 1), (2 * s, 0), (2 * p + 1, 1), (2 * q + 1, 0)))
        return T
    raise ValueError(label)


def qubit_T(label):
    k = label[0]
    if k == 's':
        _, sp, p, q = label
        return qraise(2 * p + sp) * qlower(2 * q + sp)
    if k == 'dss':
        _, sp, p, r, q, s = label
        return qraise(2 * p + sp) * qraise(2 * r + sp) * qlower(2 * s + sp) * qlower(2 * q + sp)
    if k == 'dab':
        _, p, r, q, s = label
        return qraise(2 * p) * qlower(2 * q) * qraise(2 * r + 1) * qlower(2 * s + 1)
    raise ValueError(label)


def cnots(label, qubit):
    if qubit:
        T = qubit_T(label); A = T - of.hermitian_conjugated(T)
    else:
        T = fermion_T(label); A = of.jordan_wigner(T - of.hermitian_conjugated(T))
    A.compress(1e-12)
    return sum(2 * (len(t) - 1) for t in A.terms if len(t) > 0), len(A.terms)


def ansatz_counts(ops, qubit, upto):
    total = 0
    for lab in ops[:upto]:
        c, _ = cnots(tuple(lab), qubit)
        total += c
    return total


if __name__ == '__main__':
    out = {}
    CHEM = 1.6e-3
    sets = [(n, f'results/{n}.json', 'D_CASCI') for n in ['CH2', 'NH', 'NH_stretch', 'O2', 'TMM']]
    sets += [('naph_rohf', 'results/naph_{p}.json', None), ('naph_no', 'results/naph_no_{p}.json', None)]
    for name, pattern, _ in sets:
        out[name] = {}
        for p in ['GSD', 'QEB', 'sfGSD']:
            f = pattern.format(p=p) if '{p}' in pattern else pattern
            try:
                o = json.load(open(f))
            except FileNotFoundError:
                continue
            run = o['runs'][p] if 'runs' in o else o
            if 'ops' not in run:
                continue
            H = run['history']; ops = run['ops']
            Eex = o.get('E_CASCI', o.get('E_exact'))
            Dex = o['D_CASCI'] if 'D_CASCI' in o else None
            if Dex is None:
                ref = json.load(open('results/naph_reference.json' if name == 'naph_rohf' else 'results/naph_reference_no.json'))
                Dex = ref['D_CASCI'] if name == 'naph_rohf' else ref['D_exact_NO']
            dE = [h['E'] - Eex for h in H]
            n_chem = next((h['n'] for h, d in zip(H, dE) if d < CHEM), None)
            n_D = next((h['n'] for i, h in enumerate(H) if all(abs(g['D'] - Dex) / abs(Dex) < 0.01 for g in H[i:])), None)
            q = (p == 'QEB')
            out[name][p] = {'n_chem': n_chem, 'cnot_chem': ansatz_counts(ops, q, n_chem) if n_chem else None,
                            'n_D1': n_D, 'cnot_D1': ansatz_counts(ops, q, n_D) if n_D else None,
                            'cnot_per_op_mean': ansatz_counts(ops, q, len(ops)) / max(len(ops), 1)}
            print(name, p, out[name][p], flush=True)
    json.dump(out, open('results/cnot_counts.json', 'w'), indent=1)
