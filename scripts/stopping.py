"""Test an exact-answer-free stopping rule for D on every ADAPT trajectory.

Window rule: stop at the first iterate n >= m with max_{k in [n-m, n]} |D_k - D_n| < 1% of |D_n|.
Scored by the true error of D at the stopping iterate (relative to the exact value in the same active space).
Uses results/errbar_*.json (12-qubit systems and naphthalene in natural orbitals) and the histories of the
naphthalene runs in ROHF orbitals. Writes results/stopping.json.
"""
import json
import os
import numpy as np


def window_stop(N, D, Dex, m, rel=0.01):
    """N: operator counts of the recorded iterates (consecutive); returns (operators at stop, true error %)."""
    D = np.asarray(D)
    for i in range(m, len(D)):
        if np.abs(D[i - m:i + 1] - D[i]).max() < rel * abs(D[i]):
            return int(N[i]), 100 * abs(D[i] - Dex) / abs(Dex)
    return None, None


def trajectories():
    for name in ['CH2', 'NH', 'NH_stretch', 'O2']:          # TMM reaches the exact state in <= 5 operators
        for p in ['GSD', 'QEB', 'sfGSD']:
            rows = json.load(open(f'results/errbar_{name}_{p}.json'))
            D = [r['D'] for r in rows]
            Dex = json.load(open(f'results/{name}.json'))['D_CASCI']
            yield name, p, list(range(len(D))), D, Dex
    ref = json.load(open('results/naph_reference.json'))['D_CASCI']
    for p in ['GSD', 'QEB', 'sfGSD']:
        h = json.load(open(f'results/naph_{p}.json'))['history']
        yield 'naph_rohf', p, [r['n'] for r in h], [r['D'] for r in h], ref
    ref = json.load(open('results/naph_reference_no.json'))['D_exact_NO']
    for p in ['GSD', 'QEB', 'sfGSD']:
        f = f'results/naph_no_{p}.json'
        if os.path.exists(f):
            h = json.load(open(f))['history']
            yield 'naph_no', p, [r['n'] for r in h], [r['D'] for r in h], ref


if __name__ == '__main__':
    out = {}
    for name, p, N, D, Dex in trajectories():
        out.setdefault(name, {})[p] = {}
        line = f'{name:10s} {p:6s}'
        for m in (5, 10, 20, 30):
            n, e = window_stop(N, D, Dex, m)
            out[name][p][f'm{m}'] = {'n_stop': n, 'true_err_pct': e}
            line += f'  m={m}: ' + (f'n={n:3d} err={e:5.2f}%' if n is not None else 'no stop        ')
        print(line)
    json.dump(out, open('results/stopping.json', 'w'), indent=1)
