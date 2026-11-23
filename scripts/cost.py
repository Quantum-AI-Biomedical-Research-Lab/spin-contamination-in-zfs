"""Cost proxies: parameters and spin-orbital excitation terms needed to reach
chemical accuracy and |D-D_ref|<1% (staying there)."""
import json, numpy as np, scipy.sparse as sp
from vqe import CISpace
from run import SYSTEMS

def n_components(space, lab):
    if lab[0] in ('s', 'dss', 'dab'):
        return 1
    if lab[0] == 'E':
        return 2
    _, p, q, r, s = lab
    mats = [space.double_same(p, r, s, q, 0), space.double_same(p, r, s, q, 1),
            space.double_ab(p, q, r, s), space.double_ab(r, s, p, q)]
    keys, cnt = set(), 0
    for M in mats:
        A = (M - M.T).tocsr(); A.eliminate_zeros()
        if A.nnz == 0: continue
        k = (tuple(A.indices), tuple(np.round(A.data, 9)))
        if k in keys: continue
        keys.add(k); cnt += 1
    return cnt

out = {}
for name in ['CH2', 'NH', 'NH_stretch', 'O2', 'TMM']:
    o = json.load(open(f'results/{name}.json'))
    _, ncas, nelecas, _ = SYSTEMS[name]
    space = CISpace(ncas, tuple(nelecas))
    Dr = o['D_CASCI']; out[name] = {}
    for p, r in o['runs'].items():
        H = r['history']; ops = r['ops']
        comps = np.cumsum([0] + [n_components(space, tuple(op)) for op in ops])
        nchem = next(h['n'] for h in H if h['dE'] < 1.6e-3)
        nD = next(h['n'] for i, h in enumerate(H) if all(abs(g['D'] - Dr) / abs(Dr) < 0.01 for g in H[i:]))
        out[name][p] = {'n_chem': nchem, 'exc_chem': int(comps[nchem]), 'n_D1': nD, 'exc_D1': int(comps[nD]),
                        'n_final': H[-1]['n'], 'exc_final': int(comps[H[-1]['n']])}
        print(name, p, out[name][p])
json.dump(out, open('results/cost.json', 'w'), indent=1)
