"""Robustness of the spin penalty mu = 3 Eh: 10 runs with selection noise dE = 0.16 mEh, GSD and QEB.
Usage (from the repository root): python scripts/penalty_noise3.py [systems...]    writes results/penalty_noise3_<system>.json"""
import json, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import remedies, penalty
from vqe import pool_gsd
for name in (sys.argv[1:] or ['CH2', 'NH', 'NH_stretch', 'O2']):
    sysd = remedies.build(name); space = sysd['space']
    S2 = penalty.spin_matrix(space, space.norb, space.nelec)
    ws, Vs = np.linalg.eigh(S2); PT = Vs[:, np.abs(ws - 2.0) < 1e-6]
    out = {'D_exact': sysd['Dex']}
    for pname in ['GSD', 'QEB']:
        pool = pool_gsd(space, qubit=(pname == 'QEB'))
        ms = [penalty.metrics(penalty.run(sysd, S2, PT, pool, pname, 3.0, 0.0, noise=0.16e-3,
                                          rng=np.random.default_rng(3000 + s))['rows'], sysd['Dex']) for s in range(10)]
        out[pname] = ms
        e = [m['err'] for m in ms]; c = [m['cnot_D1'] for m in ms if m['cnot_D1'] is not None]
        print(f"{name:10s} {pname:4s} mu=3 noise: err [{min(e):+.3f},{max(e):+.3f}] D1 cnot [{min(c)},{max(c)}] ({len(c)}/10)", flush=True)
    json.dump(out, open(f'results/penalty_noise3_{name}.json', 'w'))
