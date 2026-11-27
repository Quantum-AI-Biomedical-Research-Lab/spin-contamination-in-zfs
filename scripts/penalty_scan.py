"""Spin-penalty scan and robustness check for penalized ADAPT (penalty.py).

  scan   mu in {0.03, 0.1, 0.3, 1, 3} Eh with GSD and QEB (w = 0), plus the unpenalized baseline A
  noise  10 runs each of A and S0.3 with selection noise dE = 0.16 mEh (as in spread.py), GSD and QEB
Usage (from the repository root): python scripts/penalty_scan.py [systems...]      writes results/penalty_scan_<system>.json
"""
import json
import sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import remedies
import penalty
from vqe import pool_gsd

MUS = [0.03, 0.1, 0.3, 1.0, 3.0]
NSEED, NOISE = 10, 0.16e-3

for name in (sys.argv[1:] or ['CH2', 'NH', 'NH_stretch', 'O2']):
    sysd = remedies.build(name); space = sysd['space']
    S2 = penalty.spin_matrix(space, space.norb, space.nelec)
    ws, Vs = np.linalg.eigh(S2); PT = Vs[:, np.abs(ws - 2.0) < 1e-6]
    out = {'D_exact': sysd['Dex'], 'scan': {}, 'noise': {}}
    for pname in ['GSD', 'QEB']:
        pool = pool_gsd(space, qubit=(pname == 'QEB'))
        for mu in [0.0] + MUS:
            r = penalty.run(sysd, S2, PT, pool, pname, mu, 0.0)
            m = penalty.metrics(r['rows'], sysd['Dex']); r['metrics'] = m
            out['scan'][f'{pname}_mu{mu}'] = r
            print(f"{name:10s} {pname:4s} mu={mu:<5} n_chem={m['n_chem']} err={m.get('err', float('nan')):+.3f}% "
                  f"proj={m.get('err_proj', float('nan')):+.3f}% S2dev={m.get('S2dev', float('nan')):.1e} "
                  f"cnot_chem={m.get('cnot_chem')} D1 cnot={m['cnot_D1']}", flush=True)
        for mu in [0.0, 0.3]:
            ms = []
            for seed in range(NSEED):
                r = penalty.run(sysd, S2, PT, pool, pname, mu, 0.0, noise=NOISE,
                                rng=np.random.default_rng(3000 + seed))
                ms.append(penalty.metrics(r['rows'], sysd['Dex']))
            out['noise'][f'{pname}_mu{mu}'] = ms
            e = [m['err'] for m in ms if 'err' in m]; c = [m['cnot_D1'] for m in ms if m['cnot_D1'] is not None]
            print(f"{name:10s} {pname:4s} mu={mu} noise: err [{min(e):+.2f},{max(e):+.2f}] "
                  f"D1 cnot [{min(c) if c else None},{max(c) if c else None}] ({len(c)}/{len(ms)})", flush=True)
        json.dump(out, open(f'results/penalty_scan_{name}.json', 'w'))
