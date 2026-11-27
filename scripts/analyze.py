import json, numpy as np
CHEM = 1.6e-3
rows = []
for name in ['CH2', 'NH', 'NH_stretch', 'O2', 'TMM']:
    o = json.load(open(f'results/{name}.json'))
    Dr = o['D_CASCI']
    print(f"\n{name}: D_ROHF={o['D_ROHF']:.4f} D_CASCI={Dr:.4f} D_CASSCF={o['D_CASSCF']:.4f} "
          f"E_CASCI={o['E_zfs_CASCI']:.4f} E_CASSCF={o['E_zfs_CASSCF']:.4f} pools={o['pool_sizes']}")
    for p, r in o['runs'].items():
        H = r['history']
        ca = next((h for h in H if h['dE'] < CHEM), None)
        maxs2 = max(abs(h['S2'] - 2) for h in H)
        maxcont = max(h['dD_contam_frob_cm'] for h in H)
        maxdD_after = max(abs(h['D'] - Dr) for h in H if h['dE'] < CHEM)
        # first n with |D - Dref| < 1% and staying there
        n1 = next((h['n'] for i, h in enumerate(H)
                   if all(abs(g['D'] - Dr) / abs(Dr) < 0.01 for g in H[i:])), None)
        print(f"  {p:6s} n_chem={ca['n']:3d} S2dev={ca['S2']-2:8.1e} D_err={ca['D']-Dr:+.4f} "
              f"({100*(ca['D']-Dr)/Dr:+.1f}%) contam={ca['dD_contam_frob_cm']:.1e}  "
              f"max|S2-2|={maxs2:.1e} max_contam={maxcont:.1e} max|dD| after chem={maxdD_after:.4f} "
              f"n(D within 1% for good)={n1} n_final={H[-1]['n']}")
