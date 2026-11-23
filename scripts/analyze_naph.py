"""Convergence analysis of the 20-qubit naphthalene ADAPT-VQE runs in ROHF orbitals and in CASSCF natural orbitals.

Writes results/naph_summary.json (both orbital sets) and the manuscript figure fig3_naphthalene.pdf
(rows: orbital set; columns: energy error and error in D versus the number of ADAPT operators).
Usage (from the repository root): python scripts/analyze_naph.py [figure_outdir]
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else 'results'
CHEM = 1.6e-3
plt.rcParams.update({'font.size': 7, 'axes.labelsize': 7, 'legend.fontsize': 6.5, 'xtick.labelsize': 6.5,
                     'ytick.labelsize': 6.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'font.family': 'sans-serif'})
COLOR = {'GSD': '#6e6e6e', 'QEB': '#b4b4b4', 'sfGSD': '#2a6fdb'}
STYLE = {'GSD': '-', 'QEB': '-', 'sfGSD': '-'}

R_rohf = json.load(open('results/naph_reference.json'))
R_no = json.load(open('results/naph_reference_no.json'))
SETS = [('rohf', '', 'ROHF orbitals', R_rohf['D_CASCI'], R_rohf['E_zfs_CASCI']),
        ('no', '_no', 'CASSCF natural orbitals', R_no['D_exact_NO'], R_no['E_zfs_exact_NO'])]
summary = {'reference': {'D_ROHF': R_rohf['D_ROHF'], 'D_CASCI_ROHF_orbitals': R_rohf['D_CASCI'],
                         'D_CASSCF': R_rohf['D_CASSCF'], 'D_exact_NO': R_no['D_exact_NO'],
                         'E_zfs_ROHF': R_rohf['E_zfs_ROHF'], 'E_zfs_CASCI': R_rohf['E_zfs_CASCI'],
                         'E_zfs_CASSCF': R_rohf['E_zfs_CASSCF'], 'E_zfs_exact_NO': R_no['E_zfs_exact_NO'],
                         'D_refdet_NO': R_no['D_refdet_NO'], 'gap_NO_mEh': R_no['gap_mEh'],
                         'E_CASCI': R_rohf['E_CASCI'], 'E_CASSCF': R_no['E_CASSCF'], 'n_det': R_rohf['n_det']},
           'experiment': {'D': 0.1006, 'E': -0.0138, 'source': 'Hutchison and Mangum, durene host'}}

fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.0), sharex='col')
for row, (key, tag, title, Dr, Er) in enumerate(SETS):
    summary[key] = {}
    for p in ['GSD', 'QEB', 'sfGSD']:
        f = f'results/naph{tag}_{p}.json'
        if not os.path.exists(f):
            continue
        o = json.load(open(f))
        H = o['history']
        n = np.array([h['n'] for h in H]); dE = np.array([h['E'] - o['E_exact'] for h in H])
        D = np.array([h['D'] for h in H])
        ca = next((i for i, d in enumerate(dE) if d < CHEM), None)
        stay = next((int(n[i]) for i in range(len(H)) if np.all(np.abs(D[i:] - Dr) < 0.01 * abs(Dr))), None)
        last = H[-1]
        s = {'n_final': int(last['n']), 'dE_final_mEh': 1e3 * dE[-1], 'fid_final': last['fid'],
             'S2dev_final': last['S2'] - 2, 'D_err_final_pct': 100 * (last['D'] - Dr) / Dr,
             'Dproj_err_final_pct': 100 * (last['D_proj'] - Dr) / Dr,
             'Ezfs_err_final': last['Ezfs'] - Er, 'time_h': (o.get('time_s') or last['t']) / 3600,
             'finished': 'time_s' in o, 'n_D_within_1pct_for_good': stay,
             'max_abs_D_err_after_chem_pct': (float(100 * np.abs(D[ca:] - Dr).max() / abs(Dr))
                                             if ca is not None else None)}
        if ca is not None:
            h = H[ca]
            s.update({'n_chem': int(h['n']), 'D_err_chem_pct': 100 * (h['D'] - Dr) / Dr,
                      'Dproj_err_chem_pct': 100 * (h['D_proj'] - Dr) / Dr, 'S2dev_chem': h['S2'] - 2,
                      'w_contam_chem': 1 - h['w_target'], 'Ezfs_err_chem': h['Ezfs'] - Er, 'fid_chem': h['fid']})
        m = n >= max(1, n[-1] - 60)
        s['dE_decay_per_100_ops'] = float(10 ** (100 * np.polyfit(n[m], np.log10(dE[m]), 1)[0]))
        summary[key][p] = s
        axes[row, 0].semilogy(n, 1e3 * dE, color=COLOR[p], ls=STYLE[p], lw=1.1, label=p)
        axes[row, 1].plot(n, 100 * (D - Dr) / Dr, color=COLOR[p], ls=STYLE[p], lw=1.1, label=p)
        if ca is not None:
            axes[row, 1].plot(n[ca], 100 * (D[ca] - Dr) / Dr, 'o', ms=3.2, color=COLOR[p], mec='k', mew=0.4)
    a0, a1 = axes[row]
    a0.axhline(1.6, color='0.35', ls='--', lw=0.7)
    a0.set_ylabel('Energy error (m$E_\\mathrm{h}$)')
    a0.set_title(title, fontsize=7, loc='left')
    a1.axhspan(-1, 1, color='0.9', lw=0); a1.axhline(0, color='0.3', lw=0.6)
    a1.set_ylim(-25, 25); a1.set_ylabel('Error in $D$ (%)')
    a1.set_title(f'{title}: exact $D$ = {Dr:.4f} cm$^{{-1}}$', fontsize=7, loc='left')
axes[0, 0].legend(frameon=False)
for a in axes[1]:
    a.set_xlabel('ADAPT operators')
fig.tight_layout(pad=0.4)
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, 'fig3_naphthalene.pdf')); fig.savefig('results/naph_adapt_convergence.png', dpi=200)
json.dump(summary, open('results/naph_summary.json', 'w'), indent=1, default=float)
print(json.dumps(summary, indent=1, default=lambda z: round(float(z), 6)))
