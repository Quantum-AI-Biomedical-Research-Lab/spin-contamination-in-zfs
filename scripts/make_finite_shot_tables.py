"""SI tables for the end-to-end finite-shot simulation (results/finite_shot.json): tab_si_finiteshot.tex and
tab_si_finiteshot_scen.tex. Usage (from the repository root): python scripts/make_finite_shot_tables.py [outdir]."""
import json, os, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else 'tables'
os.makedirs(OUT, exist_ok=True)
d = json.load(open('results/finite_shot.json'))
NAME = {'CH2': r'\ce{CH2}', 'NH': 'NH', 'NH_stretch': r'NH ($2r_\mathrm{e}$)', 'O2': r'\ce{O2}', 'naph_no': 'Naphthalene (NO)'}
sci = lambda x: f"${x/10**len(str(int(x))[1:]):.1f}\\times10^{{{len(str(int(x)))-1}}}$"
rows = []
for s in ['CH2', 'NH', 'NH_stretch', 'O2', 'naph_no']:
    for p in ['GSD', 'QEB', 'sfGSD']:
        k = f'{p}_chem'
        if k not in d[s]:
            continue
        b = d[s][k]['baseline']
        rows.append(f"{NAME[s]} & {p} & {d[s][k]['dE_mEh']:.2f} & {b['tau_med']:.2f} & {sci(b['N_med'])} & {b['err_med']:.2f} & "
                    f"{b['err_95']:.2f} & {b['coverage']:.3f} & {b['sys_max']:.4f}\\\\")
open(f'{OUT}/tab_si_finiteshot.tex', 'w').write('\n'.join(rows) + '\n')
SC = [('baseline', 'Baseline'), ('exact_sigma', r'$\sigma$ known exactly'), ('W1', r'$W_1$ instead of $W$'),
      ('gap90', r'Gap bound $0.9\times$ true gap'), ('eps_sim=1e-06', r'$\epsilon_\mathrm{sim}=10^{-6}$'),
      ('eps_sim=0.001', r'$\epsilon_\mathrm{sim}=10^{-3}$'), ('eps_sim=0.01', r'$\epsilon_\mathrm{sim}=10^{-2}$'),
      ('dc=5mEh', r'Unknown center offset 5~m$E_\mathrm{h}$'), ('dc=10mEh', r'Unknown center offset 10~m$E_\mathrm{h}$'),
      ('dc=20mEh', r'Unknown center offset 20~m$E_\mathrm{h}$')]
CASES = [('CH2', 'GSD_chem'), ('NH_stretch', 'QEB_chem'), ('naph_no', 'GSD_chem')]
rows = []
for key, lab in SC:
    cells = []
    for s, k in CASES:
        b = d[s][k][key]
        cells.append(f"{b['tau_med']:.2f} & {b['coverage']:.2f} & {b['sys_max']:.3f}")
    rows.append(lab + ' & ' + ' & '.join(cells) + '\\\\')
open(f'{OUT}/tab_si_finiteshot_scen.tex', 'w').write('\n'.join(rows) + '\n')
print(open(f'{OUT}/tab_si_finiteshot.tex').read()); print(open(f'{OUT}/tab_si_finiteshot_scen.tex').read())
