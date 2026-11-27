"""Write the LaTeX bodies of the cost, shot and naphthalene tables from results/*.json, so that the manuscript
tables are generated rather than transcribed. Usage (from the repository root): python scripts/make_tables.py [outdir]"""
import json
import numpy as np
import os
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else 'tables'
os.makedirs(OUT, exist_ok=True)
R = lambda f: json.load(open(os.path.join('results', f)))
POOLS = ['GSD', 'QEB', 'sfGSD']
LABEL = {'CH2': r'\ce{CH2}', 'NH': 'NH', 'NH_stretch': r'NH ($2r_\mathrm{e}$)', 'O2': r'\ce{O2}', 'TMM': 'TMM',
         'naph_rohf': 'Naphthalene (ROHF)', 'naph_no': 'Naphthalene (NO)'}


def sci(x):
    m, e = f'{x:.1e}'.split('e')
    return rf'${m}\times10^{{{int(e)}}}$'


def triple(vals):
    return ' / '.join('-' if v is None else str(v) for v in vals)


# ---------------------------------------------------------------- Table (repository only): ansatz cost (parameters, CNOTs)
cost, cn = R('cost.json'), R('cnot_counts.json')
rows = []
for s in ['CH2', 'NH', 'NH_stretch', 'O2', 'TMM', 'naph_rohf', 'naph_no']:
    c = cn.get(s, {})
    g = lambda p, k: c.get(p, {}).get(k)
    rows.append(f"{LABEL[s]} & {triple([g(p, 'n_chem') for p in POOLS])} & {triple([g(p, 'cnot_chem') for p in POOLS])}"
                f" & {triple([g(p, 'n_D1') for p in POOLS])} & {triple([g(p, 'cnot_D1') for p in POOLS])}\\\\")
    if s in cost:   # consistency: parameters from the CNOT script equal those of cost.py
        for p in POOLS:
            assert cost[s][p]['n_chem'] == g(p, 'n_chem') and cost[s][p]['n_D1'] == g(p, 'n_D1'), (s, p)
open(os.path.join(OUT, 'tab_cost.tex'), 'w').write('\n'.join(rows) + '\n')

# ---------------------------------------------------------------- Table (repository only): shots
M, SS = R('measure.json'), R('shots_seeds.json')
rows = []
for s in ['CH2', 'NH', 'NH_stretch', 'O2', 'TMM']:
    m, q = M[s], SS[s]
    NH_, ND_, NF = q['H']['median'], q['Dz']['median'], q['full']['median']
    Dcm = abs(q['D_cm'])
    Nabs = ND_ * (0.01 * Dcm / 0.001) ** 2
    rows.append(f"{LABEL[s]} & {q['nterms_H']} ({q['gH']['median']:.0f}) & {sci(NH_)} & {q['nterms_D']} ({q['gD']['median']:.0f})"
                f" & {sci(ND_)} & {ND_ / NH_:.2g} & {sci(Nabs)} & {sci(NF)} ({NF / ND_:.1f})\\\\")
open(os.path.join(OUT, 'tab_shots.tex'), 'w').write('\n'.join(rows) + '\n')

# ---------------------------------------------------------------- Table (repository only): naphthalene runs
N = R('naph_summary.json')
rows = []
for key, lab in [('rohf', 'ROHF'), ('no', 'NO')]:
    for p in POOLS:
        s = N.get(key, {}).get(p)
        if s is None:
            continue
        f = lambda k, fmt: '-' if s.get(k) is None else fmt.format(s[k])
        rows.append(f"{lab} & {p} & {s['n_final']} & {s['dE_final_mEh']:.2f} & {f('n_chem', '{}')}"
                    f" & {f('D_err_chem_pct', '${:+.2f}$')} & {f('Dproj_err_chem_pct', '${:+.2f}$')}"
                    f" & ${s['D_err_final_pct']:+.2f}$ & {f('n_D_within_1pct_for_good', '{}')} & {s['time_h']:.1f}\\\\")
open(os.path.join(OUT, 'tab_naph.tex'), 'w').write('\n'.join(rows) + '\n')
for f in ['tab_cost.tex', 'tab_shots.tex', 'tab_naph.tex']:
    print(f'--- {f}\n' + open(os.path.join(OUT, f)).read())
# ratios quoted in the text
print('term-by-term / QWC, H:', ', '.join(f"{s} {M[s]['shots_H_chem'][0] / SS[s]['H']['median']:.1f}" for s in SS))
print('term-by-term / QWC, D:', ', '.join(f"{s} {M[s]['shots_D_1pct'][0] / SS[s]['Dz']['median']:.1f}" for s in SS))
print('full separate / axis (measure_full):', {s: round(v['ratio_separate_to_axis'], 2) for s, v in R('measure_full.json').items()})

# ---------------------------------------------------------------- Table (repository only): remedies for the target-spin error
REM = R('remedies.json')
NR = R('naph_remedies.json') if os.path.exists('results/naph_remedies.json') else {}
METH = ['A', 'B2', 'C1', 'C2', 'D']


def cell(m):
    if m is None or m.get('err_chem_pct') is None:
        return '-'
    c = '-' if m.get('cnot_D1') is None else str(m['cnot_D1'])
    return f"${m['err_chem_pct']:+.2f}$ ({c})"


rows, verdict = [], {}
for p in ['GSD', 'sfGSD']:
    for s in ['CH2', 'NH', 'NH_stretch', 'O2']:
        r = REM[s][p]
        rows.append(f"{LABEL[s]} & {p} & " + ' & '.join(cell(r.get(k)) for k in METH) + '\\\\')
    for k in METH:        # pre-registered criterion, evaluated per pool
        acc = sum(abs(REM[s][p][k]['err_chem_pct'] or 1e9) < 1 for s in REM)
        cheap = all(REM[s][p][k]['cnot_D1'] is not None and REM[s][p][k]['cnot_D1'] <= REM[s][p]['A']['cnot_D1']
                    for s in REM)
        verdict[p, k] = (acc, cheap)
    rows.append('\\midrule')
for p in ['GSD', 'sfGSD']:          # naphthalene QEB never reaches chemical accuracy: no entries to report
    if p in NR:
        rows.append(f"{LABEL['naph_no']} & {p} & {cell(NR[p]['A'])} & n.c. & " +
                    ' & '.join(cell(NR[p][k]) for k in ['C1', 'C2', 'D']) + '\\\\')
if rows[-1] == '\\midrule':
    rows.pop()
open(os.path.join(OUT, 'tab_remedies.tex'), 'w').write('\n'.join(rows) + '\n')
print(open(os.path.join(OUT, 'tab_remedies.tex')).read())
print('criterion (systems within 1% at chem, CNOTs <= A everywhere):',
      {f'{p}/{k}': v for (p, k), v in verdict.items()})

# ---------------------------------------------------------------- Table 1: spin penalty and Gaussian filter (shift-invert: SI Table S-refine)
def fix_cell(err, cnot=None):
    if err is None:
        return '-'
    txt = '$0.00$' if abs(err) < 0.005 else f"${err:+.2f}$"
    return txt + ('' if cnot is False else f" ({'-' if cnot is None else cnot})")


def s2_cell(x):
    """<S^2> - S(S+1) at chemical accuracy; spin-pure states (|x| < 1e-12) are shown as 0"""
    if x is None:
        return '-'
    if abs(x) < 1e-12:
        return '$0$'
    m_, e_ = f"{x:.2e}".split('e'); return rf"${m_}\times10^{{{int(e_)}}}$"


# every system and pool: ADAPT and spin-penalized ADAPT (mu = 3 Eh) at chemical accuracy, and the
# spin-shifted Gaussian filter (T = 3/(E1 - E0), mu_f = 0.05 Eh) applied to the unpenalized state there
from cnot_counts import cnots
CNC = R('cnot_counts.json')
cnc = lambda s_, p_: '-' if CNC.get(s_, {}).get(p_, {}).get('cnot_chem') is None else str(CNC[s_][p_]['cnot_chem'])
rows = []
for s in ['CH2', 'NH', 'NH_stretch', 'O2']:
    F = R(f'fix_{s}.json')['runs']; PF = R(f'propfilter_{s}.json')['runs']
    for k, p in enumerate(POOLS):
        m = F[p]['metrics']
        pen = fix_cell(F[p + '_mu3']['metrics']['err'], F[p + '_mu3']['metrics']['cnot_D1']) if p != 'sfGSD' else 'n.a.'
        pen_s2 = s2_cell(F[p + '_mu3']['metrics']['S2dev']) if p != 'sfGSD' else 'n.a.'
        rows.append(f"{LABEL[s] if k == 0 else ''} & {p} & {cnc(s, p)} & {fix_cell(m['err'], m['cnot_D1'])} & {fix_cell(m['err_proj'], False)} & {s2_cell(m['S2dev'])} & {pen} & {pen_s2} & "
                    f"{fix_cell(PF[p]['gauss']['rule_spin']['D'], False)}\\\\")
    rows.append(r'\midrule')
SI, NF = R('shiftinvert_naph.json'), R('naph_filter.json')
cn_naph = R('cnot_counts.json')['naph_no']
Dex = R('naph_reference_no.json')['D_exact_NO']
P = R('naph_penalty_GSD_mu3_w0.json'); h = P['history']
ic = next(i for i, r in enumerate(h) if r['dE'] < 1.6e-3)
ok = [abs(r['D'] - Dex) < 0.01 * abs(Dex) for r in h]
i1 = next((i for i in range(len(h)) if all(ok[i:])), None)
c1 = sum(cnots(tuple(l), False)[0] for l in P['ops'][:h[i1]['n']]) if i1 is not None else None
for k, p in enumerate(POOLS):
    lab = LABEL['naph_no'] if k == 0 else ''
    if p == 'QEB':
        rows.append(f"{lab} & QEB & - & - & - & - & n.c. & n.c. & -\\\\")
        continue
    pen = fix_cell(100 * (h[ic]['D'] - Dex) / Dex, c1) if p == 'GSD' else 'n.a.'
    pen_s2 = s2_cell(h[ic]['S2'] - 2) if p == 'GSD' else 'n.a.'
    std_s2 = s2_cell(R('naph_summary.json')['no'][p]['S2dev_chem'])
    std_proj = fix_cell(R('naph_summary.json')['no'][p]['Dproj_err_chem_pct'], False)
    rows.append(f"{lab} & {p} & {cnc('naph_no', p)} & {fix_cell(SI['naph_' + p]['err'][0], cn_naph[p]['cnot_D1'])} & {std_proj} & {std_s2} & {pen} & {pen_s2} & "
                f"{fix_cell(NF['runs'][p]['gauss']['rule_spin']['D'], False)}\\\\")
open(os.path.join(OUT, 'tab_fix.tex'), 'w').write('\n'.join(rows) + '\n')
print(open(os.path.join(OUT, 'tab_fix.tex')).read())

# ---------------------------------------------------------------- Table 3: compiled cost of the Gaussian filter (naphthalene)
C = R('filter_compile.json'); ch = C['chosen']
def sci1(x):
    m_, e_ = f"{x:.2e}".split('e'); return rf"${m_}\times10^{{{int(e_)}}}$"
CC = R('certified_cost.json')
PS = R('precision_scaling.json')                  # 1% and 5% targets, scripts/precision_scaling.py
def grp(title):                                      # group heading spanning all three columns
    return (rf"\multicolumn{{3}}{{l}}{{\textit{{{title}}}}}",)
both = lambda x: rf"\multicolumn{{2}}{{c}}{{{x}}}"
crow = [
    grp('Block encoding (both protocols)'),
    (r'\quad DF threshold; rank; eigenvectors', both(rf"$10^{{{int(np.log10(ch['thresh']))}}}$; {ch['L']}; {ch['Lxi']}")),
    (r'\quad Change of $E_0$ (m$E_\mathrm{h}$); of $D$', both(rf"${ch['dE0_mEh']:.2f}$; ${ch['dD_pct']:+.2f}\%$")),
    (r'\quad $\lambda_\mathrm{DF}$ ($E_\mathrm{h}$)', both(f"{ch['lambda_DF']:.2f}")),
    (r'\quad Logical qubits', both(f"{C['logical_qubits']}")),
    (r'\quad Toffolis per controlled walk step', both(f"{C['toffoli_per_controlled_step']}")),
    grp('One circuit (Hadamard test, two controlled evolutions)'),
    (r'\quad Filter length $\tau$ ($E_\mathrm{h}^{-1}$)', f"{C['T']:.2f}", f"{CC['T_Eh']:.2f}"),
    (r'\quad Filter length $\tau(E_1-E_0)$', '3', f"{CC['x_cert']:.2f}"),
    (r'\quad Error per evolution, $\epsilon_\mathrm{sim}$', sci1(C['eps']).replace('$1.00\\times', '$'), sci1(CC['eps_sim'])),
    (r'\quad Walk steps, mean; $4\sigma$', f"{C['walk_steps_mean_numerator']:.2f}; {C['walk_steps_max_4sigma']}", f"{CC['walk_steps_mean']:.2f}; {CC['walk_steps_max_4sigma']}"),
    (r'\quad Toffolis, mean', sci1(C['toffoli_per_circuit_mean']), sci1(CC['toffoli_per_circuit'])),
    (r'\quad Physical qubits (code distance)', f"{sci1(C['physical_qubits'])} ({C['code_distance']})", f"{sci1(CC['physical_qubits'])} ({CC['code_distance']})"),
    (r'\quad Time, mean (min)', f"{C['circuit_time_s']/60:.2f}", f"{CC['circuit_time_s']/60:.2f}"),
    grp(r'Standard error of 1\% in $D$, one device'),
    (r'\quad Sampling: circuits', sci1(C['shots']), sci1(CC['shots'])),
    (r'\quad Sampling: time (yr)', sci1(C['total_time_days_sequential']/365.25), sci1(CC['total_time_years_sequential'])),
    (r'\quad Amplitude estimation: degree; repetitions', f"{C['qsvt_degree']}; {sci1(C['ae_repetitions'])}", f"{CC['qsvt_degree']}; {sci1(CC['ae_repetitions'])}"),
    (r'\quad Amplitude estimation: Toffolis', sci1(C['ae_toffoli_total']), sci1(CC['ae_toffoli_total'])),
    (r'\quad Amplitude estimation: time (d)', f"{C['ae_time_days']:.2f}", f"{CC['ae_time_days']:.2f}"),
    grp(r'Standard error of 5\% in $D$, one device'),
    (r'\quad Sampling: circuits', sci1(PS['rule']['0.05']['shots']), sci1(PS['certified']['0.05']['shots'])),
    (r'\quad Sampling: time (yr)', f"{PS['rule']['0.05']['sampling_years']:.2f}", f"{PS['certified']['0.05']['sampling_years']:.2f}"),
    (r'\quad Amplitude estimation: Toffolis', sci1(PS['rule']['0.05']['ae_toffoli']), sci1(PS['certified']['0.05']['ae_toffoli'])),
    (r'\quad Amplitude estimation: time (d)', f"{PS['rule']['0.05']['ae_days']:.2f}", f"{PS['certified']['0.05']['ae_days']:.2f}"),
    grp('Additional costs and reference'),
    (r'\quad Certificate inputs: $\sigma$ by amplitude estimation (Toffolis)', '--', sci1(CC['sigma_measurement_toffoli'])),
    (r'\quad QPE energy to 1.6~m$E_\mathrm{h}$: Toffolis; logical qubits', both(f"{sci1(C['qpe_toffoli_chem_acc'])}; {C['logical_qubits'] + C['qpe_register_removed'] - 1}")),
]
open(os.path.join(OUT, 'tab_compile.tex'), 'w').write('\n'.join(' & '.join(r_) + '\\\\' for r_ in crow) + '\n')
print(open(os.path.join(OUT, 'tab_compile.tex')).read())
