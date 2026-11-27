"""Compiled cost of the certified protocol for naphthalene (Table 3, second column).

Certified protocol: total filter length T = x_cert/(E1 - E0), x_cert from naph_certificate.json (0.1%, largest over
GSD and sfGSD), evolutions to eps_sim from filter_guarantee.json (implementation error below 0.01% of |D|). Same
double-factorized block encoding, Toffoli count per controlled walk step and surface-code costing as
filter_compile.py; walk steps from qpe_compare.json. The retained weight Ng at x_cert is computed on the rebuilt
ADAPT states (results/naph_no_<pool>_chem_state.npy) and the smaller value enters the shot overhead.
Writes results/certified_cost.json.
"""
import json, warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.special import jv
from openfermion.resource_estimates.surface_code_compilation.physical_costing import cost_estimator
import vqe_sparse as sps
from naph_filter_lib import gauss_filter

fc = json.load(open('results/filter_compile.json')); q = json.load(open('results/qpe_projector_compare.json'))
nc = json.load(open('results/naph_certificate.json')); NF = json.load(open('results/naph_filter.json'))
lam = fc['chosen']['lambda_DF']; G = fc['E1'] - fc['E0']; cstep = fc['toffoli_per_controlled_step']
fq = q['filter']['cert_T01']; x = fq['x']; eps = fq['eps_sim']; T = x / G

def walk_steps(v, e):
    v = abs(v); kmax = int(v + 60 + 10 * np.sqrt(v + 1))
    tail = 2 * np.cumsum(np.abs(jv(np.arange(kmax, -1, -1), v)))[::-1]
    ok = np.nonzero(tail[1:] <= e)[0]
    return int(ok[0]) if len(ok) else kmax

ws_max = 2 * walk_steps(lam * 4 * T, eps) * 2                    # both times at 4 standard deviations
dat = np.load('results/naph_setup_no.npz'); spc = sps.SparseSpace(10, (6, 4)); shape = (spc.na, spc.nb)
Hs = sps.Hamiltonian(dat['h1'], dat['eri'], float(dat['ecore']), 10, (6, 4)); Hm = lambda v: Hs(v, shape)
Ng = {p: gauss_filter(np.load(f'results/naph_no_{p}_chem_state.npy'), Hm, T, m=300)[1] for p in ['GSD', 'sfGSD']}
c = NF['cost']; ND = (c['sum_sigma_g'] / (0.01 * abs(c['X_mean']))) ** 2
Fshot = (c['sum_sqrt_m2_g'] / c['sum_sigma_g']) ** 2 / min(Ng.values()) ** 2; shots = Fshot * ND
tof = fq['walk_steps'] * cstep
best, params = cost_estimator(fc['logical_qubits'], int(np.ceil(tof)), physical_error_rate=1e-3)
rate = tof / best.duration.total_seconds()
tof_ae = fc['ae_repetitions'] * 2 * fq['qsvt_degree'] * cstep
sig_meas = max(v['toffoli'] for k, v in q['sigma_measurement'].items() if v['x1'] == nc['GSD']['chosen']['x1'])
out = {'x_cert': x, 'T_Eh': T, 'eps_sim': eps, 'walk_steps_mean': fq['walk_steps'], 'walk_steps_max_4sigma': ws_max,
       'toffoli_per_circuit': tof, 'physical_qubits': best.physical_qubit_count, 'code_distance': params.logical_data_qubit_distance,
       'circuit_time_s': best.duration.total_seconds(), 'Ng': Ng, 'F': Fshot, 'shots': shots,
       'total_time_years_sequential': shots * best.duration.total_seconds() / 86400 / 365.25,
       'qsvt_degree': fq['qsvt_degree'], 'ae_repetitions': fc['ae_repetitions'], 'ae_toffoli_total': tof_ae,
       'ae_time_days': tof_ae / rate / 86400, 'sigma_measurement_toffoli': sig_meas, 'prefilter_x1': nc['GSD']['chosen']['x1']}
json.dump(out, open('results/certified_cost.json', 'w'), indent=1, default=float)
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()})
