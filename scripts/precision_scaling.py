"""Run time of the naphthalene protocols versus the target standard error of D (Table 3, 1% and 5% rows).

Uses the formulas of filter_compile.py and certified_cost.py with the target precision eps as a parameter:
sampling shots F * N_D with N_D = (sum_sigma_g / (eps |X|))^2, so shots scale as 1/eps^2; amplitude estimation
with n_ae = ceil(pi alpha_X / (2 eps |X|)) coherent repetitions of the filter and its inverse (QSVT degree as stored).
The filter length, eps_sim and the Toffoli rate are those of the stored runs; only the statistical target changes.
Checks that eps = 1% reproduces filter_compile.json and certified_cost.json. Writes results/precision_scaling.json.
"""
import json, math

fc = json.load(open('results/filter_compile.json')); cc = json.load(open('results/certified_cost.json'))
c = json.load(open('results/naph_filter.json'))['cost']
alphaX = c['X_l1'] + abs(c.get('X_identity', 0.0)); X = abs(c['X_mean']); cstep = fc['toffoli_per_controlled_step']
cols = {'rule': dict(shots1=fc['shots'], t_circ=fc['circuit_time_s'], deg=fc['qsvt_degree'],
                     rate=fc['toffoli_rate_per_s'], ae1=fc['ae_time_days']),
        'certified': dict(shots1=cc['shots'], t_circ=cc['circuit_time_s'], deg=cc['qsvt_degree'],
                          rate=cc['toffoli_per_circuit'] / cc['circuit_time_s'], ae1=cc['ae_time_days'])}
out = {}
for name, v in cols.items():
    out[name] = {}
    for eps in (0.01, 0.05):
        shots = v['shots1'] * (0.01 / eps) ** 2
        n_ae = int(math.ceil(math.pi * alphaX / (2 * eps * X)))
        tof_ae = n_ae * 2 * v['deg'] * cstep
        out[name][f'{eps:g}'] = {'shots': shots, 'sampling_years': shots * v['t_circ'] / 86400 / 365.25,
                                 'ae_repetitions': n_ae, 'ae_toffoli': tof_ae, 'ae_days': tof_ae / v['rate'] / 86400}
    r1 = out[name]['0.01']
    assert abs(r1['ae_days'] / v['ae1'] - 1) < 1e-9, (name, r1['ae_days'], v['ae1'])
    assert r1['ae_repetitions'] == fc['ae_repetitions']
json.dump(out, open('results/precision_scaling.json', 'w'), indent=1)
for n, d in out.items():
    for e, r in d.items():
        print(n, e, f"shots {r['shots']:.3e}  sampling {r['sampling_years']:.2f} yr  AE reps {r['ae_repetitions']}  AE {r['ae_days']:.2f} d")
