"""Effect of finite-shot readout of D on the error at chemical accuracy and on the stability-window stopping rule.

Each recorded iterate of a noiseless ADAPT trajectory is read out as D_n + xi_n, xi_n ~ N(0, (r |D_exact|)^2),
independently per iterate, with relative standard error r = 1%, 0.3% and 0.1% (N_D, about 11 N_D and 100 N_D shots per
iterate, where N_D is the shot count for 1% in the shot-cost table). The chemical-accuracy iterate is fixed by the exact energy.
NSAMP noise realisations per trajectory. Writes results/shotnoise.json.
"""
import json
import numpy as np
from stopping import window_stop, trajectories

R_LIST = [0.01, 0.003, 0.001]
M_LIST = [10, 20, 30]
NSAMP = 200
CHEM = 1.6e-3


def dE_list(name, p):
    if name in ('CH2', 'NH', 'NH_stretch', 'O2'):
        return [r['dE'] for r in json.load(open(f'results/errbar_{name}_{p}.json'))]
    tag = '' if name == 'naph_rohf' else '_no'
    return [r['dE'] for r in json.load(open(f'results/naph{tag}_{p}.json'))['history']]


out = {}
rng = np.random.default_rng(7)
for name, p, N, D, Dex in trajectories():
    D = np.asarray(D); N = np.asarray(N); dE = np.asarray(dE_list(name, p))
    ic = next((i for i, x in enumerate(dE) if x < CHEM), None)
    res = {'n_chem': None if ic is None else int(N[ic])}
    for r in R_LIST:
        noise = rng.normal(0.0, r * abs(Dex), (NSAMP, len(D)))
        Dn = D[None, :] + noise
        e = {}
        if ic is not None:
            err = 100 * np.abs(Dn[:, ic] - Dex) / abs(Dex)
            e['err_chem_median'] = float(np.median(err)); e['err_chem_p95'] = float(np.percentile(err, 95))
            e['frac_chem_within_1pct'] = float(np.mean(err < 1))
        for m in M_LIST:
            stops = [window_stop(N, Dn[k], Dex, m) for k in range(NSAMP)]
            n_s = [s[0] for s in stops if s[0] is not None]
            # true error of the noiseless D at the stopping iterate
            tru = [100 * abs(D[list(N).index(s[0])] - Dex) / abs(Dex) for s in stops if s[0] is not None]
            e[f'm{m}'] = {'frac_stop': len(n_s) / NSAMP,
                          'n_stop_median': float(np.median(n_s)) if n_s else None,
                          'true_err_median': float(np.median(tru)) if tru else None,
                          'true_err_p95': float(np.percentile(tru, 95)) if tru else None,
                          'frac_true_err_above_1pct': float(np.mean(np.array(tru) > 1)) if tru else None,
                          'reported_err_p95': float(np.percentile([s[1] for s in stops if s[0] is not None], 95))
                          if n_s else None}
        res[f'r{r}'] = e
    noiseless = {f'm{m}': window_stop(N, D, Dex, m) for m in M_LIST}
    res['noiseless'] = {k: {'n_stop': v[0], 'true_err': v[1]} for k, v in noiseless.items()}
    out.setdefault(name, {})[p] = res
    line = f'{name:10s} {p:6s}'
    for r in R_LIST:
        q = res[f'r{r}']
        line += f" | r={r:.3f}: chem<1% {q.get('frac_chem_within_1pct', float('nan')):.2f}"
        q20 = q['m20']
        line += f" m20 stop {q20['frac_stop']:.2f} err95 {q20['true_err_p95'] if q20['true_err_p95'] is None else round(q20['true_err_p95'], 1)}"
    print(line, flush=True)
json.dump(out, open('results/shotnoise.json', 'w'), indent=1)
