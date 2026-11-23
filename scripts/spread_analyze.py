"""Summarize the spread of ADAPT results under tie-breaking, orbital gauge and selection noise (spread.py,
naph_spread.py). Writes results/spread_summary.json and prints the numbers quoted in the manuscript."""
import json
import os
import numpy as np

CHEM = 1.6e-3
SYSTEMS = ['CH2', 'NH', 'NH_stretch', 'O2']
POOLS = ['GSD', 'QEB', 'sfGSD']
VARIANTS = ['tie', 'gauge', 'noise0.16', 'noise0.5', 'noise1.6']


def metrics(rows, Dex):
    ic = next((i for i, r in enumerate(rows) if r['dE'] < CHEM), None)
    ok = [abs(r['D'] - Dex) < 0.01 * abs(Dex) for r in rows]
    i1 = next((i for i in range(len(rows)) if all(ok[i:])), None)
    if ic is None:
        return {'n_chem': None}
    r = rows[ic]
    return {'n_chem': ic, 'cnot_chem': r.get('cnot'), 'err': 100 * (r['D'] - Dex) / Dex,
            'err_proj': 100 * (r['D_proj'] - Dex) / Dex, 'n_D1': i1,
            'cnot_D1': rows[i1].get('cnot') if i1 is not None else None}


def seq_key(run, upto):
    return tuple(tuple(o) for o in run['ops'][:upto])


def stats(vals):
    v = np.array([x for x in vals if x is not None], float)
    if len(v) == 0:
        return None
    return {'n': len(v), 'mean': float(v.mean()), 'sd': float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            'min': float(v.min()), 'max': float(v.max())}


out = {}
for s in SYSTEMS:
    for p in POOLS:
        f = f'results/spread_{s}_{p}.json'
        if not os.path.exists(f):
            continue
        d = json.load(open(f)); Dex = d['D_exact']
        res = {'standard': metrics(d['standard']['rows'], Dex)}
        allruns = []
        for v in VARIANTS:
            if v not in d:
                continue
            ms = [metrics(r['rows'], r.get('D_exact', Dex)) for r in d[v]]
            nch = [m['n_chem'] for m in ms]
            distinct = len({seq_key(r, m['n_chem'] if m['n_chem'] is not None else len(r['ops']))
                            for r, m in zip(d[v], ms)})
            res[v] = {'err': stats([m.get('err') for m in ms]), 'err_proj': stats([m.get('err_proj') for m in ms]),
                      'n_chem': stats(nch), 'cnot_D1': stats([m.get('cnot_D1') for m in ms]),
                      'n_reach_chem': sum(x is not None for x in nch), 'n_runs': len(ms),
                      'distinct_paths_to_chem': distinct,
                      'abs_err_values': [abs(m['err']) for m in ms if m.get('err') is not None]}
            if v in ('tie', 'gauge', 'noise0.16'):
                allruns += [m.get('err') for m in ms]
        res['pooled_low_noise'] = stats(allruns)
        out.setdefault(s, {})[p] = res

# naphthalene: original run + selection-noise repeats, natural orbitals (naph_no) and ROHF orbitals (naph_rohf)
NAPH_SETS = {'naph_no': ('naph_no_{p}', 'naph_spread_{p}_s{k}', json.load(open('results/naph_reference_no.json'))['D_exact_NO']),
             'naph_rohf': ('naph_{p}', 'naph_spread_rohf_{p}_s{k}', json.load(open('results/naph_reference.json'))['D_CASCI'])}
for key, (std, rep, ref) in NAPH_SETS.items():
    for p in ['GSD', 'sfGSD']:
        runs = [json.load(open(f'results/{std.format(p=p)}.json'))['history']]
        k = 1
        while os.path.exists(f'results/{rep.format(p=p, k=k)}.json'):
            o = json.load(open(f'results/{rep.format(p=p, k=k)}.json'))
            if o.get('finished'):
                runs.append(o['history'])
            k += 1
        e0 = json.load(open(f'results/{std.format(p=p)}.json'))['E_exact']
        ms = []
        for h in runs:
            rows = [{'dE': r['E'] - e0 if 'dE' not in r else r['dE'], 'D': r['D'], 'D_proj': r['D_proj']} for r in h]
            ms.append(metrics(rows, ref))
        out.setdefault(key, {})[p] = {'runs': ms, 'err': stats([m.get('err') for m in ms]),
                                      'err_proj': stats([m.get('err_proj') for m in ms]),
                                      'n_chem': stats([m.get('n_chem') for m in ms])}
json.dump(out, open('results/spread_summary.json', 'w'), indent=1)

fmt = lambda st: '--' if st is None else f"{st['mean']:+.2f}±{st['sd']:.2f} [{st['min']:+.2f},{st['max']:+.2f}]"
for s in out:
    for p in out[s]:
        r = out[s][p]
        if s in ('naph_no', 'naph_rohf'):
            print(s, p, 'err', fmt(r['err']), 'proj', fmt(r['err_proj']), 'n_chem', fmt(r['n_chem']),
                  [round(m['err'], 2) for m in r['runs'] if m.get('err') is not None])
            continue
        print(f"{s:10s} {p:5s} std {r['standard'].get('err', float('nan')):+.2f} (n={r['standard']['n_chem']})")
        for v in VARIANTS:
            if v in r:
                q = r[v]
                print(f"    {v:9s} err {fmt(q['err'])}  proj {fmt(q['err_proj'])}  n_chem {fmt(q['n_chem'])} "
                      f"reach {q['n_reach_chem']}/{q['n_runs']} paths {q['distinct_paths_to_chem']}")
