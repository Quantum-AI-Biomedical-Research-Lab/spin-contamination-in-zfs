"""Post hoc checks on the remedies study (12 qubits), from the saved per-iterate rows:
  * D-aware selection B: how often the same operator is re-selected (stalling mechanism);
  * variance extrapolation fitted against sigma_H instead of sigma_H^2 (D-sigma), as eq s would suggest.
Writes results/remedies_post.json."""
import json
from collections import Counter
import numpy as np
from remedies import metrics, M_EXTRAP

R = json.load(open('results/remedies.json'))
out = {}
for s in ['CH2', 'NH', 'NH_stretch', 'O2']:
    for p in ['GSD', 'sfGSD']:
        o = json.load(open(f'results/remedies_{s}_{p}_B.json'))
        ops = [tuple(x) for x in o['ops']]
        op, cnt = Counter(ops).most_common(1)[0]
        rows = json.load(open(f'results/remedies_{s}_{p}_A.json'))['rows']
        for i, r in enumerate(rows):
            r['Dsig'] = None
            if i >= M_EXTRAP - 1:
                seg = rows[i - M_EXTRAP + 1:i + 1]
                x = np.sqrt([q['sigma_H2'] for q in seg]); y = [q['D'] for q in seg]
                r['Dsig'] = float(np.polyfit(x, y, 1)[1]) if np.ptp(x) > 0 else r['D']
        out[f'{s}/{p}'] = {'B_n_ops': len(ops), 'B_distinct': len(set(ops)), 'B_most_common': list(op),
                           'B_most_common_count': cnt, 'B_first_repeat_at': ops.index(op),
                           'D_sigma': metrics(rows, 'Dsig', R[s]['D_exact'])}
        print(s, p, out[f'{s}/{p}'])
json.dump(out, open('results/remedies_post.json', 'w'), indent=1)
