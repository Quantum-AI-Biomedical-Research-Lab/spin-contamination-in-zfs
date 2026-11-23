"""Practical (exact-answer-free) error estimates for D along ADAPT trajectories, tested against the true error.
  window(m):  e = max_{k in [n-m, n]} |D_k - D_n|
  extrap(m):  fit D_k = a + b sigma_H,k over the last m iterates; D* = a; e = |D_n - a|
A stop rule "e < 1% of |D_n|" is scored by the true error at the stopping iterate."""
import json, sys
import numpy as np

def estimators(rows, m):
    D = np.array([r['D'] for r in rows]); s = np.array([r['sigma_H'] for r in rows])
    win, ext, Dx = [np.nan] * len(rows), [np.nan] * len(rows), [np.nan] * len(rows)
    for n in range(m, len(rows)):
        win[n] = np.abs(D[n - m:n + 1] - D[n]).max()
        a, b = np.polynomial.polynomial.polyfit(s[n - m + 1:n + 1], D[n - m + 1:n + 1], 1)
        Dx[n] = a; ext[n] = abs(D[n] - a)
    return np.array(win), np.array(ext), np.array(Dx)

def score(rows, Dex, m, rel=0.01):
    err = np.array([r['err'] for r in rows]); D = np.array([r['D'] for r in rows])
    win, ext, Dx = estimators(rows, m)
    out = {}
    for lab, e in (('window', win), ('extrap', ext)):
        ok = ~np.isnan(e)
        cover = float(np.mean(err[ok] <= e[ok])) if ok.any() else np.nan
        stop = next((n for n in range(len(rows)) if ok[n] and e[n] < rel * abs(D[n])), None)
        out[lab] = {'coverage': cover, 'n_stop': stop,
                    'true_err_at_stop_pct': (100 * err[stop] / abs(Dex)) if stop is not None else None}
    good = ~np.isnan(Dx)
    out['extrap_D_err_pct_last'] = float(100 * abs(Dx[good][-1] - Dex) / abs(Dex)) if good.any() else None
    out['naive_D_err_pct_last'] = float(100 * err[-1] / abs(Dex))
    return out

if __name__ == '__main__':
    files = sys.argv[1:]
    for f in files:
        rows = json.load(open(f))
        Dex = rows[-1]['D'] - np.sign(rows[-1]['D'] - rows[-1]['D_T']) * 0  # placeholder, replaced below
