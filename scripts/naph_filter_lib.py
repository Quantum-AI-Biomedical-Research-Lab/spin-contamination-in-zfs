"""Gaussian filter g(H) psi = exp[-(H - E(psi))^2 T^2 / 2] psi in a Lanczos basis (full reorthogonalization)."""
import numpy as np


def gauss_filter(psi, Hm, T, m=300):
    n = psi.size; Q = np.zeros((m, n)); a = np.zeros(m); b = np.zeros(m)
    q = psi / np.linalg.norm(psi); E = float(q @ Hm(q))
    for j in range(m):
        Q[j] = q; w = Hm(q); a[j] = q @ w
        w -= Q[:j + 1].T @ (Q[:j + 1] @ w); w -= Q[:j + 1].T @ (Q[:j + 1] @ w)
        if j + 1 < m:
            b[j] = np.linalg.norm(w); q = w / b[j]
    th, U = np.linalg.eigh(np.diag(a) + np.diag(b[:m - 1], 1) + np.diag(b[:m - 1], -1))
    y = Q.T @ (U @ (np.exp(-0.5 * ((th - E) * T) ** 2) * U[0]))
    Ng = float(y @ y)
    return y / np.sqrt(Ng), Ng
