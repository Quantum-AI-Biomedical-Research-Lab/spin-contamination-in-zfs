"""Measured-input accuracy certificate (manuscript eqs 12 and 13): Temple bound on eps, fidelity bound, filter bound."""
import numpy as np


def cert_inputs(sig, Delta):
    """Measured-input certificate (Section 2.3): with a gap lower bound Delta and the hypothesis eps < Delta/2,
    Temple's inequality eps (Delta - eps) <= sigma^2 gives eps <= eb; Eckart and the variance bound give 1 - F <= u."""
    if not (sig < Delta / 2):
        return None
    eb = Delta / 2 - np.sqrt(Delta ** 2 / 4 - sig ** 2)
    u = min(eb / Delta, sig ** 2 / (Delta - eb) ** 2)
    return (eb, u) if u < 1 else None

def cert_bound_pct(Tg, eb, u, Df, W):
    """proven |D_g - D| / |D| (%) with sigma_X(T) <= W/2 and ||X - <X>_T|| <= W (W = spectral width of X / |D|)"""
    q = u / (1 - u) * np.exp(-Tg ** 2 * Df * (Df - 2 * eb)); f = q / (1 + q)
    return 100 * W * (np.sqrt(f * (1 - f)) + f)

