import numpy as np

def michaelis_menten(S, Vmax, Km):
    r"""
    V = \frac{V_{\max}\, S}{K_m + S}
    """
    return (Vmax * S) / (Km + S)

def haldane(S, Vmax, Km, Ki):
    r"""
    $V = \frac{V_{\max}\, S}{K_m + S + \frac{S^2}{K_i}}$
    """
    return (Vmax * S) / (Km + S + (S**2 / Ki))

def hill(S, Vmax, K05, n):
    r"""
    $V = \frac{V_{\max}\, S^n}{K_d^n + S^n}$
    """
    return (Vmax * (S**n)) / (K05**n + S**n)
