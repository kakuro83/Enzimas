import numpy as np

def michaelis_menten(S, Vmax, Km):
    r"""
    V = \frac{V_{\max}\, S}{K_m + S}
    """
    return (Vmax * S) / (Km + S)

def haldane(S, Vmax, Km, Ks):
    r"""
    V = \frac{V_{\max}\, S}{K_m + S + \frac{S^2}{K_s}}
    """
    return (Vmax * S) / (Km + S + (S**2 / Ks))

def hill(S, Vmax, Kd, n):
    r"""
    V = \frac{V_{\max}\, S^n}{K_d^n + S^n}
    """
    return (Vmax * (S**n)) / (Kd**n + S**n)
