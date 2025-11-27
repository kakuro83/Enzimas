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

def adair_simple(S, Vmax, K1, K2):
    r"""
    V = V_{\max}\, \frac{K_1 S + 2 K_1 K_2 S^2}{1 + K_1 S + K_1 K_2 S^2}
    """
    num = K1*S + 2*K1*K2*(S**2)
    den = 1 + K1*S + K1*K2*(S**2)
    Y = num / den
    return Vmax * Y
