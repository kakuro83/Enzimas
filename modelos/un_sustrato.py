import numpy as np

def michaelis_menten(S, Vmax, Km):
    r"""
    $V = \frac{V_{\max}\, S}{K_m + S}$
    """
    return (Vmax * S) / (Km + S)

def haldane(S, Vmax, Km, Ki):
    """
    Modelo de Haldane (Inhibición por sustrato).
    S: Concentración de sustrato
    Vmax: Velocidad máxima
    Km: Constante de saturación
    Ki: Constante de inhibición
    """
    return (Vmax * S) / (Km + S + (S**2 / Ki))

def hill(S, Vmax, K05, n):
    """
    Modelo de Hill (Cooperatividad).
    S: Concentración de sustrato
    Vmax: Velocidad máxima
    K05: Constante de semi-saturación (equivalente a Km)
    n: Coeficiente de Hill
    """
    return (Vmax * (S**n)) / (K05**n + S**n)
