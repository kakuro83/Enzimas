import numpy as np

# ... (Tus funciones anteriores: michaelis_menten, haldane, hill) ...
def michaelis_menten(S, Vmax, Km):
    r"""
    V = \frac{V_{\max} S}{K_m + S}
    """
    return (Vmax * S) / (Km + S)

def haldane(S, Vmax, Km, Ki):
    r"""
    V = \frac{V_{\max} S}{K_m + S + S^2/K_i}
    """
    return (Vmax * S) / (Km + S + (S**2 / Ki))

def hill(S, Vmax, K05, n):
    r"""
    V = \frac{V_{\max} S^n}{K_{0.5}^n + S^n}
    """
    return (Vmax * (S**n)) / (K05**n + S**n)

def adair_simplificado(S, Vmax, a, b, c, d):
    r"""
    v = V_{\max} \frac{a S + 3 b S^2 + 3 c S^3 + d S^4}{1 + 4 a S + 6 b S^2 + 4 c S^3 + d S^4}
    """
    # Precalcular potencias de S
    S2 = S**2
    S3 = S**3
    S4 = S**4
    
    # Numerador: (a*S) + (3*b*S^2) + (3*c*S^3) + (d*S^4)
    numerador = (a * S) + (3 * b * S2) + (3 * c * S3) + (d * S4)
    
    # Denominador: 1 + (4*a*S) + (6*b*S^2) + (4*c*S^3) + (d*S^4)
    denominador = 1 + (4 * a * S) + (6 * b * S2) + (4 * c * S3) + (d * S4)
    
    # Velocidad final
    return np.divide(Vmax * numerador, denominador, out=np.zeros_like(numerador), where=denominador != 0)

def isoenzimas_michaelis_menten(S, Vmax1, Km1, Vmax2, Km2):
    r"""
    v = \frac{V_{max1} S}{K_{m1} + S} + \frac{V_{max2} S}{K_{m2} + S}
    """
    mm1 = (Vmax1 * S) / (Km1 + S)
    mm2 = (Vmax2 * S) / (Km2 + S)
    return mm1 + mm2

def michaelis_menten_y_lineal(S, Vmax, Km, k_ns):
    r"""
    v = \frac{V_{max} S}{K_m + S} + k_{ns} S
    """
    mm = (Vmax * S) / (Km + S)
    lineal = k_ns * S
    return mm + lineal

def inhibicion_sustrato_parcial(S, Vmax, Km, Ki, beta):
    r"""
    Modelo de inhibición por sustrato parcial (común en proteasas).
    El complejo ESS puede formar producto pero a una velocidad diferente (beta * Vmax).
    
    v = \frac{V_{\max} S (1 + \frac{\beta S}{K_i})}{K_m + S (1 + \frac{S}{K_i})}
    
    - Si beta = 0: Se reduce al modelo de Haldane (Inhibición total).
    - Si beta > 1: Activación por sustrato.
    - Si 0 < beta < 1: Inhibición parcial.
    """
    Ki_safe = np.where(Ki < 1e-6, 1e-6, Ki) # Evitar división por cero
    
    numerador = Vmax * S * (1 + (beta * S / Ki_safe))
    denominador = Km + S * (1 + (S / Ki_safe))
    
    return numerador / denominador
