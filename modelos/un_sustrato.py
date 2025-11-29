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

def Adair(a, Vmax, K1, K2, K3, K4):
    r"""
    v = V_{\max} \cdot \frac{\frac{a}{K_1} + \frac{3a^2}{K_1 K_2} + \frac{3a^3}{K_1 K_2 K_3} + \frac{a^4}{K_1 K_2 K_3 K_4}}{1 + \frac{4a}{K_1} + \frac{6a^2}{K_1 K_2} + \frac{4a^3}{K_1 K_2 K_3} + \frac{a^4}{K_1 K_2 K_3 K_4}}
    """
    # Términos individuales de la fracción (a / K...)
    t1 = a / K1
    t2 = (a**2) / (K1 * K2)
    t3 = (a**3) / (K1 * K2 * K3)
    t4 = (a**4) / (K1 * K2 * K3 * K4)
    
    # Numerador según la imagen (Coeficientes: 1, 3, 3, 1)
    numerador_y = t1 + (3 * t2) + (3 * t3) + t4
    
    # Denominador según la imagen (Coeficientes: 1, 4, 6, 4, 1)
    denominador_y = 1 + (4 * t1) + (6 * t2) + (4 * t3) + t4
    
    # Fracción de saturación (y)
    y = np.divide(numerador_y, denominador_y, out=np.zeros_like(numerador_y), where=denominador_y != 0)
    
    # Velocidad final
    return Vmax * y
