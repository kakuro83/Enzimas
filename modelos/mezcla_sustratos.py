import numpy as np

def hill_generalizado_mezcla(X, Vmax, Kh, n, beta):
    r"""
    V = \frac{V_{\max}\, (S_1 + \beta S_2)^n}{K_{\mathrm{h}}^{\,n} + (S_1 + \beta S_2)^n}
    """
    S1, S2 = X  # Desempaquetamos las variables
    
    # Ecuación suministrada:
    # v = Vmax * (S1 + beta*S2)**n / (Kh**n + (S1 + beta*S2)**n)
    
    termino_comun = S1 + (beta * S2)
    numerador = Vmax * (termino_comun**n)
    denominador = (Kh**n) + (termino_comun**n)
    
    return numerador / denominador
