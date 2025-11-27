import numpy as np

def hill_generalizado_mezcla(X, Vmax, K_half, n, beta):
    r"""
    v = \frac{V_{\max}\, (S_1 + \beta S_2)^n}{K_{\mathrm{half}}^{\,n} + (S_1 + \beta S_2)^n}
    """
    S1, S2 = X  # Desempaquetamos las variables
    
    # Ecuación suministrada:
    # v = Vmax * (S1 + beta*S2)**n / (K_half**n + (S1 + beta*S2)**n)
    
    termino_comun = S1 + (beta * S2)
    numerador = Vmax * (termino_comun**n)
    denominador = (K_half**n) + (termino_comun**n)
    
    return numerador / denominador
