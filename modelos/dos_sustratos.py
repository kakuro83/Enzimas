import numpy as np

def ping_pong_bi_bi(X, Vmax, KA, KB):
    r"""
    v = \frac{V_{\max}\, A\, B}{K_A B + K_B A + S_A S_B}
    """
    S_A, S_B = X # Desempaquetar
    
    # Ecuación de velocidad inicial para Ping-Pong Bi-Bi
    numerador = Vmax * S_A * S_B
    denominador = (KB * S_A) + (KA * S_B) + (S_A * S_B)
    
    return numerador / denominador

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
