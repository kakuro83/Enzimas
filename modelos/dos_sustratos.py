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

def inhibicion_mixta(X, Vmax, Km, Ki, Kip):
    r"""
    v = \frac{V_{\max} S}{K_m \cdot (1 + \frac{I}{K_i \alpha}) + S \cdot (1 + \frac{I}{K_i'})}
    """
    # X debe ser una tupla o lista: [S1, S2]
    S1, S2 = X
    
    # Manejo de alpha muy pequeño o cero para evitar división por cero en Ki*alpha
    alpha_safe = np.where(alpha < 1e-6, 1e-6, alpha)
    
    numerador = Vmax * S1
    denominador = Km * (1 + S2 / Ki) + S1 * (1 + S2 / Kip)
    
    return numerador / denominador
