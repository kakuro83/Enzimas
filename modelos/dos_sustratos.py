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
