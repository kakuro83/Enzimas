import numpy as np

def ping_pong_bi_bi(X, Vmax, Km_A, Km_B):
    """
    Mecanismo Ping-Pong Bi-Bi (común en transesterificación de lipasas).
    X: Lista o array conteniendo [S_A, S_B]
       S_A: Primer sustrato (ej. éster/triglicérido)
       S_B: Segundo sustrato (ej. alcohol)
    Vmax: Velocidad máxima
    Km_A: Constante de Michaelis para A
    Km_B: Constante de Michaelis para B
    """
    S_A, S_B = X # Desempaquetar
    
    # Ecuación de velocidad inicial para Ping-Pong Bi-Bi
    numerador = Vmax * S_A * S_B
    denominador = (Km_B * S_A) + (Km_A * S_B) + (S_A * S_B)
    
    return numerador / denominador
