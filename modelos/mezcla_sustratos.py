import numpy as np

def hill_generalizado_mezcla(X, Vmax, K_half, n, beta):
    """
    Modelo de Hill generalizado para mezclas de sustratos.
    X: Lista o array conteniendo [S1, S2] donde:
       S1: Concentración sustrato 1
       S2: Concentración sustrato 2
    Vmax: Velocidad máxima
    K_half: Constante de afinidad media
    n: Coeficiente de forma
    beta: Factor de interacción o contribución de S2
    """
    S1, S2 = X  # Desempaquetamos las variables
    
    # Ecuación suministrada:
    # v = Vmax * (S1 + beta*S2)**n / (K_half**n + (S1 + beta*S2)**n)
    
    termino_comun = S1 + (beta * S2)
    numerador = Vmax * (termino_comun**n)
    denominador = (K_half**n) + (termino_comun**n)
    
    return numerador / denominador
