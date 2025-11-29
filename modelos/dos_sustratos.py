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

Asegúrate de agregar este código al final de tu archivo **`modelos/dos_sustratos.py`**. Tu aplicación `App.py` lo detectará automáticamente.

```python
import numpy as np

def inhibicion_mixta_s2_variable(X, Vmax, Km, Ki, alpha):
    r"""
    Modelo de inhibición mixta donde $S_2$ actúa como inhibidor variable ($I$).
    
    v = \frac{V_{\max} S_1}{K_m \cdot (1 + \frac{S_2}{K_i \alpha}) + S_1 \cdot (1 + \frac{S_2}{K_i})}
    
    Donde:
    - $S_1$ es el Sustrato.
    - $S_2$ es el Inhibidor ($I$).
    - $V_{\max}$ es la velocidad máxima aparente.
    - $K_m$ es la constante de Michaelis.
    - $K_i$ es la constante de disociación del inhibidor al complejo ES.
    - $\alpha$ (alpha) es el factor de competitividad ($\alpha \ge 0$).
    
    Casos especiales:
    - Si $\alpha = 0$: Inhibición acompetitiva pura.
    - Si $\alpha = 1$: Inhibición no competitiva pura.
    - Si $\alpha = \infty$ (valor grande y fijo): Inhibición competitiva pura.
    """
    # X debe ser una tupla o lista: [S1, S2]
    S1, S2 = X
    
    # Manejo de alpha muy pequeño o cero para evitar división por cero en Ki*alpha
    alpha_safe = np.where(alpha < 1e-6, 1e-6, alpha)
    
    numerador = Vmax * S1
    denominador = Km * (1 + S2 / (Ki * alpha_safe)) + S1 * (1 + S2 / Ki)
    
    return numerador / denominador
