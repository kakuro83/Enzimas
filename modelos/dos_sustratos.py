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
    v = \frac{V_{\max} S}{K_m \cdot (1 + \frac{I}{K_i}) + S \cdot (1 + \frac{I}{K_i'})}
    """
    # X debe ser una tupla o lista: [S1, S2]
    S1, S2 = X

    # La lógica de manejo de división por cero debe estar dentro de la función Python
    Ki_safe = np.where(Ki < 1e-6, 1e-6, Ki)
    Kip_safe = np.where(Kip < 1e-6, 1e-6, Kip)

    numerador = Vmax * S1
    denominador = Km * (1 + S2 / Ki_safe) + S1 * (1 + S2 / Kip_safe)
    
    return np.divide(numerador, denominador, out=np.zeros_like(numerador), where=denominador!=0)
    
def hill_mezcla_general(X, Vmax, K_S1, n, K_S2):
    r"""
    v = \frac{V_{\max} \cdot S_1^n}{K_{S1}^n + S_1^n + K_{S2} \cdot S_2^n}
    """
    S_1, S_2 = X
    
    # Manejo de potencias
    S1_n = S_1**n
    S2_n = S_2**n
    K_S1_n = K_S1**n
    
    numerador = Vmax * S1_n
    denominador = K_S1_n + S1_n + K_S2 * S2_n
    
    # Manejo de división por cero
    return np.divide(numerador, denominador, out=np.zeros_like(numerador), where=denominador!=0)

# --- CLASE DINÁMICA PARA MODELOS MULTISUSTRATO ---
class Modelo_Interaccion_General:
    """
    Clase para modelos de interacción polinomial con orden variable N.
    """
    def __init__(self, n):
        self.n = int(n)
    
    def obtener_funcion(self):
        # 1. Definir nombres dinámicos: Vmax, A1, B1, A2, B2...
        args_names = ['X', 'Vmax']
        for i in range(self.n):
            args_names.append(f'A{i+1}')
            args_names.append(f'B{i+1}')
            
        arg_str = ", ".join(args_names)
        
        # 2. Construir la ecuación LaTeX (solo la fórmula)
        sum_latex_num = ""
        sum_latex_den = ""
        for i in range(self.n):
            sum_latex_num += f" + A_{i+1} S_1^{i+1}"
            sum_latex_den += f" + B_{i+1} S_2^{i+1}"
            
        docstring = rf"v = \frac{{V_{{max}} S_1 S_2}}{{K_{{base}} + S_1 S_2 {sum_latex_num} {sum_latex_den}}}"
        
        # 3. Generar la función Python ejecutable (que curve_fit entenderá)
        code = f"""
def interaccion_generada({arg_str}):
    import numpy as np
    S1, S2 = X
    
    # K_base es una constante de afinidad simple (fijada en 1.0 para simplicidad del ejemplo)
    K_base = 1.0 
    
    denominador_interaccion = 0.0
    params_a = [{", ".join([f'A{i+1}' for i in range(self.n)])}]
    params_b = [{", ".join([f'B{i+1}' for i in range(self.n)])}]
    
    for i in range({self.n}):
        denominador_interaccion += params_a[i] * (S1 ** (i + 1))
        denominador_interaccion += params_b[i] * (S2 ** (i + 1))
        
    numerador = Vmax * S1 * S2
    # La forma general de un modelo racional con términos de interacción
    denominador = K_base + (S1 * S2) + denominador_interaccion
    
    # Manejo de división por cero
    return np.divide(numerador, denominador, out=np.zeros_like(numerador), where=denominador!=0)

"""
        # Ejecutar el código generado dinámicamente
        local_vars = {}
        exec(code, local_vars)
        func = local_vars['interaccion_generada']
        func.__doc__ = docstring
        return func
