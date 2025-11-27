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

# --- NUEVO: MODELO DINÁMICO DE ADAIR ---
class Adair:
    def __init__(self, n):
        self.n = int(n)
    
    def obtener_funcion(self):
        # 1. Creamos la ecuación LaTeX dinámica para mostrarla en pantalla
        numerador_latex = " + ".join([f"{i+1} K_{i+1} S^{i+1}" for i in range(self.n)])
        denominador_latex = "1 + " + " + ".join([f"K_{i+1} S^{i+1}" for i in range(self.n)])
        docstring = rf"V = \frac{{V_{{max}} ({numerador_latex})}}{{{self.n} ({denominador_latex})}}"
        
        # 2. Definimos los nombres de los argumentos dinámicamente
        # Argumentos: S, Vmax, K1, K2, ..., Kn
        args_names = ['S', 'Vmax'] + [f'K{i+1}' for i in range(self.n)]
        
        # 3. Creamos la función Python real usando exec (necesario para que curve_fit vea los nombres)
        # Esto genera código como: "def adair_gen(S, Vmax, K1, K2): ..."
        arg_str = ", ".join(args_names)
        
        # Lógica matemática de Adair (Simplificada: K son constantes de asociación aparentes)
        code = f"""
def adair_generada({arg_str}):
    import numpy as np
    S_arr = np.array(S)
    
    # Construcción de sumatorias
    # Numerador: Sum(i * Ki * S^i)
    num = 0
    den = 1 # El 1 de la fórmula
    
    params_k = [{", ".join([f'K{i+1}' for i in range(self.n)])}]
    
    for i, k_val in enumerate(params_k):
        term = k_val * (S_arr ** (i + 1))
        num += (i + 1) * term
        den += term
        
    return (Vmax * num) / ({self.n} * den)
"""
        # Compilamos la función en un entorno local
        local_vars = {}
        exec(code, local_vars)
        func = local_vars['adair_generada']
        
        # 4. Asignamos el docstring (LaTeX)
        func.__doc__ = docstring
        return func
