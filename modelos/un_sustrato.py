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

# --- CLASE DINÁMICA ADAIR ---
class Adair:
    """
    Clase para el Modelo de Adair generalizado con n sitios de unión.
    """
    def __init__(self, n):
        # n es el número de sitios de unión.
        self.n = int(n)
    
    def obtener_funcion(self):
        # 1. Definir nombres de los parámetros dinámicamente
        # Los parámetros son Vmax, K1, K2, ..., Kn
        args_names = ['S', 'Vmax'] + [f'K{i+1}' for i in range(self.n)]
        arg_str = ", ".join(args_names)
        
        # 2. Construir la ecuación LaTeX (solo la fórmula)
        # La ecuación es compleja debido a los productos secuenciales K1*K2...
        
        # Numerador: Sumatoria de i * beta_i * S^i
        num_terms_latex = []
        for i in range(1, self.n + 1):
            k_product = " ".join([f"K_{j}" for j in range(1, i + 1)])
            num_terms_latex.append(f"{i} ({k_product}) S^{i}")
        
        # Denominador: Sumatoria de 1 + beta_i * S^i
        den_terms_latex = ["1"]
        for i in range(1, self.n + 1):
            k_product = " ".join([f"K_{j}" for j in range(1, i + 1)])
            den_terms_latex.append(f"({k_product}) S^{i}")
            
        docstring = rf"v = \frac{{V_{{max}} \left( { ' + '.join(num_terms_latex) } \right)}}{{{self.n} \left( { ' + '.join(den_terms_latex) } \right)}}"
        
        # 3. Generar la función Python ejecutable
        code = f"""
def adair_generada({arg_str}):
    import numpy as np
    
    S = np.array(S)
    
    # Extraer los K's de los argumentos
    # El slice [2:] toma los argumentos después de S y Vmax (i.e., K1, K2, ...)
    params_k = list(locals().values())[2:] 
    
    # Inicialización de Numerador y Denominador
    numerador_sum = 0.0
    denominador_sum = 1.0 # El '1' inicial del denominador
    
    # Coeficientes beta_i = K1 * K2 * ... * Ki
    beta_i = 1.0 
    
    for i in range({self.n}):
        K_i = params_k[i]
        
        # Calcular el producto secuencial de K's (beta_i)
        beta_i *= K_i 
        
        # Termino: beta_i * S^(i+1)
        term = beta_i * (S ** (i + 1))
        
        # Sumatoria del numerador: Sum( (i+1) * beta_i * S^(i+1) )
        numerador_sum += (i + 1) * term
        
        # Sumatoria del denominador: Sum( beta_i * S^(i+1) )
        denominador_sum += term
        
    # Ecuación final: Vmax * (Num / (n * Den))
    denominador_final = {self.n} * denominador_sum

    # Manejo de división por cero
    return np.divide(Vmax * numerador_sum, denominador_final, 
                     out=np.zeros_like(Vmax * numerador_sum), where=denominador_final != 0)

"""
        # Ejecutar el código generado dinámicamente
        local_vars = {}
        exec(code, local_vars)
        func = local_vars['adair_generada']
        func.__doc__ = docstring
        return func
