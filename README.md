🧪 Aplicación de Ajuste de Cinética Enzimática (Enzimas App)

Esta aplicación web gratuita, construida con Python y Streamlit, permite a estudiantes e investigadores realizar ajustes de regresión no lineal de datos cinéticos experimentales a modelos enzimáticos comunes (Michaelis-Menten, Haldane, Hill, Adair, etc.) y avanzados (multisustrato).

La principal ventaja de esta aplicación es su diseño modular, que permite agregar nuevos modelos cinéticos (tanto simples como de orden variable) sin modificar el código principal (App.py).

🚀 Despliegue y Uso

Esta aplicación está diseñada para ser desplegada gratuitamente en Streamlit Community Cloud.

Requisitos Técnicos

El archivo requirements.txt ya incluye todas las librerías necesarias:

streamlit
pandas
numpy
scipy
matplotlib
openpyxl
scikit-learn


Flujo de Trabajo

Selección de Modalidad: Elige si tus datos son de Un solo Sustrato, Mezcla, o Dos Sustratos. Esto define la estructura de la tabla de entrada.

Ingreso de Datos: Copia y pega tus datos de Velocidad (columna fija) y Sustrato(s) directamente desde Excel/CSV a la tabla de datos.

Selección de Modelo: Elige el modelo a ajustar (ej. Michaelis-Menten). Si eliges un modelo dinámico (como Adair), se te pedirá seleccionar el orden.

Configuración Avanzada (Opcional): Usa la sección de opciones avanzadas para:

Ajustar los valores iniciales de la regresión.

Fijar constantes específicas (ej. un coeficiente de Hill, o la concentración inicial de un inhibidor) para que el algoritmo solo ajuste las constantes libres.

Ejecutar Ajuste: Presiona el botón para obtener los resultados.

Análisis de Resultados:

Obtén los valores ajustados de las constantes cinéticas.

Evalúa la Bondad de Ajuste con métricas clave (R², RMSE, MAE, AIC).

Para modelos de un sustrato, visualiza la Gráfica de los puntos experimentales vs. la curva ajustada.

Descarga la tabla de parámetros y la gráfica en formato PNG/CSV.

⚙️ Estructura Modular (Para Desarrolladores)

El código está organizado para facilitar la adición de nuevos modelos sin tocar App.py.

1. Modelos con Funciones Simples (.py)

Para agregar un modelo con una ecuación fija (ej. Inhibición No Competitiva), solo necesitas:

Escribir la función con numpy en el archivo correspondiente (ej. modelos/un_sustrato.py).

Documentar la Ecuación: Incluir la ecuación en formato LaTeX dentro de un raw string (r"""...""") como Docstring de la función.

Ejemplo:

def mi_nuevo_modelo(S, Vmax, Km, Kx):
    r"""
    V = \frac{V_{\max} S}{K_m + S + \frac{S^2}{K_x}}
    """
    return (Vmax * S) / (Km + S + (S**2 / Kx))


El App.py detectará automáticamente el nombre de la función (Mi Nuevo Modelo) y sus parámetros (Vmax, Km, Kx).

2. Modelos de Orden Variable (Clases Dinámicas)

Para modelos como Adair donde el número de constantes depende de una variable (el orden $n$), debes crear una Clase que contenga un método obtener_funcion().

El programa App.py detectará la clase y te preguntará el orden n antes de construir la función matemática final con el número correcto de constantes.

Licencia: Este proyecto es de código abierto.

Autor: [Tu Nombre o Contacto]
