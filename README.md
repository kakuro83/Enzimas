🧪 Aplicación de Ajuste de Cinética Enzimática (Enzimas App)

Esta aplicación web gratuita, construida con Python y Streamlit, permite a estudiantes e investigadores realizar ajustes de regresión no lineal de datos cinéticos experimentales a modelos enzimáticos comunes (Michaelis-Menten, Haldane, Hill, Adair, etc.) y avanzados (multisustrato/inhibición).

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
plotly


Flujo de Trabajo

Selección de Modalidad: Elige una de las dos modalidades: Un solo Sustrato o Doble Variable (Dos Sustratos / Inhibidores).

Etiquetas de Datos: Antes de pegar, define la etiqueta (nombre y unidad) para cada columna (ejemplo: Velocidad (μM/min)).

Ingreso de Datos: Copia tus datos de Excel y pégalos en la primera celda (Ctrl+V).

Selección de Modelo: Elige el modelo a ajustar. Si seleccionas Modelo Cleland (Dinámico), establece el orden de la interacción.

Configuración Avanzada (Opcional):

Ajusta los valores iniciales de la regresión.

Fijar constantes específicas (ej. un coeficiente de Hill, o la concentración de un inhibidor) para que el algoritmo no las ajuste.

Ejecutar Ajuste: Presiona el botón para obtener los resultados.

Análisis de Resultados:

Obtén los valores ajustados de las constantes cinéticas.

Evalúa la Bondad de Ajuste con métricas clave (R², RMSE, MAE, AIC).

Gráficos:

Un solo Sustrato: Gráfica 2D de ajuste de curva.

Doble Variable: Gráfica 3D de Superficie de Respuesta (interactiva con el mouse).

⚙️ Estructura Modular (Para Desarrolladores)

El código está organizado para facilitar la adición de nuevos modelos sin tocar App.py.

1. Modelos con Funciones Simples (.py)

Para agregar un modelo con una ecuación fija (ej. Inhibición No Competitiva), solo necesitas:

Escribir la función con numpy en el archivo correspondiente (ej. modelos/un_sustrato.py).

Documentar la Ecuación: Incluir la ecuación en formato LaTeX dentro de un raw string (r"""...""") como Docstring de la función. Solo la ecuación.

2. Modelos Dinámicos (Clase Cleland)

Para modelos como Adair o el Modelo Cleland Generalizado, donde el número de constantes depende de una variable (el orden $n$), debes crear una Clase que contenga un método obtener_funcion().

Ejemplo de Modelo Cleland (en modelos/dos_sustratos.py):

La clase se llama Cleland en el código y aparece como Cleland (Dinámico) en la interfaz. Permite modelar interacciones complejas de doble variable mediante una estructura polinomial general.

Licencia: Este proyecto es de código abierto.

Autor: Gerardo Caicedo
