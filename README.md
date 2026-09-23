# 🧪 Ajuste de Cinética Enzimática — Enzimas App

Aplicación web desarrollada con Python y Streamlit para ajustar datos experimentales de velocidad a modelos de cinética enzimática mediante regresión no lineal. Está orientada al trabajo de estudiantes e investigadores y permite analizar una concentración de sustrato o dos variables independientes.

## Funcionalidades

- Dos modalidades: **Un solo sustrato** y **Dos sustratos o con efectos de inhibidores/cofactores (Doble Variable)**.
- Etiquetas personalizables para las columnas, incluyendo nombres y unidades.
- Orden de columnas configurable antes de pegar los datos.
- Ingreso mediante una tabla editable o un área de texto, útil desde celulares.
- Valores iniciales configurables y opción de fijar parámetros del modelo.
- Visualización de la ecuación, parámetros y métricas del ajuste: R², RMSE, MAE y AIC.
- Curva 2D para un sustrato y superficie 3D interactiva para doble variable.
- Descarga de parámetros en CSV y de la gráfica 2D en PNG a 300 dpi.

## Instalación y ejecución local

Se requiere Python y pip. El repositorio no fija una versión de Python ni versiones de las dependencias.

```bash
git clone https://github.com/kakuro83/Enzimas.git
cd Enzimas
python -m venv .venv
```

Activa el entorno virtual:

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

Instala las dependencias e inicia la aplicación desde la carpeta del repositorio:

```bash
python -m pip install -r requirements.txt
python -m streamlit run App.py
```

Abre en el navegador la dirección que indique Streamlit.

### Dependencias

[requirements.txt](requirements.txt) incluye: Streamlit, pandas, NumPy, SciPy, Matplotlib, openpyxl, scikit-learn y Plotly. Aunque se incluye openpyxl, la interfaz actual recibe datos por pegado o edición; no tiene un cargador de archivos Excel.

### Despliegue en Streamlit Community Cloud

Al configurar el despliegue, selecciona este repositorio, la rama que quieras publicar y **`App.py`** como archivo principal. Mantén `requirements.txt` y la carpeta `modelos/` en la estructura del proyecto.

## Uso

1. **Selecciona la modalidad.** Usa un sustrato para datos de velocidad frente a una concentración; usa doble variable para dos sustratos o para sustrato e inhibidor/cofactor, según el modelo elegido.
2. **Define las etiquetas.** Utiliza nombres distintos para velocidad, sustrato y segunda variable. Puedes añadir las unidades; la aplicación no realiza conversiones de unidades.
3. **Configura el orden de pegado.**
   - En un sustrato, selecciona el orden en la lista desplegable.
   - En doble variable, mueve las etiquetas con los botones **←** y **→**.
4. **Ingresa los datos.**
   - Copia las celdas numéricas de Excel o Sheets y pégalas desde la primera celda de la tabla.
   - También puedes usar **Pegar datos desde celular** y pulsar **Cargar datos pegados**.
5. **Selecciona el modelo.** La aplicación muestra su ecuación. Para **Cleland (Dinámico)**, elige un orden entre **1 y 10**; el valor inicial es 2.
6. **Configura el ajuste**, si es necesario, en **Opciones Avanzadas: Valores Iniciales y Parámetros Fijos**. Introduce valores iniciales y marca **Fijar** en los parámetros que no deban estimarse.
7. Pulsa **Ejecutar ajuste de datos** y revisa los parámetros, estadísticas y gráficos.
8. Descarga los parámetros mediante **Parámetros CSV** y, en un sustrato, la imagen mediante **Descargar Gráfica**.

### Formato de los datos

Cada fila representa una observación. Pega únicamente valores, sin encabezados, en el orden seleccionado.

El área de texto admite columnas separadas por tabulaciones, espacios o punto y coma, y admite punto o coma decimal. No uses comas como separador de columnas: se interpretan como separador decimal. Si pegas más columnas de las esperadas, esta vía conserva solo las primeras dos o tres, según la modalidad.

Ejemplo para el orden **Velocidad | Sustrato**:

```text
0.20;0.50
0.33;1.00
0.50;2.00
0.67;4.00
0.80;8.00
```

Antes del ajuste, se excluyen las filas incompletas o con valores que no se puedan convertir a números. Se requieren:

- Al menos **3 observaciones válidas**.
- Datos finitos, sin infinitos.
- Al menos **2 valores distintos** de concentración en un sustrato.
- Al menos **2 valores distintos en cada variable** en doble variable.

El mínimo de tres observaciones es una validación de entrada; no garantiza información suficiente para estimar todos los parámetros de modelos complejos.

**Limpiar Datos** reinicia la tabla y elimina los resultados almacenados. Cambiar de modalidad también reinicia los datos. Tras modificar datos, orden del modelo o parámetros, vuelve a ejecutar el ajuste para actualizar los resultados.

## Modelos disponibles

### Un solo sustrato

Implementados en [modelos/un_sustrato.py](modelos/un_sustrato.py).

| Modelo | Parámetros | Descripción de la implementación |
| --- | --- | --- |
| Michaelis-Menten | `Vmax`, `Km` | Cinética de saturación de Michaelis-Menten. |
| Haldane | `Vmax`, `Km`, `Ki` | Incluye inhibición por sustrato mediante el término `S²/Ki`. |
| Hill | `Vmax`, `K05`, `n` | Ecuación de Hill con exponente ajustable. |
| Adair simplificado | `Vmax`, `a`, `b`, `c`, `d` | Expresión racional fija con términos hasta `S⁴`. |
| Isoenzimas Michaelis-Menten | `Vmax1`, `Km1`, `Vmax2`, `Km2` | Suma de dos contribuciones Michaelis-Menten. |
| Michaelis-Menten y lineal | `Vmax`, `Km`, `k_ns` | Componente saturable más el término lineal `k_ns·S`. |
| Bifásica dos sitios Hill | `Vmax1`, `K1`, `n1`, `Vmax2`, `K2`, `n2` | Suma de dos contribuciones de Hill. |

Adair simplificado tiene un número fijo de parámetros; no es un modelo de orden variable en esta versión.

### Doble variable

Implementados en [modelos/dos_sustratos.py](modelos/dos_sustratos.py). La primera y segunda variables se interpretan según el modelo:

| Modelo | Variables | Parámetros |
| --- | --- | --- |
| Ping-Pong Bi-Bi | Sustratos `S_A` y `S_B` | `Vmax`, `KA`, `KB` |
| Hill generalizado mezcla | Concentraciones `S1` y `S2`, combinadas como `S1 + beta·S2` | `Vmax`, `Kh`, `n`, `beta` |
| Inhibición mixta | Sustrato `S1` e inhibidor `S2` | `Vmax`, `Km`, `Ki`, `Kip` |
| Cleland (Dinámico) | Variables `S1` y `S2` | `Vmax` y pares `Ai`, `Bi` para cada orden |

La clase Cleland implementa la siguiente forma racional polinomial:

$$
v = \frac{V_{\max} S_1 S_2}{1 + S_1 S_2 + \sum_{i=1}^{n} A_i S_1^i + \sum_{i=1}^{n} B_i S_2^i}
$$

El término `K_base` está fijado internamente en **1.0** y no se ajusta desde la interfaz. Para un orden `n`, el modelo tiene `1 + 2n` parámetros. Esta implementación es una forma generalizada; su interpretación debe corresponder al sistema experimental que se estudie.

## Ajuste y resultados

La aplicación utiliza `scipy.optimize.curve_fit`, con hasta **500 000 evaluaciones** y límites de **0 a infinito** para los parámetros libres. Los parámetros marcados como fijos conservan el valor introducido. Si todos están fijos, se evalúa el modelo sin realizar optimización.

Los valores iniciales influyen en la convergencia. Si el ajuste falla, revisa los datos, las unidades, los valores iniciales y la cantidad de parámetros libres.

| Resultado | Contenido |
| --- | --- |
| Parámetros | Valores ajustados o fijados; descarga como `params.csv`. |
| R² | Coeficiente de determinación. |
| RMSE | Raíz del error cuadrático medio. |
| MAE | Error absoluto medio. |
| AIC | Calculado como `N·ln(RSS/N) + 2·k`, donde `k` es el número de parámetros libres más uno. Si `RSS = 0`, se muestra `−∞`. |
| Gráfica 2D | Datos experimentales y curva del modelo; descarga como `plot.png`. |
| Gráfica 3D | Superficie del modelo y puntos experimentales; permite rotar la vista con el mouse. |

Las gráficas abarcan el intervalo de las concentraciones ingresadas. La interfaz no presenta intervalos de confianza de los parámetros ni ofrece un botón propio para exportar la superficie 3D. Las métricas describen el ajuste y deben interpretarse junto con la pertinencia del modelo y el diseño experimental.

## Estructura del repositorio

```text
Enzimas/
├── App.py                    # Interfaz, ingreso de datos, ajuste y resultados
├── README.md                 # Documentación
├── requirements.txt          # Dependencias
└── modelos/
    ├── __init__.py
    ├── un_sustrato.py         # Modelos de una variable
    └── dos_sustratos.py       # Modelos de doble variable y clase Cleland
```

### Agregar modelos

La aplicación descubre automáticamente las funciones y clases definidas en los dos módulos de modelos, sin necesidad de modificar `App.py`. Los nombres de las funciones se muestran con espacios en lugar de guiones bajos; las clases añaden el sufijo **(Dinámico)**.

Para un modelo de parámetros fijos:

1. Define una función en el módulo correspondiente.
2. Usa el primer argumento para los datos: `S` en un sustrato o `X` en doble variable, desempaquetando `S1, S2 = X`.
3. Declara explícitamente los parámetros restantes en la firma; la interfaz obtiene sus nombres mediante `inspect.signature`.
4. Incluye únicamente la ecuación LaTeX en el docstring, preferiblemente como raw string.
5. Asegura que la función opere sobre arreglos NumPy y devuelva una predicción por observación.

Para un modelo dinámico, define una clase que reciba el orden `n` y tenga un método `obtener_funcion()`. Este debe devolver una función con firma explícita y docstring de la ecuación, como en `Cleland`.

Evita definir funciones o clases auxiliares al nivel del módulo que no deban aparecer como modelos: el descubrimiento actual también las incluiría.

## Autor y licencia

**Autor:** Gerardo Caicedo.

El código está disponible públicamente. El repositorio actualmente no incluye un archivo de licencia; no se especifican aquí permisos de reutilización o distribución.
