import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from io import BytesIO
import inspect
import plotly.graph_objects as go # Importamos Plotly para gráficos 3D interactivos

# --- IMPORTACIÓN DE MÓDULOS ---
# Importamos solo los dos módulos que quedan
try:
    import modelos.un_sustrato as mod_un_sustrato
    import modelos.dos_sustratos as mod_dos_sustratos
except ImportError as e:
    st.error(f"Error importando módulos: {e}. Asegúrate de que 'modelos/un_sustrato.py' y 'modelos/dos_sustratos.py' existan.")
    st.stop()

def get_models_from_module(module):
    """Obtiene funciones y Clases dinámicas del módulo."""
    models = {}
    
    # 1. Buscar Funciones normales
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if func.__module__ == module.__name__:
            display_name = name.replace("_", " ").title()
            models[display_name] = func
            
    # 2. Buscar Clases (Modelos dinámicos como Adair)
    for name, cls in inspect.getmembers(module, inspect.isclass):
        if cls.__module__ == module.__name__:
            # Eliminamos la lista de exclusión innecesaria. Cualquier clase definida en el módulo 
            # es considerada un modelo dinámico (Ej. Adair) para ser listado.
            display_name = name.replace("_", " ").title() + " (Dinámico)"
            models[display_name] = cls
            
    return models

# Función para generar el DataFrame inicial vacío
def get_empty_data_df(col_v_name, col_s1_name, col_s2_name=None, num_rows=5):
    if col_s2_name:
        data = {col_v_name: [None]*num_rows, col_s1_name: [None]*num_rows, col_s2_name: [None]*num_rows}
    else:
        data = {col_v_name: [None]*num_rows, col_s1_name: [None]*num_rows}
    return pd.DataFrame(data)

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Ajuste de Cinética Enzimática", layout="centered")
st.title("Ajuste de Modelos Enzimáticos")

# Inicializar estados de sesión
if 'resultados' not in st.session_state:
    st.session_state.resultados = None
if 'experimental_data' not in st.session_state:
    st.session_state.experimental_data = pd.DataFrame()
if 'modalidad_last' not in st.session_state:
    st.session_state.modalidad_last = ""
if 'col_names_last' not in st.session_state:
    st.session_state.col_names_last = {} # Para rastrear los nombres de las columnas

# --- 1. SELECCIÓN DE MODALIDAD ---
modalidad = st.selectbox(
    "Seleccione la modalidad de trabajo:",
    [
        "Un solo sustrato",
        "Dos sustratos o con efectos de inhibidores/cofactores (Doble Variable)"
    ]
)

# --- 2. DATOS ---
st.subheader("Ingreso de Datos Experimentales")
st.info("💡 Tip: Copia tus datos de Excel y pégalos en la primera celda (Ctrl+V). Los campos vacíos serán ignorados. Solo se aceptan valores numéricos.")

# --- Define columns and template based on modality ---
c_names = st.columns(3 if modalidad == "Un solo sustrato" else 3) 
# Columna de Velocidad (Ahora editable)
with c_names[0]:
    col_v_name = st.text_input("Etiqueta Velocidad:", value="Velocidad") # Etiqueta cambiada
# Columnas de Sustrato(s)
if modalidad == "Un solo sustrato":
    with c_names[1]:
        col_s1_name = st.text_input("Etiqueta Sustrato:", value="Sustrato") # Etiqueta cambiada
    cols = [col_v_name, col_s1_name]
    col_s2_name = None 
else:
    with c_names[1]:
        col_s1_name = st.text_input("Etiqueta Sustrato principal:", value="Sustrato 1") # Etiqueta cambiada
    with c_names[2]:
        col_s2_name = st.text_input("Etiqueta Sustrato/Inhibidor/Cofactor:", value="Variable 2") # Etiqueta cambiada
    cols = [col_v_name, col_s1_name, col_s2_name]

# Generar el DataFrame de plantilla según la modalidad
data_template_df = get_empty_data_df(col_v_name, col_s1_name, col_s2_name)

# --- Column Configuration to enforce number type ---
# El formato "%.4f" mostrará 4 decimales, usando notación científica para números muy grandes o pequeños.
col_config = {
    col_v_name: st.column_config.NumberColumn(col_v_name, format="%.4f"),
    col_s1_name: st.column_config.NumberColumn(col_s1_name, format="%.4f")
}
if col_s2_name:
    col_config[col_s2_name] = st.column_config.NumberColumn(col_s2_name, format="%.4f")

# --- Session State Management and Data Persistence Fix ---
# Check for modal change (change in number of columns)
is_modal_change = (st.session_state.modalidad_last != modalidad)

# Almacenar los nombres de columnas de la última ejecución para mapeo
old_col_names_map = st.session_state.col_names_last.get(st.session_state.modalidad_last, {})

# Si hay datos en la sesión, intentamos renombrarlos para que persistan.
if not st.session_state.experimental_data.empty:
    session_data = st.session_state.experimental_data.copy()
    
    # 1. Mapeo para renombrar
    rename_mapping = {}
    
    # Obtener los nombres antiguos basados en el ROL fijo
    old_v_name = old_col_names_map.get('v_col')
    old_s1_name = old_col_names_map.get('s1_col')
    old_s2_name = old_col_names_map.get('s2_col')
    
    # Mapeo de Velocidad
    if old_v_name and old_v_name in session_data.columns and old_v_name != col_v_name:
        rename_mapping[old_v_name] = col_v_name
        
    # Mapeo de Sustrato 1
    if old_s1_name and old_s1_name in session_data.columns and old_s1_name != col_s1_name:
        rename_mapping[old_s1_name] = col_s1_name
        
    # Mapeo de Sustrato 2 (solo si existe)
    if col_s2_name:
        if old_s2_name and old_s2_name in session_data.columns and old_s2_name != col_s2_name:
            rename_mapping[old_s2_name] = col_s2_name

    # Apply renaming and reindex
    if rename_mapping:
        session_data.rename(columns=rename_mapping, inplace=True)
        # Asegurar que el DataFrame tiene las columnas correctas
        session_data = session_data.reindex(columns=cols, fill_value=None)
    
    # 2. Resetear si la modalidad cambió fundamentalmente (diferente número de columnas)
    if is_modal_change or len(session_data.columns) != len(cols):
        st.session_state.experimental_data = data_template_df
    else:
        st.session_state.experimental_data = session_data # Persistir los datos renombrados
else:
    # Inicializar datos vacíos
    st.session_state.experimental_data = data_template_df


# Actualizar los nombres de columnas almacenados para la próxima ejecución (usando roles fijos)
col_names_to_save = {'v_col': col_v_name, 's1_col': col_s1_name}
if col_s2_name:
    col_names_to_save['s2_col'] = col_s2_name

st.session_state.col_names_last[modalidad] = col_names_to_save
st.session_state.modalidad_last = modalidad


c_editor, c_button = st.columns([5, 1])

with c_button:
    # Botón de Limpiar Datos
    if st.button("Limpiar Datos", key="clear_data_btn", use_container_width=True):
        st.session_state.experimental_data = data_template_df
        st.session_state.resultados = None # Limpiar resultados anteriores
        st.rerun()

with c_editor:
    df_edited = st.data_editor(
        st.session_state.experimental_data,
        num_rows="dynamic",
        use_container_width=True,
        column_config=col_config,
        key="data_input_editor" # Key is important for state management
    )

# Actualizar el estado con los datos editados
st.session_state.experimental_data = df_edited


# Limpieza y preparación de DataFrame final
df = st.session_state.experimental_data.copy()
df = df.dropna(how='all').copy()
df = df.dropna(subset=[col_v_name]) # Usar el nombre de velocidad dinámico
for col in cols:
    if col in df.columns: 
        df[col] = pd.to_numeric(df[col], errors='coerce') 
df = df.dropna()

# --- 3. SELECCIÓN DE MODELO ---
st.divider()
st.subheader("Configuración del Ajuste")

if modalidad == "Un solo sustrato": model_source = mod_un_sustrato
else: model_source = mod_dos_sustratos # Usa dos_sustratos para ambas variantes multisustrato

model_options = get_models_from_module(model_source)
nombre_modelo_sel = st.selectbox("Seleccione el modelo cinético:", list(model_options.keys()))
objeto_modelo = model_options[nombre_modelo_sel]

# --- LÓGICA DE MODELOS DINÁMICOS VS FUNCIONES ---
funcion_final = None

# Manejo de Modelos Dinámicos
if inspect.isclass(objeto_modelo):
    st.info(f"Este es un modelo de orden variable. Selecciona el número de términos.")
    orden_n = st.number_input("Orden del Modelo (n):", min_value=1, max_value=10, value=2, step=1)
    
    instancia = objeto_modelo(orden_n) 
    # La clase debe tener un método 'obtener_funcion'
    try:
        funcion_final = instancia.obtener_funcion()
    except AttributeError:
        st.error("Error: La clase dinámica debe tener el método 'obtener_funcion()'.")
        st.stop()
else:
    funcion_final = objeto_modelo

# --- VISUALIZACIÓN DE ECUACIÓN ---
doc_ecuacion = inspect.getdoc(funcion_final)
if doc_ecuacion:
    st.latex(doc_ecuacion.replace("$", "").strip())
else:
    st.caption("Ecuación no disponible o generada dinámicamente.")

# Detección de parámetros
try:
    sig = inspect.signature(funcion_final)
    param_names = list(sig.parameters.keys())[1:] # Excluir la primera variable (X o S)
except ValueError:
    st.error("Error al obtener parámetros de la función. Asegúrate de que la función dinámica haya sido generada correctamente.")
    st.stop()


# --- CONFIGURACIÓN DE PARÁMETROS ---
with st.expander("🛠️ Opciones Avanzadas: Valores Iniciales y Parámetros Fijos"):
    st.caption("Ajusta los valores iniciales o marca 'Fijar' para bloquear una constante.")
    param_settings = {}
    v_max_guess = np.max(df[col_v_name].values) if not df.empty else 1.0 # Usar nombre de velocidad dinámico

    for p in param_names:
        c_lbl, c_val, c_fix = st.columns([1, 2, 1])
        default_val = 1.0
        # Heurística de guesses
        if "Vmax" in p: default_val = float(v_max_guess)
        elif "n" == p or "beta" in p: default_val = 1.0 # beta = 1.0 (neutro)
        elif not df.empty and ("Km" in p or "K_" in p): default_val = float(np.mean(df.iloc[:, 1]))
        elif "a_" in p or "b_" in p: default_val = 0.1
        
        with c_lbl: st.markdown(f"**{p}**")
        with c_val: val = st.number_input(f"Valor", value=default_val, label_visibility="collapsed", key=f"v_{p}_{nombre_modelo_sel}")
        with c_fix: fixed = st.checkbox("Fijar", key=f"f_{p}_{nombre_modelo_sel}")
        param_settings[p] = {"value": val, "fixed": fixed}

# Estética - Se han eliminado los campos de unidades (unidad_v, unidad_s)

# --- 4. EJECUCIÓN ---
if st.button("Ejecutar ajuste de datos", type="primary"):
    if df.empty or len(df) < 3:
        st.error("Datos insuficientes (mínimo 3 puntos).")
    else:
        try:
            # Preparar datos X, Y
            y_data = df[col_v_name].values # Usar nombre de velocidad dinámico
            if modalidad == "Un solo sustrato": 
                x_data = df[col_s1_name].values
            else: 
                # Modalidad de dos variables: pasa como lista de arrays [S1, S2]
                x_data = [df[col_s1_name].values, df[col_s2_name].values]

            # Separar parámetros
            p0, fixed_map, free_keys = [], {}, []
            for p in param_names:
                cfg = param_settings[p]
                if cfg["fixed"]: fixed_map[p] = cfg["value"]
                else:
                    free_keys.append(p)
                    p0.append(cfg["value"])

            # Wrapper para fijar constantes
            def model_wrapper(x, *free_args):
                full_args = []
                idx = 0
                for name in param_names:
                    if name in fixed_map: full_args.append(fixed_map[name])
                    else:
                        full_args.append(free_args[idx])
                        idx += 1
                return funcion_final(x, *full_args)

            # Optimización
            if not free_keys:
                st.info("Todos los parámetros fijos. Solo se calcula R².")
                popt_full = [param_settings[p]["value"] for p in param_names]
            else:
                popt_free, _ = curve_fit(model_wrapper, x_data, y_data, p0=p0, maxfev=10000, bounds=(0, np.inf))
                # Reconstruir lista completa
                popt_full = []
                idx = 0
                for p in param_names:
                    if param_settings[p]["fixed"]: popt_full.append(param_settings[p]["value"])
                    else:
                        popt_full.append(popt_free[idx])
                        idx += 1

            # Estadísticas
            y_pred = funcion_final(x_data, *popt_full)
            r2 = r2_score(y_data, y_pred)
            rmse = np.sqrt(mean_squared_error(y_data, y_pred))
            mae = mean_absolute_error(y_data, y_pred)
            
            rss = np.sum((y_data - y_pred)**2)
            n_samples = len(y_data)
            k_params = len(free_keys) + 1
            aic = n_samples * np.log(rss/n_samples) + 2 * k_params if rss > 0 else -np.inf

            st.session_state.resultados = {
                "modalidad": modalidad, "model_name": nombre_modelo_sel,
                "popt": popt_full, "r2": r2, "rmse": rmse, "mae": mae, "aic": aic,
                "param_names": param_names, "x_data": x_data, "y_data": y_data,
                "v_col": col_v_name, # Store dynamic velocity name
                "s1_col": col_s1_name, 
                "s2_col": col_s2_name
            }
            st.rerun()

        except Exception as e:
            st.error(f"Error en el cálculo: {e}")

# --- 5. RESULTADOS ---
if st.session_state.resultados:
    res = st.session_state.resultados
    
    # Validar consistencia
    if res.get("modalidad") != modalidad or res.get("model_name") != nombre_modelo_sel:
        st.warning("⚠️ Configuración cambiada. Ejecuta de nuevo.")
    else:
        st.success("¡Resultados disponibles!")
        df_p = pd.DataFrame({"Parámetro": res["param_names"], "Valor": res["popt"]})
        df_s = pd.DataFrame({
            "Estadístico": ["R²", "RMSE", "MAE", "AIC"],
            "Valor": [res['r2'], res['rmse'], res['mae'], res['aic']]
        })
        help_txt = "R²: Coef. Determinación (cercano a 1 es mejor).\nRMSE: Raíz Error Cuadrático Medio (misma unidad que Velocidad).\nMAE: Error Absoluto Medio.\nAIC: Criterio Akaike (menor es mejor, penaliza la complejidad)."

        # LAYOUT: Tablas arriba, Gráfico abajo
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("### Parámetros")
            st.dataframe(df_p, hide_index=True, use_container_width=True)
            st.download_button("📥 Parámetros CSV", df_p.to_csv(index=False).encode(), "params.csv")
        
        with c2:
            st.markdown("### Estadísticas")
            st.dataframe(df_s, hide_index=True, use_container_width=True, 
                         column_config={"Estadístico": st.column_config.TextColumn("Métrica", help=help_txt)})
        
        st.divider()
        st.markdown("### Visualización Gráfica")

        if modalidad == "Un solo sustrato":
            # --- GRÁFICO 2D (MATPLOTLIB) ---
            fig, ax = plt.subplots(figsize=(8, 5))
            
            x_vals = res["x_data"]
            ax.scatter(x_vals, res["y_data"], c='blue', label='Experimental', zorder=2, s=50)
            
            x_smooth = np.linspace(min(x_vals), max(x_vals), 100)
            y_smooth = funcion_final(x_smooth, *res["popt"])
            
            ax.plot(x_smooth, y_smooth, c='red', lw=2, label='Modelo', zorder=1)
            # Etiquetas de eje usan el nombre dinámico (se asume que incluye la unidad)
            ax.set_xlabel(f"{res['s1_col']}")
            ax.set_ylabel(f"{res['v_col']}")
            ax.legend()
            ax.grid(True, alpha=0.5, ls="--")
            st.pyplot(fig)
            
            img = BytesIO()
            fig.savefig(img, format='png', dpi=300, bbox_inches='tight')
            st.download_button("📷 Descargar Gráfica", img.getvalue(), "plot.png", "image/png")
            
        else:
            # --- GRÁFICO 3D INTERACTIVO (PLOTLY) ---
            st.markdown("#### Superficie de Respuesta (3D) - Interactivo")
            
            # 1. Preparar el rango de datos para la superficie (mesh)
            s1_exp = res["x_data"][0]
            s2_exp = res["x_data"][1]
            v_exp = res["y_data"]

            s1_line = np.linspace(s1_exp.min(), s1_exp.max(), 50)
            s2_line = np.linspace(s2_exp.min(), s2_exp.max(), 50)
            S1_MESH, S2_MESH = np.meshgrid(s1_line, s2_line)

            # 2. Calcular la superficie del modelo ajustado
            X_MESH = [S1_MESH.ravel(), S2_MESH.ravel()]
            
            try:
                Z_MESH = funcion_final(X_MESH, *res["popt"])
                Z_MESH = Z_MESH.reshape(S1_MESH.shape)
                
                # 3. Construir la figura de Plotly
                fig = go.Figure(data=[
                    # Superficie del Modelo Ajustado
                    go.Surface(z=Z_MESH, x=S1_MESH, y=S2_MESH, 
                               colorscale='Viridis', opacity=0.8, showscale=False,
                               name='Modelo Ajustado'),
                    
                    # Puntos Experimentales
                    go.Scatter3d(x=s1_exp, y=s2_exp, z=v_exp, 
                                 mode='markers', marker=dict(size=5, color='red', opacity=1.0),
                                 name='Datos Experimentales')
                ])

                # 4. Configuración del Layout (Ejes y Reinicio)
                fig.update_layout(
                    scene=dict(
                        # Etiquetas de eje usan el nombre dinámico (se asume que incluye la unidad)
                        xaxis_title=f"{res['s1_col']}",
                        yaxis_title=f"{res['s2_col']}",
                        zaxis_title=f"{res['v_col']}", 
                        aspectmode='auto'
                    ),
                    # Botón de Reinicio de Vista (Custom button)
                    updatemenus=[dict(
                        type="buttons",
                        direction="left",
                        showactive=True,
                        buttons=[dict(
                            label="Reiniciar Vista",
                            method="relayout",
                            # Posición estándar de la cámara para la vista inicial
                            args=[{"scene.camera.up": {'x': 0, 'y': 0, 'z': 1}, 
                                   "scene.camera.center": {'x': 0, 'y': 0, 'z': 0}, 
                                   "scene.camera.eye": {'x': 1.25, 'y': 1.25, 'z': 1.25}}],
                        )]
                    )]
                )
                
                # Renderiza el gráfico interactivo
                st.plotly_chart(fig, use_container_width=True) 
                
                # Nota: Plotly tiene su propio botón de descarga (cámara en la esquina superior derecha)
                st.info("El gráfico 3D es interactivo (clic y arrastrar para girar). Usa el icono de la cámara para descargar.")

            except Exception as e:
                st.error(f"Error al generar la gráfica 3D. El modelo puede ser muy complejo o los datos insuficientes para el rango. Detalle: {e}")
