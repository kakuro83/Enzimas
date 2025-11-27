import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Eliminamos mpl_toolkits.mplot3d ya que no se usará
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from io import BytesIO
import inspect

# --- IMPORTACIÓN DE MÓDULOS ---
try:
    import modelos.un_sustrato as mod_un_sustrato
    import modelos.mezcla_sustratos as mod_mezcla
    import modelos.dos_sustratos as mod_dos
except ImportError as e:
    st.error(f"Error importando módulos: {e}. Verifica la carpeta 'modelos'.")
    st.stop()

def get_models_from_module(module):
    models = {}
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if func.__module__ == module.__name__:
            display_name = name.replace("_", " ").title()
            models[display_name] = func
    return models

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Ajuste de Cinética Enzimática", layout="centered")
st.title("Ajuste de Modelos Enzimáticos")

# Inicializar estado para guardar resultados y evitar reinicios al descargar
if 'resultados' not in st.session_state:
    st.session_state.resultados = None

# --- 1. SELECCIÓN DE MODALIDAD ---
modalidad = st.selectbox(
    "Seleccione la modalidad de trabajo:",
    [
        "Un solo sustrato",
        "Diferentes fuentes de un sustrato (Mezcla)",
        "Dos sustratos (Bi-Sustrato)"
    ]
)

# --- 2. DATOS ---
st.subheader("Ingreso de Datos Experimentales")
st.info("💡 Tip: Copia tus datos de Excel y pégalos en la primera celda (Ctrl+V).")

col_config = {}
data_template = {}

if modalidad == "Un solo sustrato":
    col_s1_name = st.text_input("Nombre de la columna de Sustrato:", value="Sustrato")
    cols = ["Velocidad", col_s1_name]
    data_template = {"Velocidad": [None]*5, col_s1_name: [None]*5}
else:
    c1, c2 = st.columns(2)
    with c1:
        col_s1_name = st.text_input("Nombre Sustrato 1:", value="Sustrato 1")
    with c2:
        col_s2_name = st.text_input("Nombre Sustrato 2:", value="Sustrato 2")
    cols = ["Velocidad", col_s1_name, col_s2_name]
    data_template = {"Velocidad": [None]*5, col_s1_name: [None]*5, col_s2_name: [None]*5}

df_input = pd.DataFrame(data_template)
df_edited = st.data_editor(df_input, num_rows="dynamic", use_container_width=True)

# Limpieza
df = df_edited.dropna(how='all').copy()
df = df.dropna(subset=["Velocidad"])
for col in cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna()

# --- 3. MODELO Y PARÁMETROS ---
st.divider()
st.subheader("Configuración del Ajuste")

model_options = {}
if modalidad == "Un solo sustrato":
    model_options = get_models_from_module(mod_un_sustrato)
elif modalidad == "Diferentes fuentes de un sustrato (Mezcla)":
    model_options = get_models_from_module(mod_mezcla)
else:
    model_options = get_models_from_module(mod_dos)

nombre_modelo = st.selectbox("Seleccione el modelo cinético:", list(model_options.keys()))
funcion_modelo = model_options[nombre_modelo]

# Detección de parámetros
sig = inspect.signature(funcion_modelo)
param_names = list(sig.parameters.keys())[1:] 

# --- CONFIGURACIÓN DE PARÁMETROS (Manual/Fijo) ---
# Envolvemos en un expander para mantener la interfaz limpia
with st.expander("🛠️ Opciones Avanzadas: Valores Iniciales y Parámetros Fijos"):
    st.markdown("##### Estimación de Valores Iniciales")
    st.caption("El algoritmo intenta adivinar los valores iniciales. Aquí puedes modificarlos manualmente o **fijar** una constante (ej. 'n' de Hill) para que no cambie durante el ajuste.")

    # Diccionario para guardar configuración del usuario
    param_settings = {}
    v_max_guess = np.max(df["Velocidad"].values) if not df.empty else 1.0

    for p in param_names:
        col_lbl, col_val, col_fix = st.columns([1, 2, 1])
        
        # Heurística simple para valor por defecto
        default_val = 1.0
        if "Vmax" in p: default_val = float(v_max_guess)
        elif "n" in p: default_val = 1.0
        elif not df.empty and ("Km" in p or "K_" in p): 
            default_val = float(np.mean(df.iloc[:, 1]))
        
        with col_lbl:
            st.markdown(f"**{p}**")
        with col_val:
            val = st.number_input(f"Valor {p}", value=default_val, key=f"val_{p}_{nombre_modelo}")
        with col_fix:
            fixed = st.checkbox("Fijar", key=f"fix_{p}_{nombre_modelo}")
        
        param_settings[p] = {"value": val, "fixed": fixed}

# Estética Gráfica
st.markdown("##### Estética")
c_units1, c_units2 = st.columns(2)
with c_units1:
    unidad_v = st.text_input("Unidades Velocidad:", value="mM/min")
with c_units2:
    unidad_s = st.text_input("Unidades Sustrato:", value="mM")

# --- 4. EJECUCIÓN ---
if st.button("Ejecutar ajuste de datos", type="primary"):
    if df.empty or len(df) < 3:
        st.error("Por favor ingresa al menos 3 puntos de datos válidos.")
    else:
        try:
            # Preparar datos
            y_data = df["Velocidad"].values
            if modalidad == "Un solo sustrato":
                x_data = df[col_s1_name].values
            else:
                x_data = [df[col_s1_name].values, df[col_s2_name].values]

            # Separar parámetros libres y fijos
            free_params_keys = []
            p0 = []
            fixed_params_map = {}

            for p in param_names:
                setting = param_settings[p]
                if setting["fixed"]:
                    fixed_params_map[p] = setting["value"]
                else:
                    free_params_keys.append(p)
                    p0.append(setting["value"])

            # Definir función wrapper para curve_fit que maneje fijos
            def model_wrapper(x, *free_args):
                full_args = []
                free_idx = 0
                for name in param_names:
                    if name in fixed_params_map:
                        full_args.append(fixed_params_map[name])
                    else:
                        full_args.append(free_args[free_idx])
                        free_idx += 1
                return funcion_modelo(x, *full_args)

            # Optimización
            if not free_params_keys:
                popt_free = []
                popt_full = [param_settings[p]["value"] for p in param_names]
                st.info("Todos los parámetros están fijos. Se calcula solo R².")
            else:
                popt_free, pcov = curve_fit(
                    model_wrapper, 
                    x_data, 
                    y_data, 
                    p0=p0, 
                    maxfev=10000,
                    bounds=(0, np.inf) 
                )
                
                # Reconstruir parámetros completos
                popt_full = []
                free_idx = 0
                for p in param_names:
                    if param_settings[p]["fixed"]:
                        popt_full.append(param_settings[p]["value"])
                    else:
                        popt_full.append(popt_free[free_idx])
                        free_idx += 1

            # Calcular Estadísticas
            y_pred = funcion_modelo(x_data, *popt_full)
            
            # R2
            r2 = r2_score(y_data, y_pred)
            
            # RMSE
            rmse = np.sqrt(mean_squared_error(y_data, y_pred))
            
            # MAE
            mae = mean_absolute_error(y_data, y_pred)
            
            # AIC
            n = len(y_data)
            rss = np.sum((y_data - y_pred)**2)
            k = len(free_params_keys) + 1 
            
            if rss > 0:
                aic = n * np.log(rss/n) + 2 * k
            else:
                aic = -np.inf 

            # GUARDAR EN SESSION STATE
            st.session_state.resultados = {
                "modalidad": modalidad,
                "model_name": nombre_modelo,
                "popt": popt_full,
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "aic": aic,
                "param_names": param_names,
                "x_data": x_data,
                "y_data": y_data,
                "s1_col": col_s1_name,
                "s2_col": col_s2_name if len(cols) > 2 else None,
            }
            
            st.rerun() 

        except Exception as e:
            st.error(f"Error en el cálculo: {e}")

# --- 5. MOSTRAR RESULTADOS (PERSISTENTE) ---
if st.session_state.resultados:
    res = st.session_state.resultados
    
    # VALIDACIÓN DE CONSISTENCIA
    if res.get("modalidad") != modalidad or res.get("model_name") != nombre_modelo:
        st.info("⚠️ La configuración ha cambiado. Por favor, ejecuta el ajuste nuevamente para actualizar los resultados.")
    else:
        st.success("¡Resultados disponibles!")
        
        # Preparar DataFrames para tablas
        df_params = pd.DataFrame({
            "Parámetro": res["param_names"],
            "Valor": res["popt"]
        })

        df_stats = pd.DataFrame({
            "Estadístico": ["R²", "RMSE", "MAE", "AIC"],
            "Valor": [res['r2'], res['rmse'], res['mae'], res['aic']],
            "Descripción": [
                "Coeficiente de determinación (cercano a 1 es mejor)",
                "Raíz del Error Cuadrático Medio (misma unidad que Velocidad)",
                "Error Absoluto Medio",
                "Criterio de Akaike (menor valor indica mejor modelo)"
            ]
        })

        # Layout condicional
        if modalidad == "Un solo sustrato":
            # LAYOUT UN SUSTRATO
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.markdown("### Parámetros")
                st.dataframe(df_params, hide_index=True, use_container_width=True)
                
                st.markdown("### Estadísticas")
                # Mostramos tabla con configuración de columnas para tooltips
                st.dataframe(
                    df_stats, 
                    hide_index=True, 
                    use_container_width=True,
                    column_config={
                        "Estadístico": st.column_config.TextColumn("Métrica", help="Nombre del indicador estadístico"),
                        "Valor": st.column_config.NumberColumn("Valor", format="%.4f"),
                        "Descripción": st.column_config.TextColumn("Ayuda", width="small")
                    }
                )
                
                csv = df_params.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Tabla Parámetros", csv, "constantes.csv", "text/csv")

            with col_res2:
                # GRÁFICO 2D
                fig, ax = plt.subplots()
                label_y = f"Velocidad ({unidad_v})"
                
                x_vals = res["x_data"]
                ax.scatter(x_vals, res["y_data"], color='blue', label='Experimental', zorder=2)
                
                x_smooth = np.linspace(min(x_vals), max(x_vals), 100)
                y_smooth = funcion_modelo(x_smooth, *res["popt"])
                ax.plot(x_smooth, y_smooth, color='red', label='Modelo', linewidth=2, zorder=1)
                ax.set_xlabel(f"{res['s1_col']} ({unidad_s})")
                ax.set_ylabel(label_y)
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig)
                
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                st.download_button("📷 Descargar Gráfica", buf.getvalue(), "grafica.png", "image/png")
        
        else:
            # LAYOUT MULTISUSTRATO
            st.markdown("### Resultados del Ajuste Multisustrato")
            
            c_table, c_stats = st.columns([1, 1])
            
            with c_table:
                st.markdown("#### Parámetros Cinéticos")
                st.dataframe(df_params, hide_index=True, use_container_width=True)
                
                csv = df_params.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Descargar Parámetros", csv, "resultados_cineticos.csv", "text/csv")
            
            with c_stats:
                st.markdown("#### Bondad de Ajuste")
                # Tabla de estadísticas unificada con tooltips en la columna de descripción
                st.dataframe(
                    df_stats, 
                    hide_index=True, 
                    use_container_width=True,
                    column_config={
                        "Estadístico": st.column_config.TextColumn("Métrica"),
                        "Valor": st.column_config.NumberColumn("Valor", format="%.4f"),
                        "Descripción": st.column_config.TextColumn("Ayuda / Descripción", width="medium")
                    }
                )
