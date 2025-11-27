import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Importación necesaria para gráficos 3D
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
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
st.markdown("##### Parámetros y Valores Iniciales")
st.caption("Ajusta los valores iniciales. Marca **'Fijar'** si quieres que un parámetro (ej. exponente 'n') se mantenga constante y no sea modificado por el ajuste.")

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
        # Usamos claves únicas para evitar conflictos al cambiar de modelo
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

            # Calcular R2
            y_pred = funcion_modelo(x_data, *popt_full)
            r2 = r2_score(y_data, y_pred)

            # GUARDAR EN SESSION STATE
            st.session_state.resultados = {
                "popt": popt_full,
                "r2": r2,
                "param_names": param_names,
                "x_data": x_data,
                "y_data": y_data,
                "s1_col": col_s1_name,
                "s2_col": col_s2_name if len(cols) > 2 else None,
                "s1_vals": df[col_s1_name].values,
                "s2_vals": df[col_s2_name].values if len(cols) > 2 else None
            }
            
            st.rerun() 

        except Exception as e:
            st.error(f"Error en el cálculo: {e}")

# --- 5. MOSTRAR RESULTADOS (PERSISTENTE) ---
if st.session_state.resultados:
    res = st.session_state.resultados
    st.success("¡Resultados disponibles!")
    
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        st.markdown("### Parámetros")
        results_df = pd.DataFrame({
            "Parámetro": res["param_names"],
            "Valor": res["popt"]
        })
        st.dataframe(results_df, hide_index=True)
        st.metric("R²", f"{res['r2']:.4f}")
        
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar Tabla CSV", csv, "constantes.csv", "text/csv")

    with col_res2:
        # Lógica de graficado
        if modalidad == "Un solo sustrato":
            # GRÁFICO 2D (Estándar)
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
            
        else:
            # GRÁFICO 3D (Multisustrato)
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection='3d')
            
            s1_vals = res["s1_vals"]
            s2_vals = res["s2_vals"]
            
            # 1. Puntos Experimentales (Scatter 3D)
            ax.scatter(s1_vals, s2_vals, res["y_data"], c='blue', marker='o', label='Experimental', s=40, depthshade=False)
            
            # 2. Superficie del Modelo
            # Creamos una malla (grid) que cubra el rango de datos
            s1_range = np.linspace(min(s1_vals), max(s1_vals), 30)
            s2_range = np.linspace(min(s2_vals), max(s2_vals), 30)
            S1_MESH, S2_MESH = np.meshgrid(s1_range, s2_range)
            
            # Calculamos la velocidad en cada punto de la malla
            # Flatten para pasar al modelo y luego reshape para graficar
            Z_MESH = funcion_modelo([S1_MESH.ravel(), S2_MESH.ravel()], *res["popt"])
            Z_MESH = Z_MESH.reshape(S1_MESH.shape)
            
            # Graficamos la superficie con un mapa de color 'viridis'
            ax.plot_surface(S1_MESH, S2_MESH, Z_MESH, cmap='viridis', alpha=0.6, edgecolor='none')
            
            # Etiquetas
            ax.set_xlabel(f"{res['s1_col']} ({unidad_s})")
            ax.set_ylabel(f"{res['s2_col']} ({unidad_s})")
            ax.set_zlabel(f"Velocidad ({unidad_v})")
            ax.set_title(f"Ajuste 3D - {nombre_modelo}", fontsize=10)
            
            # Ajuste de vista inicial
            ax.view_init(elev=20, azim=45)
            
            st.pyplot(fig)

        # Botón de descarga común
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button("📷 Descargar Gráfica", buf.getvalue(), "grafica.png", "image/png")
