import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from io import BytesIO
import inspect # Necesario para leer funciones dinámicamente

# --- IMPORTACIÓN DE MÓDULOS COMPLETOS ---
try:
    import modelos.un_sustrato as mod_un_sustrato
    import modelos.mezcla_sustratos as mod_mezcla
    import modelos.dos_sustratos as mod_dos
except ImportError as e:
    st.error(f"Error importando módulos: {e}. Verifica la carpeta 'modelos'.")
    st.stop()

# --- FUNCIÓN AUXILIAR PARA CARGA DINÁMICA ---
def get_models_from_module(module):
    models = {}
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if func.__module__ == module.__name__:
            display_name = name.replace("_", " ").title()
            models[display_name] = func
    return models

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Ajuste de Cinética Enzimática", layout="centered")
st.title("Ajuste de Modelos Enzimáticos")

# --- 1. SELECCIÓN DE MODALIDAD ---
modalidad = st.selectbox(
    "Seleccione la modalidad de trabajo:",
    [
        "Un solo sustrato",
        "Diferentes fuentes de un sustrato (Mezcla)",
        "Dos sustratos (Bi-Sustrato)"
    ]
)

# --- 2. CONFIGURACIÓN DE COLUMNAS Y DATOS ---
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
    data_template = {
        "Velocidad": [None]*5,
        col_s1_name: [None]*5,
        col_s2_name: [None]*5
    }

df_input = pd.DataFrame(data_template)
df_edited = st.data_editor(df_input, num_rows="dynamic", use_container_width=True)

# Limpieza de datos
df = df_edited.dropna(how='all').copy()
df = df.dropna(subset=["Velocidad"])
for col in cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna()

# --- 3. SELECCIÓN DE MODELO Y PARÁMETROS ---
st.divider()
st.subheader("Configuración del Ajuste")

model_options = {}
if modalidad == "Un solo sustrato":
    model_options = get_models_from_module(mod_un_sustrato)
elif modalidad == "Diferentes fuentes de un sustrato (Mezcla)":
    model_options = get_models_from_module(mod_mezcla)
else:
    model_options = get_models_from_module(mod_dos)

if not model_options:
    st.warning("No se encontraron modelos en el archivo seleccionado.")
    st.stop()

nombre_modelo = st.selectbox("Seleccione el modelo cinético:", list(model_options.keys()))
funcion_modelo = model_options[nombre_modelo]

# --- DETECCIÓN DE PARÁMETROS ---
# Inspeccionamos la función ANTES de ejecutar para poder mostrar los inputs manuales
sig = inspect.signature(funcion_modelo)
param_names = list(sig.parameters.keys())[1:] # Omitimos el primer argumento (S o X)

# Sección de Valores Iniciales Manuales
st.markdown("##### Valores Iniciales de la Regresión")
use_manual_p0 = st.checkbox("Definir valores iniciales manualmente (Recomendado para parámetros a, b, c)")

manual_p0_values = {}
if use_manual_p0:
    st.caption("Ingresa una estimación inicial para ayudar al algoritmo:")
    cols_p = st.columns(len(param_names))
    for i, p in enumerate(param_names):
        # Creamos un input para cada parámetro detectado
        manual_p0_values[p] = cols_p[i].number_input(f"Inicial para {p}", value=1.0)

# Inputs para unidades gráficas
st.markdown("##### Estética de la Gráfica")
c_units1, c_units2 = st.columns(2)
with c_units1:
    unidad_v = st.text_input("Unidades de Velocidad (Eje Y):", value="mM/min")
with c_units2:
    unidad_s = st.text_input("Unidades de Sustrato (Eje X):", value="mM")

# --- 4. LÓGICA DE EJECUCIÓN ---
if st.button("Ejecutar ajuste de datos", type="primary"):
    if df.empty or len(df) < 3:
        st.error("Por favor ingresa al menos 3 puntos de datos válidos.")
    else:
        try:
            # Preparación de datos Y
            y_data = df["Velocidad"].values
            
            # Preparación de datos X
            v_max_guess = np.max(y_data)
            
            if modalidad == "Un solo sustrato":
                x_data = df[col_s1_name].values
                km_guess = np.mean(x_data)
            else:
                s1_data = df[col_s1_name].values
                s2_data = df[col_s2_name].values
                x_data = [s1_data, s2_data]
                km_guess = np.mean(s1_data)
                km_guess_2 = np.mean(s2_data)

            # --- DEFINICIÓN DE VALORES INICIALES (p0) ---
            p0 = []
            bounds_lower = []
            bounds_upper = []

            for p in param_names:
                bounds_lower.append(0)
                bounds_upper.append(np.inf)
                
                if use_manual_p0:
                    # Usar el valor que el usuario escribió en la cajita
                    p0.append(manual_p0_values[p])
                else:
                    # Lógica Heurística (Automática)
                    if "Vmax" in p or "V_max" in p:
                        p0.append(v_max_guess)
                    elif "Km" in p or "K_" in p or "K05" in p:
                        if "2" in p or "B" in p: 
                             p0.append(np.mean(df[cols[2]].values) if len(cols)>2 else km_guess)
                        else:
                            p0.append(km_guess)
                    elif "n" == p:
                        p0.append(1.0)
                    elif "beta" in p:
                        p0.append(1.0)
                    elif "Ki" in p:
                        p0.append(np.max(df[cols[1]].values))
                    else:
                        # Si se llama 'a', 'b', 'c', entra aquí
                        p0.append(1.0)

            # --- AJUSTE DE CURVA ---
            popt, pcov = curve_fit(
                funcion_modelo, 
                x_data, 
                y_data, 
                p0=p0, 
                bounds=(bounds_lower, bounds_upper), 
                maxfev=10000
            )
            
            # --- CÁLCULO DE R2 ---
            y_pred = funcion_modelo(x_data, *popt)
            r2 = r2_score(y_data, y_pred)

            # --- RESULTADOS ---
            st.success(f"¡Ajuste exitoso usando {nombre_modelo}!")
            
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.markdown("### Parámetros")
                results_df = pd.DataFrame({
                    "Parámetro": param_names,
                    "Valor": popt
                })
                st.dataframe(results_df, hide_index=True)
                st.metric("R²", f"{r2:.4f}")
                
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Descargar Tabla CSV", csv, "constantes.csv", "text/csv")

            with col_res2:
                # --- GRÁFICA ---
                fig, ax = plt.subplots()
                label_y = f"Velocidad ({unidad_v})"
                label_x = ""

                if modalidad == "Un solo sustrato":
                    ax.scatter(x_data, y_data, color='blue', label='Experimental', zorder=2)
                    x_smooth = np.linspace(min(x_data), max(x_data), 100)
                    y_smooth = funcion_modelo(x_smooth, *popt)
                    ax.plot(x_smooth, y_smooth, color='red', label='Modelo', linewidth=2, zorder=1)
                    label_x = f"{col_s1_name} ({unidad_s})"
                    
                else:
                    axis_choice = st.radio("Eje X para gráfica:", [col_s1_name, col_s2_name], horizontal=True)
                    if axis_choice == col_s1_name:
                        x_plot = s1_data
                        x_model_input = [np.linspace(min(s1_data), max(s1_data), 100), np.full(100, np.mean(s2_data))]
                        label_x = f"{col_s1_name} ({unidad_s})"
                        subtitle = f"(A {col_s2_name} cte = {np.mean(s2_data):.2f})"
                    else:
                        x_plot = s2_data
                        x_model_input = [np.full(100, np.mean(s1_data)), np.linspace(min(s2_data), max(s2_data), 100)]
                        label_x = f"{col_s2_name} ({unidad_s})"
                        subtitle = f"(A {col_s1_name} cte = {np.mean(s1_data):.2f})"

                    ax.scatter(x_plot, y_data, color='blue', label='Experimental')
                    y_smooth = funcion_modelo(x_model_input, *popt)
                    ax.plot(x_model_input[0] if axis_choice==col_s1_name else x_model_input[1], y_smooth, color='red', label='Modelo')
                    ax.set_title(subtitle, fontsize=9)

                ax.set_xlabel(label_x)
                ax.set_ylabel(label_y)
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig)
                
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=300)
                st.download_button("📷 Descargar Gráfica", buf.getvalue(), "grafica.png", "image/png")

        except Exception as e:
            st.error(f"Error en el cálculo: {e}")
            st.warning("Verifica tus datos o intenta usar valores iniciales manuales.")
