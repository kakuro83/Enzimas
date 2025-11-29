import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from io import BytesIO
import inspect

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
            # Nos aseguramos de que la clase no sea una clase base interna de Python
            if name not in ['Adair', 'Hill']: # Si usas clases base, agrégalas aquí para evitar conflictos
                display_name = name.replace("_", " ").title() + " (Dinámico)"
                models[display_name] = cls
            
    return models

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Ajuste de Cinética Enzimática", layout="centered")
st.title("Ajuste de Modelos Enzimáticos")

if 'resultados' not in st.session_state:
    st.session_state.resultados = None

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
st.info("💡 Tip: Copia tus datos de Excel y pégalos en la primera celda (Ctrl+V).")

data_template = {}
if modalidad == "Un solo sustrato":
    col_s1_name = st.text_input("Nombre de la columna de Sustrato:", value="Sustrato")
    cols = ["Velocidad", col_s1_name]
    data_template = {"Velocidad": [None]*5, col_s1_name: [None]*5}
    col_s2_name = None # Limpiamos la variable para un sustrato
else:
    # Usamos S1 y S2 para ambas variables (ej. Sustrato y Cofactor/Inhibidor)
    c1, c2 = st.columns(2)
    with c1: col_s1_name = st.text_input("Nombre Variable 1 (Sustrato principal):", value="Sustrato 1")
    with c2: col_s2_name = st.text_input("Nombre Variable 2 (Sustrato/Inhibidor/Cofactor):", value="Variable 2")
    cols = ["Velocidad", col_s1_name, col_s2_name]
    data_template = {"Velocidad": [None]*5, col_s1_name: [None]*5, col_s2_name: [None]*5}

df_edited = st.data_editor(pd.DataFrame(data_template), num_rows="dynamic", use_container_width=True)

# Limpieza y preparación de DataFrame
df = df_edited.dropna(how='all').copy()
df = df.dropna(subset=["Velocidad"])
for col in cols:
    if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
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
    v_max_guess = np.max(df["Velocidad"].values) if not df.empty else 1.0

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

# Estética
st.markdown("##### Estética de Gráfica")
c_u1, c_u2 = st.columns(2)
with c_u1: unidad_v = st.text_input("Unidades Velocidad:", value="mM/min")
with c_u2: unidad_s = st.text_input("Unidades Sustrato:", value="mM")

# --- 4. EJECUCIÓN ---
if st.button("Ejecutar ajuste de datos", type="primary"):
    if df.empty or len(df) < 3:
        st.error("Datos insuficientes (mínimo 3 puntos).")
    else:
        try:
            # Preparar datos X, Y
            y_data = df["Velocidad"].values
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
                "s1_col": col_s1_name, "s2_col": col_s2_name
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
        
        # Gráfico solo si es Un solo sustrato
        if modalidad == "Un solo sustrato":
            st.divider()
            st.markdown("### Visualización Gráfica")
            fig, ax = plt.subplots(figsize=(8, 5))
            
            x_vals = res["x_data"]
            ax.scatter(x_vals, res["y_data"], c='blue', label='Experimental', zorder=2, s=50)
            
            x_smooth = np.linspace(min(x_vals), max(x_vals), 100)
            y_smooth = funcion_final(x_smooth, *res["popt"])
            
            ax.plot(x_smooth, y_smooth, c='red', lw=2, label='Modelo', zorder=1)
            ax.set_xlabel(f"{res['s1_col']} ({unidad_s})")
            ax.set_ylabel(f"Velocidad ({unidad_v})")
            ax.legend()
            ax.grid(True, alpha=0.5, ls="--")
            st.pyplot(fig)
            
            img = BytesIO()
            fig.savefig(img, format='png', dpi=300, bbox_inches='tight')
            st.download_button("📷 Descargar Gráfica", img.getvalue(), "plot.png", "image/png")
