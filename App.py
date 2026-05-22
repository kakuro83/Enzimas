import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from io import BytesIO
import inspect
import plotly.graph_objects as go # Importamos Plotly para gráficos 3D interactivos

# --- IMPORTACIÓN DE MÓDULOS ---
try:
    import modelos.un_sustrato as mod_un_sustrato
    import modelos.dos_sustratos as mod_dos_sustratos
except ImportError as e:
    st.error(f"Error importando módulos: {e}. Asegúrate de que 'modelos/un_sustrato.py' y 'modelos/dos_sustratos.py' existan.")
    st.stop()

def get_models_from_module(module):
    """Obtiene funciones y Clases dinámicas del módulo."""
    models = {}
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if func.__module__ == module.__name__:
            display_name = name.replace("_", " ").title()
            models[display_name] = func
    for name, cls in inspect.getmembers(module, inspect.isclass):
        if cls.__module__ == module.__name__:
            display_name = name.replace("_", " ").title() + " (Dinámico)"
            models[display_name] = cls
    return models

# Función para generar el DataFrame inicial vacío con TIPO FLOAT explícito
def get_empty_data_df(col_v_name, col_s1_name, col_s2_name=None, num_rows=5):
    # Usamos np.nan en lugar de None para que Pandas reconozca la columna como numérica (float)
    data = {col_v_name: [np.nan]*num_rows, col_s1_name: [np.nan]*num_rows}
    if col_s2_name:
        data[col_s2_name] = [np.nan]*num_rows
    return pd.DataFrame(data).astype(float)

def coerce_numeric_cols(df_in, columns):
    """Convierte columnas objetivo a numérico, enviando valores inválidos a NaN."""
    df_out = df_in.copy()
    for col in columns:
        if col in df_out.columns:
            series = df_out[col]
            # Soporte para pegado con coma decimal (p.ej. 1,23)
            if not pd.api.types.is_numeric_dtype(series):
                series = series.astype("string").str.strip().str.replace(",", ".", regex=False)
            df_out[col] = pd.to_numeric(series, errors='coerce')
    return df_out

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
    st.session_state.col_names_last = {} 
# Inicializar llave dinámica para el editor
if 'editor_key' not in st.session_state:
    st.session_state.editor_key = 0

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
st.info("💡 Tip: Copia tus datos de Excel y pégalos en la primera celda (Ctrl+V). Los campos vacíos serán ignorados.")

# --- Define columns based on modality ---
c_names = st.columns(3) 
with c_names[0]:
    col_v_name = st.text_input("Etiqueta Velocidad:", value="Velocidad") 
if modalidad == "Un solo sustrato":
    with c_names[1]:
        col_s1_name = st.text_input("Etiqueta Sustrato:", value="Sustrato") 
    cols = [col_v_name, col_s1_name]
    col_s2_name = None 
else:
    with c_names[1]:
        col_s1_name = st.text_input("Etiqueta Sustrato principal:", value="Sustrato 1") 
    with c_names[2]:
        col_s2_name = st.text_input("Etiqueta Sustrato/Inhibidor/Cofactor:", value="Variable 2") 
    cols = [col_v_name, col_s1_name, col_s2_name]

if modalidad == "Un solo sustrato":
    order_options = list(permutations(cols, len(cols)))
    default_order_idx = order_options.index(tuple(cols))
    selected_order = st.selectbox(
        "Orden de columnas para pegado:",
        options=order_options,
        index=default_order_idx,
        format_func=lambda p: " | ".join(p),
        help="Selecciona el orden en que están tus columnas nativas y luego pega toda la tabla de una vez."
    )
    display_cols = list(selected_order)
else:
    st.markdown("**Orden de columnas para pegado (mueve cada etiqueta):**")
    order_key = "display_order_dos"
    if (
        order_key not in st.session_state
        or len(st.session_state[order_key]) != len(cols)
        or set(st.session_state[order_key]) != set(cols)
    ):
        st.session_state[order_key] = cols.copy()

    move_action = None
    for idx, label in enumerate(st.session_state[order_key]):
        c_lbl, c_left, c_right = st.columns([6, 1, 1])
        with c_lbl:
            st.write(f"{idx + 1}. **{label}**")
        with c_left:
            if st.button("←", key=f"move_left_{idx}_{modalidad}", use_container_width=True, disabled=(idx == 0)):
                move_action = ("left", idx)
        with c_right:
            if st.button("→", key=f"move_right_{idx}_{modalidad}", use_container_width=True, disabled=(idx == len(cols)-1)):
                move_action = ("right", idx)

    if move_action:
        direction, idx = move_action
        order = st.session_state[order_key].copy()
        if direction == "left" and idx > 0:
            order[idx - 1], order[idx] = order[idx], order[idx - 1]
        elif direction == "right" and idx < len(order) - 1:
            order[idx + 1], order[idx] = order[idx], order[idx + 1]
        st.session_state[order_key] = order
        st.rerun()

    display_cols = st.session_state[order_key]

st.caption("Una vez definido el orden, pega todas las columnas en un solo paso desde la primera celda.")

# Generar el DataFrame de plantilla
data_template_df = get_empty_data_df(col_v_name, col_s1_name, col_s2_name)

# --- Column Configuration ---
col_config = {
    col_v_name: st.column_config.NumberColumn(col_v_name, format="%.4f"),
    col_s1_name: st.column_config.NumberColumn(col_s1_name, format="%.4f")
}
if col_s2_name:
    col_config[col_s2_name] = st.column_config.NumberColumn(col_s2_name, format="%.4f")

# --- Session State Management ---
is_modal_change = (st.session_state.modalidad_last != modalidad)
old_col_names_map = st.session_state.col_names_last.get(st.session_state.modalidad_last, {})

if not st.session_state.experimental_data.empty:
    session_data = st.session_state.experimental_data.copy()
    rename_mapping = {}
    
    old_v_name = old_col_names_map.get('v_col')
    old_s1_name = old_col_names_map.get('s1_col')
    old_s2_name = old_col_names_map.get('s2_col')
    
    if old_v_name and old_v_name in session_data.columns and old_v_name != col_v_name:
        rename_mapping[old_v_name] = col_v_name
    if old_s1_name and old_s1_name in session_data.columns and old_s1_name != col_s1_name:
        rename_mapping[old_s1_name] = col_s1_name
    if col_s2_name and old_s2_name and old_s2_name in session_data.columns and old_s2_name != col_s2_name:
        rename_mapping[old_s2_name] = col_s2_name

    if rename_mapping:
        session_data.rename(columns=rename_mapping, inplace=True)
    
    # Reindexar para asegurar que tenemos las columnas correctas
    session_data = session_data.reindex(columns=cols)

    # ⚠️ CRUCIAL: Forzar conversión a numérico para evitar conflicto con NumberColumn
    session_data = coerce_numeric_cols(session_data, cols)
    
    if is_modal_change or len(session_data.columns) != len(cols):
        st.session_state.experimental_data = data_template_df
    else:
        st.session_state.experimental_data = session_data 
else:
    st.session_state.experimental_data = data_template_df

col_names_to_save = {'v_col': col_v_name, 's1_col': col_s1_name}
if col_s2_name: col_names_to_save['s2_col'] = col_s2_name
st.session_state.col_names_last[modalidad] = col_names_to_save
st.session_state.modalidad_last = modalidad

st.markdown("### Pegar datos desde celular")

texto_pegado = st.text_area(
    "Pega aquí los datos copiados desde Excel, Sheets o una tabla",
    height=150,
    placeholder="Ejemplo:\n0.25\t1.0\n0.48\t2.0\n0.70\t4.0"
)

if st.button("Cargar datos pegados"):
    try:
        from io import StringIO

        texto_limpio = texto_pegado.strip().replace(",", ".")

        df_paste = pd.read_csv(
            StringIO(texto_limpio),
            sep=r"\s+|\t|;",
            engine="python",
            header=None
        )

        df_paste = df_paste.iloc[:, :len(display_cols)]
        df_paste.columns = display_cols[:df_paste.shape[1]]
        df_paste = df_paste.reindex(columns=cols)
        df_paste = coerce_numeric_cols(df_paste, cols)

        st.session_state.experimental_data = df_paste
        st.session_state.editor_key += 1
        st.success("Datos cargados correctamente.")
        st.rerun()

    except Exception as e:
        st.error(f"No se pudieron cargar los datos pegados: {e}")
        
c_editor, c_button = st.columns([5, 1])
with c_button:
    if st.button("Limpiar Datos", key="clear_data_btn", use_container_width=True):
        st.session_state.experimental_data = data_template_df
        st.session_state.resultados = None 
        st.session_state.editor_key += 1 # Incrementar llave para forzar reinicio del widget
        st.rerun()

with c_editor:
    # ⚠️ Aseguramos que los datos que entran al editor sean numéricos puros
    df_to_edit = st.session_state.experimental_data.copy()
    df_to_edit = coerce_numeric_cols(
        df_to_edit.reindex(columns=display_cols).reset_index(drop=True),
        cols
    )

    df_edited = st.data_editor(
        df_to_edit,
        num_rows="dynamic",
        use_container_width=True,
        column_config=col_config,
        hide_index=True,
        key=f"data_editor_{st.session_state.editor_key}" # Llave dinámica
    )

df_edited = df_edited.reindex(columns=cols).reset_index(drop=True)
df_edited_raw = df_edited.replace(r'^\s*$', np.nan, regex=True)
df_edited_num = coerce_numeric_cols(df_edited_raw, cols)
df_base = coerce_numeric_cols(
    st.session_state.experimental_data.reindex(columns=cols).reset_index(drop=True),
    cols
)
max_rows = max(len(df_edited_num), len(df_base))
df_edited_raw = df_edited_raw.reindex(range(max_rows))
df_edited_num = df_edited_num.reindex(range(max_rows))
df_base = df_base.reindex(range(max_rows))
df_updated = df_base.copy()
for col in cols:
    if col in df_updated.columns:
        mask = df_edited_raw[col].notna()
        df_updated.loc[mask, col] = df_edited_num.loc[mask, col]
st.session_state.experimental_data = df_updated

# Limpieza final
df = st.session_state.experimental_data.copy()
df = df.dropna(how='all').copy()
df = df.dropna(subset=[col_v_name]) 
df = coerce_numeric_cols(df, cols)
df = df.dropna()

def validar_datos(df_validar, modalidad_actual, col_v, col_s1, col_s2=None):
    if df_validar.empty or len(df_validar) < 3:
        return "Datos insuficientes (mínimo 3 puntos)."
    if not np.isfinite(df_validar[[col_v, col_s1]].to_numpy()).all():
        return "Los datos contienen valores no finitos (NaN/Inf)."
    if modalidad_actual == "Un solo sustrato":
        if df_validar[col_s1].nunique() < 2:
            return "El sustrato principal requiere al menos 2 valores distintos."
    else:
        if col_s2 is None:
            return "Falta la columna de la segunda variable."
        if not np.isfinite(df_validar[[col_s2]].to_numpy()).all():
            return "La segunda variable contiene valores no finitos (NaN/Inf)."
        if df_validar[col_s1].nunique() < 2 or df_validar[col_s2].nunique() < 2:
            return "Cada variable requiere al menos 2 valores distintos."
    return None

# --- 3. SELECCIÓN DE MODELO ---
st.divider()
st.subheader("Configuración del Ajuste")

if modalidad == "Un solo sustrato": model_source = mod_un_sustrato
else: model_source = mod_dos_sustratos 

model_options = get_models_from_module(model_source)
nombre_modelo_sel = st.selectbox("Seleccione el modelo cinético:", list(model_options.keys()))
objeto_modelo = model_options[nombre_modelo_sel]

funcion_final = None
if inspect.isclass(objeto_modelo):
    st.info(f"Este es un modelo de orden variable. Selecciona el número de términos.")
    orden_n = st.number_input("Orden del Modelo (n):", min_value=1, max_value=10, value=2, step=1)
    instancia = objeto_modelo(orden_n) 
    try:
        funcion_final = instancia.obtener_funcion()
    except AttributeError:
        st.error("Error: La clase dinámica debe tener el método 'obtener_funcion()'.")
        st.stop()
else:
    funcion_final = objeto_modelo

doc_ecuacion = inspect.getdoc(funcion_final)
if doc_ecuacion:
    st.latex(doc_ecuacion.replace("$", "").strip())
else:
    st.caption("Ecuación no disponible o generada dinámicamente.")

try:
    sig = inspect.signature(funcion_final)
    param_names = list(sig.parameters.keys())[1:] 
except ValueError:
    st.error("Error al obtener parámetros.")
    st.stop()

# --- CONFIGURACIÓN DE PARÁMETROS ---
with st.expander("🛠️ Opciones Avanzadas: Valores Iniciales y Parámetros Fijos"):
    st.caption("Si el ajuste falla, intenta cambiar manualmente estos valores iniciales.")
    param_settings = {}
    v_max_guess = np.max(df[col_v_name].values) if not df.empty else 1.0 

    for p in param_names:
        c_lbl, c_val, c_fix = st.columns([1, 2, 1])
        default_val = 1.0
        if "Vmax" in p or "a" == p: default_val = float(v_max_guess)
        elif "n" == p or "beta" in p: default_val = 1.0 
        elif not df.empty and ("Km" in p or "K_" in p or "K" in p): default_val = float(np.mean(df.iloc[:, 1]))
        elif "a_" in p or "b_" in p: default_val = 0.1
        
        with c_lbl: st.markdown(f"**{p}**")
        with c_val: val = st.number_input(f"Valor", value=default_val, label_visibility="collapsed", key=f"v_{p}_{nombre_modelo_sel}")
        with c_fix: fixed = st.checkbox("Fijar", key=f"f_{p}_{nombre_modelo_sel}")
        param_settings[p] = {"value": val, "fixed": fixed}

# --- 4. EJECUCIÓN ---
if st.button("Ejecutar ajuste de datos", type="primary"):
    validation_error = validar_datos(df, modalidad, col_v_name, col_s1_name, col_s2_name)
    if validation_error:
        st.error(validation_error)
    else:
        try:
            y_data = df[col_v_name].values 
            if modalidad == "Un solo sustrato": 
                x_data = df[col_s1_name].values
            else: 
                x_data = [df[col_s1_name].values, df[col_s2_name].values]

            p0, fixed_map, free_keys = [], {}, []
            for p in param_names:
                cfg = param_settings[p]
                if cfg["fixed"]: fixed_map[p] = cfg["value"]
                else:
                    free_keys.append(p)
                    p0.append(cfg["value"])

            def model_wrapper(x, *free_args):
                full_args = []
                idx = 0
                for name in param_names:
                    if name in fixed_map: full_args.append(fixed_map[name])
                    else:
                        full_args.append(free_args[idx])
                        idx += 1
                return funcion_final(x, *full_args)

            if not free_keys:
                st.info("Todos los parámetros fijos.")
                popt_full = [param_settings[p]["value"] for p in param_names]
            else:
                try:
                    popt_free, _ = curve_fit(model_wrapper, x_data, y_data, p0=p0, maxfev=500000, bounds=(0, np.inf))
                except RuntimeError as optim_err:
                    st.error(f"⚠️ No se pudo encontrar el ajuste óptimo. (Error: {optim_err})")
                    st.stop()
                    
                popt_full = []
                idx = 0
                for p in param_names:
                    if param_settings[p]["fixed"]: popt_full.append(param_settings[p]["value"])
                    else:
                        popt_full.append(popt_free[idx])
                        idx += 1

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
                "v_col": col_v_name, "s1_col": col_s1_name, "s2_col": col_s2_name
            }
            st.rerun()

        except Exception as e:
            st.error(f"Error crítico en el cálculo: {e}")

# --- 5. RESULTADOS ---
if st.session_state.resultados:
    res = st.session_state.resultados
    
    if res.get("modalidad") != modalidad or res.get("model_name") != nombre_modelo_sel:
        st.warning("⚠️ Configuración cambiada. Ejecuta de nuevo.")
    else:
        st.success("¡Resultados disponibles!")
        df_p = pd.DataFrame({"Parámetro": res["param_names"], "Valor": res["popt"]})
        df_s = pd.DataFrame({
            "Estadístico": ["R²", "RMSE", "MAE", "AIC"],
            "Valor": [res['r2'], res['rmse'], res['mae'], res['aic']]
        })
        help_txt = "R²: Coef. Determinación.\nRMSE: Raíz Error Cuadrático Medio.\nMAE: Error Absoluto Medio.\nAIC: Criterio Akaike."

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
            fig, ax = plt.subplots(figsize=(8, 5))
            x_vals = res["x_data"]
            ax.scatter(x_vals, res["y_data"], c='blue', label='Experimental', zorder=2, s=50)
            x_smooth = np.linspace(min(x_vals), max(x_vals), 100)
            y_smooth = funcion_final(x_smooth, *res["popt"])
            ax.plot(x_smooth, y_smooth, c='red', lw=2, label='Modelo', zorder=1)
            ax.set_xlabel(f"{res['s1_col']}"); ax.set_ylabel(f"{res['v_col']}")
            ax.legend(); ax.grid(True, alpha=0.5, ls="--")
            st.pyplot(fig)
            img = BytesIO(); fig.savefig(img, format='png', dpi=300, bbox_inches='tight')
            st.download_button("📷 Descargar Gráfica", img.getvalue(), "plot.png", "image/png")
        else:
            st.markdown("#### Superficie de Respuesta (3D) - Interactivo")
            s1_exp = res["x_data"][0]; s2_exp = res["x_data"][1]; v_exp = res["y_data"]
            s1_line = np.linspace(s1_exp.min(), s1_exp.max(), 50)
            s2_line = np.linspace(s2_exp.min(), s2_exp.max(), 50)
            S1_MESH, S2_MESH = np.meshgrid(s1_line, s2_line)
            X_MESH = [S1_MESH.ravel(), S2_MESH.ravel()]
            try:
                Z_MESH = funcion_final(X_MESH, *res["popt"]).reshape(S1_MESH.shape)
                fig = go.Figure(data=[
                    go.Surface(z=Z_MESH, x=S1_MESH, y=S2_MESH, colorscale='Viridis', opacity=0.8, showscale=False, name='Modelo'),
                    go.Scatter3d(x=s1_exp, y=s2_exp, z=v_exp, mode='markers', marker=dict(size=5, color='red'), name='Datos')
                ])
                fig.update_layout(scene=dict(xaxis_title=f"{res['s1_col']}", yaxis_title=f"{res['s2_col']}", zaxis_title=f"{res['v_col']}"))
                st.plotly_chart(fig, use_container_width=True) 
                st.info("Usa el mouse para girar el gráfico.")
            except Exception as e:
                st.error(f"Error 3D: {e}")
