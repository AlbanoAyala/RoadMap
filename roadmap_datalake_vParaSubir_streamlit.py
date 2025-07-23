import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import sys
import math
import itertools
import plotly.express as px
import plotly.graph_objects as go
#

# ==============================================================================
# 1. SECCIÓN DE CONFIGURACIÓN
# ==============================================================================
import streamlit as st

SERVER_HOSTNAME = st.secrets["SERVER_HOSTNAME"]
HTTP_PATH = st.secrets["HTTP_PATH"]
ACCESS_TOKEN = st.secrets["ACCESS_TOKEN"]

EXCEL_FILE_PATH = os.path.join(os.path.dirname(__file__), "pozos top10.xlsx")

try:
    df = pd.read_excel(EXCEL_FILE_PATH)
    st.write("¡Archivo Excel cargado exitosamente!")
except FileNotFoundError:
    st.error(f"Error: Archivo Excel no encontrado en la ruta: {EXCEL_FILE_PATH}. Por favor, verifica que el archivo 'pozos_top10.xlsx' está en el mismo directorio que tu app en el repositorio de GitHub.")

EXCEL_WELL_COLUMN_NAME = 'well_name'

CATALOG_NAME = "cgc_data_bronze_prod"
SCHEMA_NAME = "pason"
WELL_INFO_VIEW_NAME = "well_info_view"
TIME_TRACES_VIEW_NAME = "time_traces_20cm_view"
WELL_NAME_COL = "well_name"
WELL_ID_COL = "well_info_id"

TIME_SERIES_COLS_TO_FETCH = ["hole_depth", "wt_on_bit", "rop", "pump_rate", "rotary"] #

DATA_LIMIT_PER_WELL = 100000
PUMP_RATE_CONVERSION_FACTOR = 264.17

# ==============================================================================
# Funciones de Extracción y Preprocesamiento (Optimizadas con Caché de Streamlit)
# ==============================================================================

@st.cache_data
def get_pozos_from_excel(file_path, well_column_name):
    try:
        df_pozos = pd.read_excel(file_path)
        pozos_a_analizar = df_pozos[well_column_name].str.strip().dropna().unique().tolist()
        if not pozos_a_analizar:
            st.error("Error: No se encontraron pozos en el archivo Excel. Asegúrate de que la columna 'well_name' exista y contenga datos.")
            return None
        return pozos_a_analizar
    except FileNotFoundError:
        st.error(f"Error: Archivo Excel no encontrado en la ruta: {file_path}. Por favor, verifica la ruta.")
        return None
    except KeyError:
        st.error(f"Error: La columna '{well_column_name}' no se encontró en el archivo Excel.")
        return None
    except Exception as e:
        st.error(f"Error crítico al leer el Excel: {e}")
        return None

@st.cache_data
def extract_data_from_databricks(pozos_list):
    from databricks import sql
    all_well_data = []
    connection = None
    try:
        st.info("Intentando conectar a Databricks...")
        connection = sql.connect(server_hostname=SERVER_HOSTNAME, http_path=HTTP_PATH, access_token=ACCESS_TOKEN)
        st.success("¡Conexión exitosa a Databricks!")
        with connection.cursor() as cursor:
            progress_text = "Extrayendo datos de pozos. Por favor, espere."
            my_bar = st.progress(0, text=progress_text)
            
            for i, pozo_name in enumerate(pozos_list):
                my_bar.progress((i + 1) / len(pozos_list), text=f"Extrayendo datos para: {pozo_name}")

                query_id = f"""
                SELECT {WELL_ID_COL} FROM {CATALOG_NAME}.{SCHEMA_NAME}.{WELL_INFO_VIEW_NAME}
                WHERE {WELL_NAME_COL} = '{pozo_name}' LIMIT 1
                """ 
                cursor.execute(query_id) 
                result = cursor.fetchone() 

                if not result:
                    st.warning(f"ADVERTENCIA: No se encontró el ID para el pozo '{pozo_name}'. Saltando...") 
                    continue
                well_id = result[0]
                # st.write(f"ID del pozo '{pozo_name}' es {well_id}.") 

                cols_to_select_str = ", ".join(TIME_SERIES_COLS_TO_FETCH)
                query_data = f"""
                SELECT {cols_to_select_str} FROM {CATALOG_NAME}.{SCHEMA_NAME}.{TIME_TRACES_VIEW_NAME}
                WHERE {WELL_ID_COL} = {well_id}
                ORDER BY hole_depth
                LIMIT {DATA_LIMIT_PER_WELL}
                """ 
                # st.write("Ejecutando consulta de datos de series de tiempo...") 
                cursor.execute(query_data) 
                rows = cursor.fetchall() 

                if not rows:
                    st.warning(f"ADVERTENCIA: No se encontraron datos de series de tiempo para el pozo '{pozo_name}'.")
                    continue

                df_pozo = pd.DataFrame(rows, columns=TIME_SERIES_COLS_TO_FETCH) 
                df_pozo['Pozo'] = pozo_name
                all_well_data.append(df_pozo)
                # st.write(f"Se extrajeron {len(df_pozo)} filas para el pozo '{pozo_name}'.") 
        my_bar.empty()
        return all_well_data

    except Exception as e:
        st.error(f"Ocurrió un error en la extracción de datos de Databricks: {e}") 
        return None
    finally:
        if connection:
            connection.close()

@st.cache_data
def preprocess_data(all_well_data_raw):
    if not all_well_data_raw:
        st.error("No se extrajo ningún dato de los pozos para procesar. Asegúrate de que el Excel contenga nombres de pozos válidos y que Databricks tenga datos para ellos.")
        return None

    combined_df = pd.concat(all_well_data_raw, ignore_index=True)

    if 'pump_rate' in combined_df.columns:
        combined_df['pump_rate'] = combined_df['pump_rate'] * PUMP_RATE_CONVERSION_FACTOR 
    else:
        st.warning("La columna 'pump_rate' no se encontró para la conversión a GPM.")

    combined_df.rename(columns={
        'wt_on_bit': 'WOB',
        'rotary': 'RPM',
        'pump_rate': 'Caudal', 
        'hole_depth': 'Profundidad_MD',
        'rop': 'ROP_Promedio',
    }, inplace=True)


    df_cleaned = combined_df.dropna(subset=['Profundidad_MD', 'WOB', 'Caudal', 'RPM', 'ROP_Promedio']).copy()

    # Filtrado de outliers
    df_no_outliers = df_cleaned[
        (df_cleaned['ROP_Promedio'] > 0.1) &
        (df_cleaned['WOB'] > 1) & (df_cleaned['WOB'] < 150) &
        (df_cleaned['RPM'] > 1) & (df_cleaned['RPM'] < 300) &
        (df_cleaned['Caudal'] > 100) & (df_cleaned['Caudal'] < 1500)
    ].copy() 

    window_size = 10
    cols_to_smooth = ['ROP_Promedio', 'WOB', 'Caudal', 'RPM', 'Profundidad_MD']
    for col in cols_to_smooth:
        if col in df_no_outliers.columns:
            df_no_outliers[f'{col}_MA'] = df_no_outliers[col].rolling(window=window_size, min_periods=1).mean()
        else:
            st.warning(f"Advertencia: La columna '{col}' no se encontró en el DataFrame para aplicar media móvil. Verifica tus datos de Databricks.") 

    df_final_clean = df_no_outliers.dropna().copy() 

    if df_final_clean.empty:
        st.error("No quedaron datos útiles después del preprocesamiento. Ajusta los filtros o verifica la calidad de los datos originales.")
        return None
    return df_final_clean

@st.cache_resource
def train_rop_model(df_final_clean):
    numeric_features = ['Profundidad_MD_MA', 'WOB_MA', 'Caudal_MA', 'RPM_MA']
    categorical_features = [] #

    all_required_features = numeric_features + categorical_features
    for f in all_required_features:
        if f not in df_final_clean.columns:
            st.error(f"Error: La característica '{f}' necesaria para el modelo no está en los datos procesados. Revisa tu configuración de `TIME_SERIES_COLS_TO_FETCH` y el preprocesamiento.")
            return None, None, None

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features), 
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    rop_model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    target_variable = 'ROP_Promedio_MA'
    X = df_final_clean[all_required_features]
    y = df_final_clean[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.info("Entrenando modelo de predicción de ROP... Esto puede tomar unos segundos.")
    rop_model_pipeline.fit(X_train, y_train)

    rop_r2 = rop_model_pipeline.score(X_test, y_test)
    st.write(f"R² del modelo de ROP en el conjunto de prueba: **{rop_r2:.4f}**")
    if rop_r2 < 0.5:
        st.warning("ADVERTENCIA: El rendimiento del modelo de ROP es bajo. Considere usar más datos, ajustar hiperparámetros o revisar la calidad de los datos.") 
    elif rop_r2 < 0.7:
        st.info("El modelo tiene un rendimiento aceptable, pero hay margen de mejora.")
    else:
        st.success("El modelo de ROP tiene un buen rendimiento.")

    return rop_model_pipeline, all_required_features, target_variable

# ==============================================================================
# Algoritmo de Optimización (Búsqueda de la Mejor Combinación)
# ==============================================================================

def optimizar_parametros_perforacion(
    profundidad_actual,
    formacion_actual,
    tipo_broca_actual,
    rop_predictor_model,
    min_wob, max_wob, step_wob,
    min_rpm, max_rpm, step_rpm,
    min_flow, max_flow, step_flow,
    features_expected_by_model
):
    best_rop = -np.inf
    best_params = {}

    wob_range = np.arange(min_wob, max_wob + step_wob, step_wob) 
    rpm_range = np.arange(min_rpm, max_rpm + step_rpm, step_rpm) 
    flow_range = np.arange(min_flow, max_flow + step_flow, step_flow) 
    
    # Asegurar que los rangos no estén vacíos si min/max son iguales y step > 0
    if len(wob_range) == 0: wob_range = [min_wob]
    if len(rpm_range) == 0: rpm_range = [min_rpm]
    if len(flow_range) == 0: flow_range = [min_flow]

    for wob, rpm, flow in itertools.product(wob_range, rpm_range, flow_range): 
        input_data_dict = {}
        for feature in features_expected_by_model:
            if feature == 'Profundidad_MD_MA':
                input_data_dict[feature] = profundidad_actual
            elif feature == 'WOB_MA':
                input_data_dict[feature] = wob 
            elif feature == 'Caudal_MA':
                input_data_dict[feature] = flow
            elif feature == 'RPM_MA':
                input_data_dict[feature] = rpm
            elif feature == 'Formacion':
                input_data_dict[feature] = formacion_actual if formacion_actual is not None else 'Desconocida' 
            elif feature == 'Tipo_Broca':
                input_data_dict[feature] = tipo_broca_actual if tipo_broca_actual is not None else 'Desconocida'
            else:
                input_data_dict[feature] = 0
        
        input_df = pd.DataFrame([input_data_dict]) 
        
        predicted_rop = rop_predictor_model.predict(input_df)[0]
        adjusted_rop = predicted_rop

        if adjusted_rop > best_rop:
            best_rop = adjusted_rop
            best_params = {'WOB': wob, 'RPM': rpm, 'Caudal': flow}

    return best_params, best_rop

# ==============================================================================
# Interfaz de Usuario con Streamlit
# ==============================================================================

st.set_page_config(layout="wide", page_title="Dashboard de Optimización de Perforación")

# --- Tema Oscuro y estilos personalizados ---
st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stApp {
        background-color: #0E1117;
    }
    .stSidebar {
        background-color: #1A1A1A;
        color: red;
    }
    .stSlider .st-fx {
        background-color: #333333; 
    }
    .stSlider .st-gn {
        background-color: #1E90FF;
    }
    /* Estilo para el valor numérico del slider (el número que se mueve) */
    .stSlider > label + div > div > div > div > div {
        color: red; /* Color rojo para el valor del slider */
    }
    /* Estilo para los valores mínimo y máximo del slider (los números estáticos en los extremos) */
    .stSlider > label + div > div > div > div:nth-child(1) > div:nth-child(1) {
        color: red;
    }
    .stSlider > label + div > div > div > div:nth-child(3) > div:nth-child(1) {
        color: red;
    }


    .stButton > button {
        background-color: #1E90FF;
        color: blue;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
    }
    .stButton > button:hover {
        background-color: #145DA0;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
        color: #FAFAFA;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] [data-testid="stMarkdownContainer"] p {
        color: #1E90FF;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #1E90FF;
    }
    .stAlert {
        background-color: #282828;
        color: #FAFAFA;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Dashboard de Optimización de Perforación")
st.markdown("Genera perfiles de parámetros óptimos para la perforación basándose en datos históricos reales.")

# --- Barra Lateral para Configuración de Entradas ---
st.sidebar.header("⚙️ Configuración del Análisis")

pozos_a_analizar = get_pozos_from_excel(EXCEL_FILE_PATH, EXCEL_WELL_COLUMN_NAME)

if pozos_a_analizar is None:
    st.sidebar.error("❌ No se pudieron cargar los pozos del Excel. Por favor, verifica la ruta y el contenido del archivo.")
    st.stop()
else:
    st.sidebar.success(f"✔️ **{len(pozos_a_analizar)}** pozos listos para analizar desde Excel.")

st.sidebar.subheader("Ventana de Rangos Operativos")

# Utilizando la información de rangos operativos guardada
min_wob_ui = st.sidebar.slider("WOB Mín. (klbs)", min_value=1, max_value=100, value=st.session_state.get('min_wob', 5), step=1) 
max_wob_ui = st.sidebar.slider("WOB Máx. (klbs)", min_value=1, max_value=100, value=st.session_state.get('max_wob', 50), step=1) 
step_wob_ui = st.sidebar.slider("Paso WOB (klbs)", min_value=1, max_value=20, value=st.session_state.get('step_wob', 5), step=1) 

min_rpm_ui = st.sidebar.slider("RPM Mín. (rpm)", min_value=10, max_value=300, value=st.session_state.get('min_rpm', 50), step=5) 
max_rpm_ui = st.sidebar.slider("RPM Máx. (rpm)", min_value=10, max_value=300, value=st.session_state.get('max_rpm', 110), step=5) 
step_rpm_ui = st.sidebar.slider("Paso RPM (rpm)", min_value=1, max_value=50, value=st.session_state.get('step_rpm', 10), step=5) 

min_flow_ui = st.sidebar.slider("Caudal Mín. (GPM)", min_value=100, max_value=1500, value=st.session_state.get('min_flow', 400), step=10) 
max_flow_ui = st.sidebar.slider("Caudal Máx. (GPM)", min_value=100, max_value=1500, value=st.session_state.get('max_flow', 800), step=10) 
step_flow_ui = st.sidebar.slider("Paso Caudal (GPM)", min_value=10, max_value=100, value=st.session_state.get('step_flow', 50), step=10) 


st.sidebar.subheader("Parámetros del Perfil de Profundidad")
td_objetivo_ui = st.sidebar.slider("Profundidad Total Objetivo (MD) (m)", min_value=50, max_value=3000, value=2400, step=100)
step_profundidad_ui = st.sidebar.slider("Intervalo de Profundidad para el Perfil (m)", min_value=10, max_value=200, value=100, step=10)

formacion_actual_ui = st.sidebar.selectbox("Formación Asumida (para el perfil)", ["Caliza Media", "Arenisca Fuerte", "Esquisto Blando", "Otra"], index=0) 
tipo_broca_actual_ui = st.sidebar.selectbox("Tipo de Broca Asumida (para el perfil)", ["PDC_5 Blades", "RollerCone_TCI", "PDC_7 Blades", "Otra"], index=0) 

if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False

if st.sidebar.button("🚀 Generar y Visualizar Perfil", type="primary"):
    st.session_state.run_analysis = True

# ==============================================================================
# Lógica de Ejecución del Algoritmo y Visualización de Resultados
# ==============================================================================

if st.session_state.run_analysis:
    with st.spinner("Cargando datos históricos, preprocesando y entrenando el modelo de ROP..."):
        all_well_data_extracted = extract_data_from_databricks(pozos_a_analizar)
        if all_well_data_extracted is None:
            st.stop()

        df_final_clean_processed = preprocess_data(all_well_data_extracted)
        if df_final_clean_processed is None:
            st.stop()

        rop_model_trained, features_for_model_used, target_variable_used = train_rop_model(df_final_clean_processed)
        if rop_model_trained is None:
            st.stop()

    st.success("¡Datos cargados, procesados y modelo entrenado con éxito!")

    st.subheader("Calculando Perfil Óptimo...")
    progress_bar_depth = st.progress(0, text="Calculando para Profundidad: 0 m")
    status_text_depth = st.empty()

    perfil_optimo_df = pd.DataFrame(columns=['Profundidad', 'WOB_Recomendado', 'RPM_Recomendado', 'Caudal_Recomendado', 'ROP_Predicha']) 

    # Ensure depth points are within the target and correctly count total steps
    depth_points = np.arange(0, td_objetivo_ui + 1, step_profundidad_ui)
    total_depth_steps = len(depth_points)

    for i, current_depth in enumerate(depth_points): 
        progress_percentage = (i + 1) / total_depth_steps
        progress_bar_depth.progress(progress_percentage, text=f"Calculando para Profundidad: {current_depth:.0f} m")
        status_text_depth.text(f"Profundidad actual: {current_depth:.0f} m / {td_objetivo_ui:.0f} m")

        opt_params, predicted_max_rop = optimizar_parametros_perforacion(
            profundidad_actual=current_depth,
            formacion_actual=formacion_actual_ui,
            tipo_broca_actual=tipo_broca_actual_ui,
            rop_predictor_model=rop_model_trained,
            min_wob=min_wob_ui, max_wob=max_wob_ui, step_wob=step_wob_ui,
            min_rpm=min_rpm_ui, max_rpm=max_rpm_ui, step_rpm=step_rpm_ui,
            min_flow=min_flow_ui, max_flow=max_flow_ui, step_flow=step_flow_ui,
            features_expected_by_model=features_for_model_used 
        )
        
        perfil_optimo_df.loc[len(perfil_optimo_df)] = {
            'Profundidad': current_depth, 
            'WOB_Recomendado': opt_params['WOB'], 
            'RPM_Recomendado': opt_params['RPM'], 
            'Caudal_Recomendado': opt_params['Caudal'], 
            'ROP_Predicha': predicted_max_rop 
        }
    
    progress_bar_depth.empty()
    status_text_depth.empty()
    st.success("¡Perfil de perforación óptimo generado!") 

    st.session_state.perfil_optimo_df = perfil_optimo_df
    st.session_state.df_final_clean = df_final_clean_processed
    st.session_state.all_extracted_wells = pozos_a_analizar 

if 'perfil_optimo_df' in st.session_state and not st.session_state.perfil_optimo_df.empty:
    tab1, tab2 = st.tabs(["📊 Perfil de Parámetros Óptimos", "📋 Tabla de Resultados"])

    with tab1:
        st.header("Perfiles de Parámetros Óptimos por Profundidad")
        st.write("Estos gráficos muestran las combinaciones óptimas de parámetros de perforación (WOB, RPM, Caudal) y la ROP predicha para maximizar la eficiencia a diferentes profundidades.")
        
        line_colors = {
            'ROP_Predicha': '#32CD32',
            'WOB_Recomendado': '#CCCC00',
            'RPM_Recomendado': '#FF6347',
            'Caudal_Recomendado': '#1E90FF'
        }

        fig_rop = go.Figure()
        fig_rop.add_trace(go.Scatter(
            x=st.session_state.perfil_optimo_df['Profundidad'], # Eje X (Profundidad)
            y=st.session_state.perfil_optimo_df['ROP_Predicha'],  # Eje Y (ROP)
            mode='lines+markers',
            name='ROP Predicha',
            line=dict(color=line_colors['ROP_Predicha'], width=2)
        ))
        # Ajuste de rangos para asegurar la visibilidad del rango completo
        fig_rop.add_trace(go.Scatter(
            x=st.session_state.perfil_optimo_df['Profundidad'],
            y=st.session_state.perfil_optimo_df['ROP_Predicha'] * 1.1,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig_rop.add_trace(go.Scatter(
            x=st.session_state.perfil_optimo_df['Profundidad'],
            y=st.session_state.perfil_optimo_df['ROP_Predicha'] * 0.9,
            mode='lines',
            line=dict(width=0),
            fill='tonexty', # Cambiado a tonexty para llenado vertical
            fillcolor='rgba(50,205,50,0.2)',
            name='Rango de ROP (simulado)',
            hoverinfo='skip'
        ))

        fig_rop.update_layout(
            title_text='ROP Predicha Óptima vs. Profundidad',
            xaxis_title="Profundidad (m)", # Eje X
            yaxis_title="ROP Óptima (m/hr o ft/hr)", # Eje Y
            plot_bgcolor='#0E1117',
            paper_bgcolor='#0E1117',
            font_color='#FAFAFA',
            xaxis=dict(showgrid=True, gridcolor='#333333', range=[0, td_objetivo_ui]), # Rango y grid para X
            yaxis=dict(showgrid=True, gridcolor='#333333'),
            hovermode='x unified' # Cambiado a x unified
        )
        st.plotly_chart(fig_rop, use_container_width=True)

        col1_charts, col2_charts = st.columns(2)

        with col1_charts:
            fig_wob = go.Figure()
            fig_wob.add_trace(go.Scatter(
                x=st.session_state.perfil_optimo_df['Profundidad'], # Eje X (Profundidad)
                y=st.session_state.perfil_optimo_df['WOB_Recomendado'], # Eje Y (WOB)
                mode='lines+markers',
                name='WOB Recomendado',
                line=dict(color=line_colors['WOB_Recomendado'], width=2)
            ))
            fig_wob.update_layout(
                title_text="WOB Óptimo",
                xaxis_title="Profundidad (m)", # Eje X
                yaxis_title="WOB (klbs)", # Eje Y
                plot_bgcolor='#0E1117',
                paper_bgcolor='#0E1117',
                font_color='#FAFAFA',
                xaxis=dict(showgrid=True, gridcolor='#333333', range=[0, td_objetivo_ui]), # Rango y grid para X
                yaxis=dict(showgrid=True, gridcolor='#333333'),
                hovermode='x unified'
            )
            st.plotly_chart(fig_wob, use_container_width=True)

            fig_rpm = go.Figure()
            fig_rpm.add_trace(go.Scatter(
                x=st.session_state.perfil_optimo_df['Profundidad'], # Eje X (Profundidad)
                y=st.session_state.perfil_optimo_df['RPM_Recomendado'], # Eje Y (RPM)
                mode='lines+markers',
                name='RPM Recomendado',
                line=dict(color=line_colors['RPM_Recomendado'], width=2)
            ))
            fig_rpm.update_layout(
                title_text="RPM Óptimo",
                xaxis_title="Profundidad (m)", # Eje X
                yaxis_title="RPM (rpm)", # Eje Y
                plot_bgcolor='#0E1117',
                paper_bgcolor='#0E1117',
                font_color='#FAFAFA',
                xaxis=dict(showgrid=True, gridcolor='#333333', range=[0, td_objetivo_ui]), # Rango y grid para X
                yaxis=dict(showgrid=True, gridcolor='#333333'),
                hovermode='x unified'
            )
            st.plotly_chart(fig_rpm, use_container_width=True)

        with col2_charts:
            fig_flow = go.Figure()
            fig_flow.add_trace(go.Scatter(
                x=st.session_state.perfil_optimo_df['Profundidad'], # Eje X (Profundidad)
                y=st.session_state.perfil_optimo_df['Caudal_Recomendado'], # Eje Y (Caudal)
                mode='lines+markers',
                name='Caudal Recomendado',
                line=dict(color=line_colors['Caudal_Recomendado'], width=2)
            ))
            fig_flow.update_layout(
                title_text="Caudal Óptimo",
                xaxis_title="Profundidad (m)", # Eje X
                yaxis_title="Caudal (GPM)", # Eje Y
                plot_bgcolor='#0E1117',
                paper_bgcolor='#0E1117',
                font_color='#FAFAFA',
                xaxis=dict(showgrid=True, gridcolor='#333333', range=[0, td_objetivo_ui]), # Rango y grid para X
                yaxis=dict(showgrid=True, gridcolor='#333333'),
                hovermode='x unified'
            )
            st.plotly_chart(fig_flow, use_container_width=True)

            st.markdown("<br><br><br>", unsafe_allow_html=True)

    with tab2:
        st.header("Tabla de Perfil de Parámetros Óptimos")
        st.dataframe(st.session_state.perfil_optimo_df.round(2), use_container_width=True)

        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv_data = convert_df_to_csv(st.session_state.perfil_optimo_df)
        st.download_button(
            label="Descargar perfil óptimo como CSV",
            data=csv_data,
            file_name="perfil_optimo_perforacion.csv",
            mime="text/csv",
        )

else:
    st.info("Ajusta los parámetros deseados en la barra lateral izquierda y haz clic en '🚀 Generar y Visualizar Perfil' para iniciar el análisis y ver los resultados.")

st.markdown("---")
st.caption("Aplicación de Optimización de Perforación v1.0. Desarrollada con Streamlit.")