import lightgbm
import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import os # Importar os para verificar archivos

# --- Código para verificar las versiones de las librerías ---
st.set_page_config(layout="wide")
st.subheader("📦 Versiones de Librerías")

try:
    lgb_version = lightgbm.__version__
    st.info(f"LightGBM version: {lgb_version}")
except Exception as e:
    st.error(f"Error getting LightGBM version: {e}")

try:
    pandas_version = pd.__version__
    st.info(f"Pandas version: {pandas_version}")
except Exception as e:
    st.error(f"Error getting Pandas version: {e}")

try:
    joblib_version = joblib.__version__
    st.info(f"Joblib version: {joblib.__version__}")
except Exception as e:
    st.error(f"Error getting Joblib version: {e}")
# ------------------------------------------------------------



st.title("🌽 Forage Maize Prediction in NW of Spain")

#---------------------
# Cargar modelos LightGBM (Asegúrate de usar los nombres de archivo .pkl correctos)
st.subheader("⚙️ Carga de Modelos")
try:
    # VERIFICA estos nombres de archivo .pkl si son exactamente los que guardaste
    model_dm = joblib.load("DM_lgb_best_model.pkl")
    model_ufl = joblib.load("UFL_lgb_best_model.pkl") # Asegúrate del nombre correcto (UFL_lgb_best_model.pkl o UFL_lgb_best_model.pkl)
    model_cp = joblib.load("CP_lgb_best_model.pkl")   # Asegúrate del nombre correcto (CP_lgb_best_model.pkl o CP_lgb_best_model.pkl)
    st.success("✅ Modelos cargados correctamente.")
except FileNotFoundError as e:
    st.error(f"❌ Error: Archivo de modelo no encontrado - {e}. Asegúrate de que los archivos .pkl están en la ubicación correcta en Streamlit Cloud.")
    st.stop() # Detiene la ejecución si los modelos no cargan
except Exception as e:
    st.error(f"❌ Error al cargar modelos: {e}")
    st.stop() # Detiene la ejecución si los modelos no cargan


# Cargar mapa
map_file_path = "AsturiasGalicia2.jpg"
if os.path.exists(map_file_path):
    map_image = Image.open(map_file_path)
    map_caption = "Study area: Galicia, Asturias"
    map_found = True
else:
    st.warning(f"🗺️ Archivo de mapa no encontrado ({map_file_path}). La imagen no se mostrará.")
    map_image = None
    map_caption = "Map not available"
    map_found = False


# Cargar datos desde Excel (para listas desplegables)
excel_file_path = "260324_ENG_MaizeForageSpainNWwtYearRadDay.xlsx"
st.subheader("📂 Carga de Datos de Entrada")
try:
    if os.path.exists(excel_file_path):
        df = pd.read_excel(excel_file_path)
        st.success("✅ Datos de Excel cargados correctamente.")
        excel_data_found = True
        # Obtener listas únicas para selectbox
        site_options = sorted(df['Site'].unique().tolist())
        cultivar_options = sorted(df['Cultivar'].unique().tolist())
    else:
        st.error(f"❌ Error: Archivo Excel no encontrado ({excel_file_path}). No se pueden cargar los datos de entrada.")
        excel_data_found = False
        st.stop() # Detiene la ejecución si no se pueden cargar los datos
except Exception as e:
    st.error(f"❌ Error al cargar datos de Excel: {e}")
    excel_data_found = False
    st.stop() # Detiene la ejecución si ocurre otro error


# --- Diseño de la interfaz (Columnas para mapa y inputs) ---
col1, col2 = st.columns([1, 2]) # Columna 1 para inputs, Columna 2 para mapa y outputs

with col2:
    if map_found:
       st.image(map_image, caption=map_caption, use_container_width=True)

    st.subheader("📈 Predicciones")
    # Inicializar session_state para las predicciones si no existen
    if 'pred_dm' not in st.session_state:
        st.session_state['pred_dm'] = '---'
    if 'pred_ufl' not in st.session_state:
        st.session_state['pred_ufl'] = '---'
    if 'pred_cp' not in st.session_state:
        st.session_state['pred_cp'] = '---'

    # Mostrar predicciones usando st.metric (ESTO VA FUERA DEL BOTÓN)
    col_dm, col_ufl, col_cp = st.columns(3)
    col_dm.metric(label="Dry Matter (kg DM/ha)", value=st.session_state.get('pred_dm', '---'))
    col_ufl.metric(label="UFL/ha", value=st.session_state.get('pred_ufl', '---'))
    col_cp.metric(label="Crude Protein (kg CP/ha)", value=st.session_state.get('pred_cp', '---'))


# --- Sidebar o Columna para Inputs ---
# Si prefieres Sidebar:
# st.sidebar.header("Input Parameters")
# with st.sidebar:
#     st.header("Parámetros de Entrada")
#     # ... tus inputs aquí ...

# Usando Columna 1 para inputs (como en tu diseño original)
with col1:
    st.header("Parameters")

    # Usar datos cargados para selectbox
    if excel_data_found:
        input_Site = st.selectbox("Site", site_options)
        input_Cultivar = st.selectbox("Cultivar", cultivar_options)
    else:
        # Opciones por defecto si el Excel no carga (menos útil)
        input_Site = st.selectbox("Site", ["DefaultSite"])
        input_Cultivar = st.selectbox("Cultivar", ["DefaultCultivar"])


    input_Elevation = st.number_input("Elevation(m)", value=25.0, min_value=0.0, format="%.1f")
    input_Radiation = st.number_input("Radiation(Mj/m2day)", value=21.0, min_value=0.0, format="%.1f")
    input_Precipitation = st.number_input("Precipitation(mm)", value=56.6, min_value=0.0, format="%.1f")
    input_Tmax = st.number_input("Tmax(ºC)", value=20.5, format="%.1f")
    input_Tmin = st.number_input("Tmin(ºC)", value=16.7, format="%.1f")
    input_WHC = st.number_input("WHC(mm)", value=90.0, min_value=0.0, format="%.1f")
    input_C = st.number_input("C(%)", value=1.9, min_value=0.0, format="%.2f") # Ajustado a 1.9 como en tu ejemplo
    input_pH = st.number_input("pH", value=5.2, min_value=0.0, max_value=14.0, format="%.2f") # Ajustado a 5.2
    input_SowingDate_doy = st.number_input("SowingDate(doy)", value=133, min_value=1, max_value=366, step=1)
    input_AnthesisDate_doy = st.number_input("AnthesisDate(doy)", value=229, min_value=1, max_value=366, step=1) # Ajustado a 229 para redondear el ejemplo
    input_HarvestDate_doy = st.number_input("HarvestDate(doy)", value=250, min_value=1, max_value=366, step=1)

    # Calcular GrowingSeason(day) - Asegúrate de que la lógica es correcta para tus datos
    # Basado en tu ejemplo (250 - 133 = 117), parece HarvestDate - SowingDate
    input_GrowingSeason = input_HarvestDate_doy - input_SowingDate_doy
    st.number_input("GrowingSeason(day)", value=input_GrowingSeason, disabled=True) # Mostrar como calculado


    # --- Botón de Predicción ---
    if st.button("Predict"):
        # --- AHORA TODO ESTE BLOQUE ESTÁ DENTRO DE try...except ---
        try:
            # Asegúrate de que las variables de input_X están definidas
            # Crea el DataFrame con los inputs actuales del usuario
            input_values = [
                input_Site, input_Cultivar, input_Elevation, input_Radiation, input_Precipitation,
                input_Tmax, input_Tmin, input_WHC, input_C, input_pH,
                input_SowingDate_doy, input_AnthesisDate_doy, input_HarvestDate_doy, input_GrowingSeason
            ]

            column_names = ['Site', 'Cultivar', 'Elevation(m)', 'Radiation(Mj/m2day)', 'Precipitation(mm)',
                            'Tmax(ºC)', 'Tmin(ºC)', 'WHC(mm)', 'C(%)', 'pH',
                            'SowingDate(doy)', 'AnthesisDate(doy)', 'HarvestDate(doy)', 'GrowingSeason(day)']

            datapredict = pd.DataFrame([input_values], columns=column_names)

            # Convertir a categóricas - CRUCIAL que estos dtypes y los VALORES coincidan con el entrenamiento
            # Streamlit selectbox devuelve strings, convertir a 'category' es necesario
            datapredict["Site"] = datapredict["Site"].astype("category")
            datapredict["Cultivar"] = datapredict["Cultivar"].astype("category")

            # Puedes mostrar el DataFrame de input si quieres para depurar, pero quítalo en producción
            # st.subheader("📋 Input Data for Prediction (Debug)")
            # st.dataframe(datapredict)


            # Realizar predicciones con modelos cargados (desde joblib)
            # Asegúrate de que model_dm, model_ufl, model_cp fueron cargados exitosamente arriba
            pred_dm = model_dm.predict(datapredict)[0]
            pred_ufl = model_ufl.predict(datapredict)[0]
            pred_cp = model_cp.predict(datapredict)[0]

            # Almacenar las predicciones redondeadas en session_state
            # Streamlit detectará el cambio en session_state y hará un rerun,
            # lo que actualizará los st.metric que están fuera del botón.
            st.session_state["pred_dm"] = round(pred_dm, 2)
            st.session_state["pred_ufl"] = round(pred_ufl, 2)
            st.session_state["pred_cp"] = round(pred_cp, 2)

            st.success("✅ Predicciones realizadas correctamente.")

        except Exception as e:
            # Si ocurre CUALQUIER error durante la creación del DF o la predicción:
            st.error(f"❌ Error durante el proceso de predicción: {e}")
            # Opcional: resetear las predicciones en caso de error
            st.session_state["pred_dm"] = 'Error'
            st.session_state["pred_ufl"] = 'Error'
            st.session_state["pred_cp"] = 'Error'


# --- El resto del código (si hay algo más) ---
# Por ejemplo, visualizaciones basadas en las predicciones, etc.
# Asegúrate de que este código también maneja el estado '---' o 'Error' de session_state.

