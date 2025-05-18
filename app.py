import lightgbm
import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import os
from functools import partial
# import numpy as np # Necesario si usas np.nan en Categorical

# --- Inicialización de session_state al inicio del script ---
# Inicializa solo las claves necesarias
# prediction_error y last_datapredict no necesitan estado inicial si los manejas en la función
for key in ["pred_dm", "pred_ufl", "pred_cp"]:
    if key not in st.session_state:
        st.session_state[key] = '---' # Valor por defecto antes de la primera predicción

# Inicializa el estado de error y el DataFrame usado a None
if "prediction_error" not in st.session_state:
    st.session_state["prediction_error"] = None
if "last_datapredict" not in st.session_state:
    st.session_state["last_datapredict"] = None


st.set_page_config(layout="wide")
st.title("🌽 Forage Maize Prediction in NW of Spain")

# --- Código para verificar las versiones de las librerías ---
st.subheader("📦 Versiones de Librerías")
try:
    st.info(f"Streamlit version: {st.__version__}") # Añadido Streamlit
except Exception as e:
    st.error(f"Error getting Streamlit version: {e}")
try:
    st.info(f"LightGBM version: {lightgbm.__version__}")
except Exception as e:
    st.error(f"Error getting LightGBM version: {e}")
try:
    st.info(f"Pandas version: {pd.__version__}")
except Exception as e:
    st.error(f"Error getting Pandas version: {e}")
try:
    st.info(f"Joblib version: {joblib.__version__}")
except Exception as e:
    st.error(f"Error getting Joblib version: {e}")
# --- Fin verificación versiones ---


#---------------------
# Cargar modelos LightGBM (asegúrate de que esto está al inicio y maneja errores)
st.subheader("⚙️ Carga de Modelos")
try:
    # VERIFICA estos nombres de archivo .pkl si son exactamente los que guardaste
    model_dm = joblib.load("DM_lgb_best_model.pkl")
    model_ufl = joblib.load("UFL_lgb_best_model.pkl")
    model_cp = joblib.load("CP_lgb_best_model.pkl")
    st.success("✅ Modelos cargados correctamente.")
    # Los modelos cargados serán accesibles en perform_prediction_logic
    # gracias a que están en el scope global/del módulo.
except FileNotFoundError as e:
    st.error(f"❌ Error: Archivo de modelo no encontrado - {e}. Asegúrate de que los archivos .pkl están en la ubicación correcta en Streamlit Cloud.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error al cargar modelos: {e}")
    st.stop()


# Cargar mapa (asegúrate de que esto está al inicio y maneja si el archivo no existe)
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


# Cargar datos desde Excel (para listas desplegables y para obtener categorías de entrenamiento)
excel_file_path = "260324_ENG_MaizeForageSpainNWwtYearRadDay.xlsx"
st.subheader("📂 Carga de Datos de Entrada (Excel)")
try:
    if os.path.exists(excel_file_path):
        df = pd.read_excel(excel_file_path)
        st.success("✅ Datos de Excel cargados correctamente.")
        excel_data_found = True
        # Obtener listas únicas para selectbox
        site_options = sorted(df['Site'].unique().tolist())
        cultivar_options = sorted(df['Cultivar'].unique().tolist())

        # --- IMPORTANTE: Obtener las categorías EXACTAS usadas en entrenamiento ---
        # Usamos las del DataFrame cargado si asumimos que contiene todas las posibles categorías
        # Si no, cárgalas desde un archivo aparte o defiéndelas manualmente si las sabes
        site_categories_train = site_options # Asumimos que las opciones del excel son las categorías de entrenamiento
        cultivar_categories_train = cultivar_options # Asumimos que las opciones del excel son las categorías de entrenamiento

        # Si tienes un archivo con las categorías o puedes extraerlas del modelo, úsalas en lugar de las del excel
        # Ejemplo (si guardaste las categorías con el modelo o aparte):
        # try:
        #     site_categories_train = joblib.load("site_categories.pkl")
        #     cultivar_categories_train = joblib.load("cultivar_categories.pkl")
        #     st.info("✅ Categorías de entrenamiento cargadas desde archivo(s).")
        # except FileNotFoundError:
        #     st.warning("⚠️ Archivos de categorías de entrenamiento no encontrados. Usando opciones de Excel.")
        #     site_categories_train = site_options
        #     cultivar_categories_train = cultivar_options
        # except Exception as e:
        #      st.warning(f"⚠️ Error al cargar categorías de entrenamiento: {e}. Usando opciones de Excel.")
        #      site_categories_train = site_options
        #      cultivar_categories_train = cultivar_options


    else:
        st.error(f"❌ Error: Archivo Excel no encontrado ({excel_file_path}). No se pueden cargar los datos de entrada.")
        excel_data_found = False
        site_options = ["DefaultSite"]
        cultivar_options = ["DefaultCultivar"]
        site_categories_train = site_options # Usar default para categorías de entrenamiento si excel falla
        cultivar_categories_train = cultivar_options
        st.stop() # Detiene la ejecución si no se pueden cargar los datos
except Exception as e:
    st.error(f"❌ Error al cargar datos de Excel: {e}")
    excel_data_found = False
    site_options = ["DefaultSite"]
    cultivar_options = ["DefaultCultivar"]
    site_categories_train = site_options # Usar default para categorías de entrenamiento si excel falla
    cultivar_categories_train = cultivar_options
    st.stop()


# --- Definir la lógica de predicción en una función ---
# Esta función se llamará cuando se haga clic en el botón
# Recibe los valores de los inputs y las listas de categorías válidas
def perform_prediction_logic(site_val, cultivar_val, elevation_val, radiation_val, precipitation_val,
                             tmax_val, tmin_val, whc_val, c_val, ph_val,
                             sowing_doy_val, anthe_doy_val, harvest_doy_val, growing_season_val,
                             site_categories, cultivar_categories):
    """
    Recoge inputs, crea DataFrame, predice, actualiza session state.
    Esta función se ejecuta cuando se hace clic en el botón "Predict".
    """
    # Limpiar errores previos al inicio de una nueva predicción
    st.session_state["prediction_error"] = None
    st.session_state["last_datapredict"] = None # Limpiar el DataFrame previo


    try:
        # 1. Recoger los valores de los inputs que fueron pasados a la función
        input_values = [
            site_val, cultivar_val, elevation_val, radiation_val, precipitation_val,
            tmax_val, tmin_val, whc_val, c_val, ph_val,
            sowing_doy_val, anthe_doy_val, harvest_doy_val, growing_season_val
        ]

        column_names = ['Site', 'Cultivar', 'Elevation(m)', 'Radiation(Mj/m2day)', 'Precipitation(mm)',
                        'Tmax(ºC)', 'Tmin(ºC)', 'WHC(mm)', 'C(%)', 'pH',
                        'SowingDate(doy)', 'AnthesisDate(doy)', 'HarvestDate(doy)', 'GrowingSeason(day)']

        # 2. Crear el DataFrame para la predicción
        datapredict = pd.DataFrame([input_values], columns=column_names)

        # 3. Convertir a categóricas usando las categorías EXACTAS del entrenamiento
        # Esto es crucial. Si un valor de input no está en estas categorías, puede dar error.
        # Usamos pd.Categorical con la lista explícita de categorías.
        # Si un valor seleccionado no está en la lista de categorías de entrenamiento, pandas lo pondrá como NaN.
        # Los modelos de árbol como LightGBM/XGBoost guardados con joblib deberían manejar NaN en categóricas,
        # pero es mejor asegurar que las categorías de input están en las de entrenamiento.
        datapredict["Site"] = pd.Categorical(datapredict["Site"], categories=site_categories)
        datapredict["Cultivar"] = pd.Categorical(datapredict["Cultivar"], categories=cultivar_categories)

        # Asegurarse del orden de las columnas (aunque pandas suele mantenerlo si vienen de dict/lista)
        # Si tus modelos esperan las columnas en un orden específico, asegúralo aquí:
        # expected_order = [...] # Define la lista de columnas en el orden correcto
        # datapredict = datapredict[expected_order]


        # 4. Realizar las predicciones
        # Acceder a los modelos cargados globalmente
        global model_dm, model_ufl, model_cp

        pred_dm = model_dm.predict(datapredict)[0]
        pred_ufl = model_ufl.predict(datapredict)[0]
        pred_cp = model_cp.predict(datapredict)[0]

        # 5. Almacenar los resultados y el DataFrame en session_state
        st.session_state["pred_dm"] = round(float(pred_dm), 2) # Asegurar float antes de redondear
        st.session_state["pred_ufl"] = round(float(pred_ufl), 2)
        st.session_state["pred_cp"] = round(float(pred_cp), 2)
        st.session_state["last_datapredict"] = datapredict # Guardar el DataFrame usado

        # Si todo va bien, el error permanece None

    except Exception as e:
        # 6. Capturar errores y almacenarlos en session_state
        st.session_state["prediction_error"] = f"❌ Error durante la predicción: {e}"
        # Resetear las predicciones a un estado de error
        st.session_state["pred_dm"] = 'Error'
        st.session_state["pred_ufl"] = 'Error'
        st.session_state["pred_cp"] = 'Error'
        st.session_state["last_datapredict"] = None # No guardar un DataFrame que causó error


# --- Diseño de la interfaz ---
col1, col2 = st.columns([1, 2])

with col2:
    # ... (Mostrar mapa) ...
    if map_found:
       st.image(map_image, caption=map_caption, use_container_width=True)


    st.subheader("📈 Predicciones")
    # Mostrar predicciones usando st.metric (ESTO VA FUERA DE CUALQUIER if button)
    col_dm, col_ufl, col_cp = st.columns(3)
    col_dm.metric(label="Dry Matter (kg DM/ha)", value=st.session_state.get('pred_dm', '---'))
    col_ufl.metric(label="UFL/ha", value=st.session_state.get('pred_ufl', '---'))
    col_cp.metric(label="Crude Protein (kg CP/ha)", value=st.session_state.get('pred_cp', '---'))

    # Mostrar mensaje de error si existe uno en session_state
    if st.session_state.get("prediction_error"):
        st.error(st.session_state["prediction_error"])

    # Mostrar el DataFrame de input que se usó para la predicción (si existe en session_state)
    if st.session_state.get("last_datapredict") is not None:
         st.subheader("📋 Input Data Used for Prediction")
         st.dataframe(st.session_state["last_datapredict"])


# --- Columna 1 para Inputs y Botón ---
with col1:
    st.header("Parameters")

    # Definir los widgets de input originales
    # Es importante asignarles una 'key' única para que Streamlit los identifique correctamente.

    if excel_data_found:
        input_Site = st.selectbox("Site", site_options, key='input_site')
        input_Cultivar = st.selectbox("Cultivar", cultivar_options, key='input_cultivar')
    else:
        input_Site = st.selectbox("Site", ["DefaultSite"], key='input_site')
        input_Cultivar = st.selectbox("Cultivar", ["DefaultCultivar"], key='input_cultivar')


    input_Elevation = st.number_input("Elevation(m)", value=25.0, min_value=0.0, format="%.1f", key='input_elevation')
    input_Radiation = st.number_input("Radiation(Mj/m2day)", value=21.0, min_value=0.0, format="%.1f", key='input_radiation')
    input_Precipitation = st.number_input("Precipitation(mm)", value=56.6, min_value=0.0, format="%.1f", key='input_precipitation')
    input_Tmax = st.number_input("Tmax(ºC)", value=20.5, format="%.1f", key='input_tmax')
    input_Tmin = st.number_input("Tmin(ºC)", value=16.7, format="%.1f", key='input_tmin')
    input_WHC = st.number_input("WHC(mm)", value=90.0, min_value=0.0, format="%.1f", key='input_whc')
    input_C = st.number_input("C(%)", value=1.9, min_value=0.0, format="%.2f", key='input_c')
    input_pH = st.number_input("pH", value=5.2, min_value=0.0, max_value=14.0, format="%.2f", key='input_ph')
    input_SowingDate_doy = st.number_input("SowingDate(doy)", value=133, min_value=1, max_value=366, step=1, key='input_sowing')
    input_AnthesisDate_doy = st.number_input("AnthesisDate(doy)", value=229, min_value=1, max_value=366, step=1, key='input_anthesis')
    input_HarvestDate_doy = st.number_input("HarvestDate(doy)", value=250, min_value=1, max_value=366, step=1, key='input_harvest')

    # Calcular GrowingSeason(day)
    input_GrowingSeason = input_HarvestDate_doy - input_SowingDate_doy
    # Usar un widget disabled para mostrar el valor calculado
    st.number_input("GrowingSeason(day)", value=float(input_GrowingSeason), disabled=True, key='input_growingseason_display')


    # --- Botón de Predicción usando on_click ---
    # Definir el botón y pasar la función de lógica y los valores de input
    st.button(
        "Predict",
        on_click=partial(
            perform_prediction_logic,
            # Pasa el VALOR actual de cada widget al hacer clic
            input_Site, # valor de st.selectbox
            input_Cultivar, # valor de st.selectbox
            input_Elevation, # valor de st.number_input
            input_Radiation, # valor de st.number_input
            input_Precipitation, # valor de st.number_input
            input_Tmax, # valor de st.number_input
            input_Tmin, # valor de st.number_input
            input_WHC, # valor de st.number_input
            input_C, # valor de st.number_input
            input_pH, # valor de st.number_input
            input_SowingDate_doy, # valor de st.number_input
            input_AnthesisDate_doy, # valor de st.number_input
            input_HarvestDate_doy, # valor de st.number_input
            input_GrowingSeason, # valor calculado
            # Pasar las listas de categorías de entrenamiento
            site_categories_train,
            cultivar_categories_train
        )
        # NO hay if st.button(): aquí. La lógica está en la función perform_prediction_logic.
    )

# --- Resto del código (si hay algo más) ---
# ...

# --- Resto del código (si hay algo más) ---
# ...
