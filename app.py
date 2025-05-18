import lightgbm
import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import os
from functools import partial # Importar partial

# --- Inicialización de session_state al inicio del script ---
# (Esto ya lo tenías, asegúrate de que está al principio)
for key in ["pred_dm", "pred_ufl", "pred_cp", "prediction_error", "last_datapredict"]:
    if key not in st.session_state:
         # Inicializa predicciones a ---, error a None, DataFrame a None
        st.session_state[key] = '---' if key in ["pred_dm", "pred_ufl", "pred_cp"] else None
# --------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("🌽 Forage Maize Prediction in NW of Spain")

#---------------------
# Cargar modelos LightGBM (asegúrate de que esto está al inicio y maneja errores)
# ... (Tu código de carga de modelos con joblib y manejo de FileNotFoundError/Exception) ...
st.subheader("⚙️ Carga de Modelos")
try:
    # VERIFICA estos nombres de archivo .pkl si son exactamente los que guardaste
    model_dm = joblib.load("DM_lgb_best_model.pkl")
    model_ufl = joblib.load("UFL_lgb_best_model.pkl")
    model_cp = joblib.load("CP_lgb_best_model.pkl")
    st.success("✅ Modelos cargados correctamente.")
    # Asumimos que estos modelos son globales o accesibles en la función perform_prediction_logic
    # Si no, tendríamos que pasarlos a la función, pero declararlos aquí globalmente funciona.
    # global model_dm, model_ufl, model_cp # Descomentar si están definidos localmente y necesitas acceso global
except FileNotFoundError as e:
    st.error(f"❌ Error: Archivo de modelo no encontrado - {e}. Asegúrate de que los archivos .pkl están en la ubicación correcta en Streamlit Cloud.")
    st.stop() # Detiene la ejecución si los modelos no cargan
except Exception as e:
    st.error(f"❌ Error al cargar modelos: {e}")
    st.stop() # Detiene la ejecución si los modelos no cargan


# Cargar mapa (asegúrate de que esto está al inicio y maneja si el archivo no existe)
# ... (Tu código de carga de mapa) ...
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


# Cargar datos desde Excel (para listas desplegables) (asegúrate de que esto está al inicio)
# ... (Tu código de carga de Excel y obtención de site_options, cultivar_options) ...
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
        # --- IMPORTANTE: Obtener las categorías EXACTAS usadas en entrenamiento ---
        # Si no tienes un archivo con las categorías, puedes intentar obtenerlas del modelo cargado si es wrapper
        # O usar las del DataFrame completo si incluye TODAS las categorías posibles
        try:
             # Esto intenta obtener las categorías de los modelos si son wrappers de LightGBM/XGBoost
             # y tienen la información de las categorías originales.
             # Puede variar dependiendo de cómo se guardó el modelo.
             site_categories_train = model_dm.get_params()['categorical_feature'][0] # Ejemplo si se guarda como en el notebook
             cultivar_categories_train = model_dm.get_params()['categorical_feature'][1] # Ejemplo

             # Nota: model.get_params() devuelve los parámetros *antes* del fit.
             # Necesitas las categorías *usadas durante* el fit. Si las guardaste por separado es mejor.
             # Si no, usa las del df cargado si sabes que son todas las posibles.
             if site_options and cultivar_options: # Si se cargaron opciones del excel
                  site_categories_train = site_options
                  cultivar_categories_train = cultivar_options
             else:
                  # Define listas manuales si no pudiste cargarlas o extraerlas
                  site_categories_train = ['Barcia', 'Deza', 'Grado', 'Ordes', 'Ribadeo', 'Sarria', 'Villaviciosa'] # Ejemplo manual
                  cultivar_categories_train = ['A200', 'A300', 'A400', 'G200', 'G300', 'G400'] # Ejemplo manual

        except Exception as e:
             st.warning(f"⚠️ No se pudieron obtener las categorías de entrenamiento: {e}. Usando opciones de Excel o listas por defecto.")
             if site_options and cultivar_options:
                  site_categories_train = site_options
                  cultivar_categories_train = cultivar_options
             else:
                  site_categories_train = ['Barcia', 'Deza', 'Grado', 'Ordes', 'Ribadeo', 'Sarria', 'Villaviciosa'] # Ejemplo manual
                  cultivar_categories_train = ['A200', 'A300', 'A400', 'G200', 'G300', 'G400'] # Ejemplo manual


    else:
        st.error(f"❌ Error: Archivo Excel no encontrado ({excel_file_path}). No se pueden cargar los datos de entrada.")
        excel_data_found = False
        site_options = ["DefaultSite"] # Opciones por defecto
        cultivar_options = ["DefaultCultivar"]
        site_categories_train = site_options # Usar por defecto para categorías de entrenamiento
        cultivar_categories_train = cultivar_options
        st.stop() # Detiene la ejecución si no se pueden cargar los datos
except Exception as e:
    st.error(f"❌ Error al cargar datos de Excel: {e}")
    excel_data_found = False
    site_options = ["DefaultSite"] # Opciones por defecto
    cultivar_options = ["DefaultCultivar"]
    site_categories_train = site_options # Usar por defecto para categorías de entrenamiento
    cultivar_categories_train = cultivar_options
    st.stop() # Detiene la ejecución si ocurre otro error


# --- Definir la lógica de predicción en una función ---
# Esta función se llamará cuando se haga clic en el botón
def perform_prediction_logic(site_val, cultivar_val, elevation_val, radiation_val, precipitation_val,
                             tmax_val, tmin_val, whc_val, c_val, ph_val,
                             sowing_doy_val, anthe_doy_val, harvest_doy_val, growing_season_val,
                             # Pasar las categorías de entrenamiento también
                             site_categories, cultivar_categories):
    """
    Recoge inputs, crea DataFrame, predice, actualiza session state.
    Esta función se ejecuta cuando se hace clic en el botón "Predict".
    """
    # Inicia el estado de error a None
    st.session_state["prediction_error"] = None
    # Inicia el estado del DataFrame usado para predecir a None
    st.session_state["last_datapredict"] = None


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
        datapredict["Site"] = pd.Categorical(datapredict["Site"], categories=site_categories)
        datapredict["Cultivar"] = pd.Categorical(datapredict["Cultivar"], categories=cultivar_categories)

        # Opcional: Si quieres que los valores no vistos se conviertan en NaN en lugar de dar error,
        # puedes añadir el argumento 'ordered=False' y 'unknown=np.nan' si tu pandas lo soporta,
        # pero los modelos de árbol no suelen manejar NaN en categóricas nativamente.
        # Lo más seguro es asegurar que las categorías de entrada son válidas o manejar el error.


        # 4. Realizar las predicciones
        # Asegúrate de que model_dm, model_ufl, model_cp fueron cargados exitosamente al inicio
        # Si usaste 'global' arriba, puedes acceder a ellos directamente.
        # Si no, podrías pasarlos como argumentos a esta función también.
        pred_dm = model_dm.predict(datapredict)[0]
        pred_ufl = model_ufl.predict(datapredict)[0]
        pred_cp = model_cp.predict(datapredict)[0]

        # 5. Almacenar los resultados y el DataFrame en session_state
        st.session_state["pred_dm"] = round(pred_dm, 2)
        st.session_state["pred_ufl"] = round(pred_ufl, 2)
        st.session_state["pred_cp"] = round(pred_cp, 2)
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

    # Definir todos los widgets de input. Es importante asignarles una 'key' única.
    # Los valores de estos widgets se leerán cuando se haga clic en el botón
    # y se pasarán a la función perform_prediction_logic.

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
    st.number_input("GrowingSeason(day)", value=float(input_GrowingSeason), disabled=True, key='input_growingseason_display') # Usar float porque number_input espera float


    # --- Botón de Predicción usando on_click ---
    # Aquí llamamos a la función perform_prediction_logic usando partial
    # Le pasamos los valores actuales de los inputs al momento del clic
    st.button(
        "Predict",
        on_click=partial(
            perform_prediction_logic,
            # Pasa cada valor de input como argumento a la función
            input_Site, input_Cultivar, input_Elevation, input_Radiation, input_Precipitation,
            input_Tmax, input_Tmin, input_WHC, input_C, input_pH,
            input_SowingDate_doy, input_AnthesisDate_doy, input_HarvestDate_doy, input_GrowingSeason,
            # Pasa también las listas de categorías de entrenamiento/válidas
            site_categories_train, cultivar_categories_train
        )
        # NO hay if st.button(): alrededor de la lógica de predicción
    )


# --- Resto del código (si hay algo más) ---
# ...
