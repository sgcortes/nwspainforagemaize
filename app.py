import lightgbm
import streamlit as st
import pandas as pd
import joblib
from PIL import Image

st.set_page_config(layout="wide")
st.title("🌽 Forage Maize Prediction in NW of Spain")
#---------------------
# Cargar modelos LightGBM
try:
    model_dm = joblib.load("DM_lgb_best_model.pkl")
    model_ufl = joblib.load("UFL_lgb_best_model.pkl")
    model_cp = joblib.load("CP_lgb_best_model.pkl")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Cargar mapa
map_image = Image.open("AsturiasGalicia2.jpg")

# Cargar datos desde Excel
df = pd.read_excel("260324_ENG_MaizeForageSpainNWwtYearRadDay.xlsx")

col1, col2 = st.columns([1, 2])
with col2:
    st.image(map_image, caption="Study area: Galicia, Asturias", use_container_width=True)
    # Inicializar session_state si aún no existe
    if 'pred_dm' not in st.session_state:
        st.session_state['pred_dm'] = '---'
    if 'pred_ufl' not in st.session_state:
        st.session_state['pred_ufl'] = '---'
    if 'pred_cp' not in st.session_state:
        st.session_state['pred_cp'] = '---'


col_dm, col_ufl, col_cp = st.columns(3)

# Usar st.metric para mostrar las predicciones
col_dm.metric(label="Dry Matter (kg DM/ha)", value=st.session_state.get('pred_dm', '---'))
col_ufl.metric(label="UFL/ha", value=st.session_state.get('pred_ufl', '---'))
col_cp.metric(label="Crude Protein (kg CP/ha)", value=st.session_state.get('pred_cp', '---'))

   
    #st.subheader("📋 Input Data for Prediction")
    #st.dataframe(datapredict)
with col1:
    st.header("Input Controls")

    site = st.selectbox("Nearest Site", sorted(df["Site"].unique()))
    cultivar = st.selectbox("Cultivar", ["A200", "A300", "A400", "G200", "G300", "G400"])
    sowing_label = st.selectbox("Sowing Date", ["Mid-May", "End-May", "Early June"])
    harvest_label = st.selectbox("Harvest Date", ["Early-Sept", "Mid-Sept", "Late-Sept"])
    weather = st.selectbox("Radiation & Weather", ["Good Year", "Average Year", "Bad Year"])

    sowing_map = {"Mid-May": 133, "End-May": 151, "Early June": 167}
    harvest_map = {"Early-Sept": 250, "Mid-Sept": 264, "Late-Sept": 287}

    if st.button("Predict"):
        df_site = df[df["Site"] == site]

        Tmin_min = df_site["Tmin(ºC)"].min()
        Tmin_max = df_site["Tmin(ºC)"].max()
        Tmin_mean = df_site["Tmin(ºC)"].mean()

        Tmax_min = df_site["Tmax(ºC)"].min()
        Tmax_max = df_site["Tmax(ºC)"].max()
        Tmax_mean = df_site["Tmax(ºC)"].mean()

        prec_min = df_site["Precipitation(mm)"].min()
        prec_max = df_site["Precipitation(mm)"].max()
        prec_mean = df_site["Precipitation(mm)"].mean()

        rad_min = df_site["Radiation(Mj/m2day)"].min()
        rad_max = df_site["Radiation(Mj/m2day)"].max()
        rad_mean = df_site["Radiation(Mj/m2day)"].mean()

        whc_mean = df_site["WHC(mm)"].mean()
        c_mean = df_site["C(%)"].mean()
        ph_mean = df_site["pH"].mean()
        anthe_mean = df_site["AnthesisDate(doy)"].mean()

        if weather == "Good Year":
            radiation = rad_max
            precipitation = prec_min
            tmin = Tmin_max
            tmax = Tmax_min
        elif weather == "Bad Year":
            radiation = rad_min
            precipitation = prec_min
            tmin = Tmin_min
            tmax = Tmax_max
        else:
            radiation = rad_mean
            precipitation = prec_mean
            tmin = Tmin_mean
            tmax = Tmax_mean

        if site == "Deza":
            elevation = 400
        elif site =="Barcia":
            elevation = 25
        elif site =="Grado":
            elevation = 50
        elif site =="Ordes":
            elevation = 300
        elif site =="Ribadeo":
            elevation = 43
        elif site =="Ribadeo":
            elevation = 43
        elif site =="Sarria":
            elevation = 520
        elif site =="Villaviciosa":
            elevation = 10
        
        sowing_doy = sowing_map[sowing_label]
        harvest_doy = harvest_map[harvest_label]
        growing_season = harvest_doy - sowing_doy

        datapredict = pd.DataFrame([{
            "Site": site,
            "Cultivar": cultivar,
            "Elevation(m)": elevation,
            "Radiation(Mj/m2day)": radiation,        
            "Precipitation(mm)": precipitation,
            "Tmax(ºC)": tmax,
            "Tmin(ºC)": tmin,
            "WHC(mm)": whc_mean,
            "C(%)": c_mean,
            "pH": ph_mean,                           
            "SowingDate(doy)": sowing_doy,
            "AnthesisDate(doy)": anthe_mean,
            "HarvestDate(doy)": harvest_doy,
            "GrowingSeason(day)": growing_season
        }])

        datapredict["Site"] = datapredict["Site"].astype("category")
        datapredict["Cultivar"] = datapredict["Cultivar"].astype("category")

        #
        columnas = ['Site', 'Cultivar', 'Elevation(m)', 'Radiation(Mj/m2day)','Precipitation(mm)', 'Tmax(ºC)', 'Tmin(ºC)', 'WHC(mm)', 'C(%)', 'pH',
       'SowingDate(doy)', 'AnthesisDate(doy)', 'HarvestDate(doy)','GrowingSeason(day)']

        datos = ['Barcia','A200',25,21,56.6,20.5,16.7,90,1.8999999999999997,5.200000000000003,133,228.70531400966183,250,117]
        X_nuevos_datos = pd.DataFrame([datos], columns=columnas)
        X_nuevos_datos["Site"] = X_nuevos_datos["Site"].astype("category")
        X_nuevos_datos["Cultivar"] = X_nuevos_datos["Cultivar"].astype("category")
        print(X_nuevos_datos.shape)
        datapredict = X_nuevos_datos 
        #
        
        expected_columns = [
    'Site', 'Cultivar', 'Elevation(m)', 'Radiation(Mj/m2day)', 'Precipitation(mm)',
    'Tmax(ºC)', 'Tmin(ºC)', 'WHC(mm)', 'C(%)', 'pH',
    'SowingDate(doy)', 'AnthesisDate(doy)', 'HarvestDate(doy)', 'GrowingSeason(day)']
        datapredict = datapredict[expected_columns]
        #st.write("Shape del DataFrame:", datapredict.shape)
        #st.write("Columnas del DataFrame:", datapredict.columns.tolist())
        st.session_state["pred_dm"] = round(model_dm.predict(datapredict)[0], 2)
        st.session_state["pred_ufl"] = round(model_ufl.predict(datapredict)[0], 2)
        st.session_state["pred_cp"] = round(model_cp.predict(datapredict)[0], 2)


# --- El resto del código (si hay algo más) ---
# Por ejemplo, visualizaciones basadas en las predicciones, etc.
# Asegúrate de que este código también maneja el estado '---' o 'Error' de session_state.

