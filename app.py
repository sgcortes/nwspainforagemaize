import lightgbm
import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# Cargar modelos LightGBM
with open("DM_lgb_best_model.pkl", "rb") as f:
    model_dm = pickle.load(f)

with open("UFL_lgb_best_model.pkl", "rb") as f:
    model_ufl = pickle.load(f)

with open("CP_lgb_best_model.pkl", "rb") as f:
    model_cp = pickle.load(f)

# Cargar mapa
map_image = Image.open("AsturiasGalicia2.jpg")

# Cargar datos desde Excel
df = pd.read_excel("260324_ENG_MaizeForageSpainNWwtYearRadDay.xlsx")

st.set_page_config(layout="wide")
st.title("🌽 Forage Maize Prediction in NW of Spain")

col1, col2 = st.columns([1, 2])

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
        # Filtrar por sitio
        df_site = df[df["Site"] == site]

        # Valores estadísticos
        Tmin_min = df_site["Tmin(ºC)"].min()
        Tmin_max = df_site["Tmin(ºC)"].max()
        Tmin_mean = df_site["Tmin(ºC)"].mean()

        Tmax_min = df_site["Tmax(ºC)"].min()
        Tmax_max = df_site["Tmax(ºC)"].max()
        Tmax_mean = df_site["Tmax(ºC)"].mean()

        prec_min = df_site["Precipitation(mm)"].min()
        prec_max = df_site["Precipitation(mm)"].max()
        prec_mean = df_site["Precipitation(mm)"].mean()

        rad_min = df_site["Radiation(MJ/m2day)"].min()
        rad_max = df_site["Radiation(MJ/m2day)"].max()
        rad_mean = df_site["Radiation(MJ/m2day)"].mean()

        whc_mean = df_site["WHC(mm)"].mean()
        c_mean = df_site["C(%)"].mean()
        ph_mean = df_site["ph"].mean()
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

        sowing_doy = sowing_map[sowing_label]
        harvest_doy = harvest_map[harvest_label]

        growing_season = harvest_doy - sowing_doy

        # Crear fila para predicción
        datapredict = pd.DataFrame([{
            "Site": site,
            "Cultivar": cultivar,
            "Radiacion(Mj/m2day)": radiation,
            "Precipitation(mm)": precipitation,
            "Tmax(ºC)": tmax,
            "Tmin(ºC)": tmin,
            "WHC(mm)": whc_mean,
            "C(%)": c_mean,
            "ph": ph_mean,
            "SowingDate(doy)": sowing_doy,
            "AnthesisDate(doy)":anthe_mean,
            "HarvestDate(doy)": harvest_doy,
            "GrowingSeason(day)": growing_season
        }])

        st.subheader("📋 Input Data for Prediction")
        st.dataframe(datapredict)

        pred_dm = model_dm.predict(datapredict)[0]
        pred_ufl = model_ufl.predict(datapredict)[0]
        pred_cp = model_cp.predict(datapredict)[0]

        st.session_state["pred_dm"] = round(pred_dm, 2)
        st.session_state["pred_ufl"] = round(pred_ufl, 2)
        st.session_state["pred_cp"] = round(pred_cp, 2)

with col2:
    st.image(map_image, caption="Study area: Galicia, Asturias", use_container_width=True)

    if "pred_dm" in st.session_state:
        col_dm, col_ufl, col_cp = st.columns(3)
        col_dm.markdown(f"### Dry Matter: {st.session_state['pred_dm']} kgDM/ha")
        col_ufl.markdown(f"### UFL/ha: {st.session_state['pred_ufl']}")
        col_cp.markdown(f"### Crude Protein: {st.session_state['pred_cp']}")
