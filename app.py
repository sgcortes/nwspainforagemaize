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

# Imagen del mapa
map_image = Image.open("AsturiasGalicia2.jpg")

st.set_page_config(layout="wide")

st.title("Forage Maize Prediction in NW of Spain")

# Layout con dos columnas
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Input Controls")

    site = st.selectbox("Nearest Site", ["Ordes", "Deza", "Sarria", "Ribadeo", "Barcia", "Grado", "Villaviciosa"])
    cultivar = st.selectbox("Cultivar", ["A200", "A300", "A400", "G200", "G300", "G400"])
    sowing_date = st.selectbox("Sowing Date", ["Mid-May", "End-May", "Early June"])
    harvest_date = st.selectbox("Harvest Date", ["Early-Sept", "Mid-Sept", "Late-Sept"])
    weather = st.selectbox("Radiation & Weather", ["Good Year", "Average Year", "Bad Year"])

    if st.button("Predict"):
        # Crear muestra sintética
        sample = pd.DataFrame([{
            "Site": site,
            "Cultivar": cultivar,
            "Radiation": "Radiation",
            "Precipitation": "Precipitation",
            "Tamx": "Tamx",
            "Tmin": "Tmin",
            "WHC": "WHC",
            "C": "C",
            "ph": "ph",
            "Sowingdate": sowing_date,
            "Anthesis date": "Anthesis date",
            "HarvestDate": harvest_date,
            "Growing Season": "Growing Season"
        }])

        # Predicciones
        pred_dm = model_dm.predict(sample)[0]
        pred_ufl = model_ufl.predict(sample)[0]
        pred_cp = model_cp.predict(sample)[0]

        st.session_state["pred_dm"] = round(pred_dm, 2)
        st.session_state["pred_ufl"] = round(pred_ufl, 2)
        st.session_state["pred_cp"] = round(pred_cp, 2)

with col2:
    st.image(map_image, caption="Study area: Galicia, Asturias, and Cantabria", use_container_width=True)

    if "pred_dm" in st.session_state:
        col_dm, col_ufl, col_cp = st.columns(3)
        col_dm.markdown(f"### Dry Matter: {st.session_state['pred_dm']} kgDM/ha")
        col_ufl.markdown(f"### UFL/ha: {st.session_state['pred_ufl']}")
        col_cp.markdown(f"### Crude Protein: {st.session_state['pred_cp']}")


