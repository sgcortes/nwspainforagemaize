# Forage MAize Yield PRediction for NW of Spain
Forage Yield (Dry Matter kg/ha, UFL/ha, Crude Protein kg/ha) in the NW of Spain


# 🌽 Forage Maize Prediction App

This **Streamlit web application** allows users to predict forage maize yield and nutritional quality in Northwestern Spain based on environmental conditions and cultivar selection.

## 🔍 Overview

Users can interactively select:
- 📍 Nearest experimental site
- 🌱 Maize cultivar
- 📅 Sowing and harvest dates
- ☀️ Expected weather conditions

The app outputs predicted values for:
- **Dry Matter (kg DM/ha)**
- **UFL/ha** *(Forage energy content)*
- **Crude Protein (kg CP/ha)**

These predictions are made using pre-trained **LightGBM models**.

---

## 🗂 File Structure

Ensure your project folder contains the following files:

```
📁 your-app-folder/
├── app.py
├── requirements.txt
├── DM_lgb_best_model.pkl
├── UFL_lgb_best_model.pkl
├── CP_lgb_best_model.pkl
├── 260324_ENG_MaizeForageSpainNWwtYearRadDay.xlsx
├── AsturiasGalicia2.jpg
```

---

## 🚀 How to Deploy on Streamlit Cloud

1. Upload all files to a public GitHub repository.
2. Visit [streamlit.io/cloud](https://streamlit.io/cloud).
3. Log in and click **"New app"**.
4. Select your repo and make sure `app.py` is the entry point.
5. Streamlit will automatically install dependencies from `requirements.txt`.

---

## 🧪 Run Locally

To run the app on your local machine:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📬 Contact

For questions or suggestions, please contact the development team.

---

*Developed using Python, Streamlit, and LightGBM.*

