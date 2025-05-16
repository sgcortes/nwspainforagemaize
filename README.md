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

