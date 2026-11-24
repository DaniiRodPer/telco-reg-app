import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer

# Cargar el modelo y el DictVectorizer
with open('churn-model.pck', 'rb') as f:
    dv, model = pickle.load(f)

# Título de la aplicación
st.title("Predicción de Churn de Clientes")

# Formulario con 11 campos: los 3 numéricos + los 8 categóricos menos influyentes
st.header("Introduce los datos del cliente:")

# 3 campos numéricos
tenure = st.number_input("Antigüedad (meses)", min_value=0, max_value=100, step=1)
monthly_charges = st.number_input("Cargos mensuales", min_value=0.0, max_value=500.0, step=0.1)
total_charges = st.number_input("Cargos totales", min_value=0.0, max_value=10000.0, step=0.1)

# 8 campos categóricos menos importantes (MI más baja)
streaming_movies = st.selectbox("Películas en streaming", ["no", "yes", "no_internet_service"])
paperless_billing = st.selectbox("Facturación sin papel", ["no", "yes"])
dependents = st.selectbox("Dependientes", ["no", "yes"])
partner = st.selectbox("Pareja", ["no", "yes"])
senior_citizen = st.selectbox("¿Es persona mayor?", ["no", "yes"])
multiplelines = st.selectbox("Múltiples líneas", ["no", "no_phone_service", "yes"])
phoneservice = st.selectbox("Servicio telefónico", ["no", "yes"])
gender = st.selectbox("Género", ["female", "male"])

# Botón de predicción
if st.button("Predecir"):
    # Diccionario completo: 11 campos del usuario + los 8 categóricos más importantes ocultos
    client_data = {
        "contract": "month-to-month",              # Oculto (Top 1 más importante - por defecto)
        "dependents": dependents,
        "deviceprotection": "no",                  # Oculto (Top 6 más importante - por defecto)
        "gender": gender,
        "internetservice": "fiber_optic",          # Oculto (Top 4 más importante - por defecto)
        "monthlycharges": monthly_charges,
        "multiplelines": multiplelines,
        "onlinebackup": "no",                      # Oculto (Top 5 más importante - por defecto)
        "onlinesecurity": "no",                    # Oculto (Top 2 más importante - por defecto)
        "paperlessbilling": paperless_billing,
        "partner": partner,
        "paymentmethod": "electronic_check",       # Oculto (Top 7 más importante - por defecto)
        "phoneservice": phoneservice,
        "seniorcitizen": senior_citizen,
        "streamingmovies": streaming_movies,
        "streamingtv": "no",                       # Oculto (Top 8 más importante - por defecto)
        "techsupport": "no",                       # Oculto (Top 3 más importante - por defecto)
        "tenure": tenure,
        "totalcharges": total_charges
    }

    # Transformar los datos del cliente
    X_client = dv.transform([client_data])

    # Realizar la predicción
    y_pred_proba = model.predict_proba(X_client)[0][1]  # Probabilidad de churn

    # Mostrar resultado
    st.subheader("Resultado:")
    if y_pred_proba > 0.5:
        st.error(f"El cliente tiene una alta probabilidad de churn: {y_pred_proba:.2f}")
    else:
        st.success(f"El cliente tiene una baja probabilidad de churn: {y_pred_proba:.2f}")
