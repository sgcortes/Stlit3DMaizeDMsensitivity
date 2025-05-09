# archivo: app.py

import streamlit as st
import pandas as pd
import plotly.express as px

# Leer datos desde archivo (o usa synthetic_df directamente si lo cargas antes)
df = pd.read_excel("Synthetic_Maize_Predictions.xlsx")

# Asegurar tipo correcto
df["Condition"] = df["Condition"].astype(int)

# Mapeo de colores y etiquetas
color_labels = {1: "Favorable conds", -1: "Unfavorable conds", 0: "Regular conds"}
df["Condition_Label"] = df["Condition"].map(color_labels)

# Crear gráfico 3D interactivo
fig = px.scatter_3d(
    df,
    x="GrowingSeason(day)",
    y="Radiation(Mj/m2day)",
    z="Predicted_kgDM/ha",
    color="Condition_Label",
    color_discrete_map={
        "Favorable conds": "green",
        "Unfavorable conds": "red",
        "Regular conds": "gray"
    },
    title="3D Scatter: Growing Season vs Radiation vs Predicted kgDM/ha",
    opacity=0.8
)

# Mostrar en Streamlit
st.title("Dry Matter Forage Maize yield  - 3D Interactive Analysis")
st.plotly_chart(fig)
