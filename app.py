# archivo: app.py
import streamlit as st
import pandas as pd
import plotly.express as px

# Leer el CSV con codificación adecuada
df = pd.read_csv("Synthetic_Maize_Predictions.csv", encoding="latin1")

# Asegurar tipo correcto
df["Condition"] = df["Condition"].astype(int)

# Mapeo de colores y etiquetas
color_labels = {1: "Favorable conds", -1: "Unfavorable conds", 0: "Intermediate conds"}
df["Condition_Label"] = df["Condition"].map(color_labels)

# --- INTERFAZ STREAMLIT ---

st.title("Dry Matter Forage Maize Yield - 3D Interactive Analysis")

# Selector para tamaño de punto
point_size = st.slider("Select marker size", min_value=1, max_value=10, value=3)

z_axis = "Predicted_kgDM/ha"
# Crear gráfico interactivo con opciones seleccionadas
fig = px.scatter_3d(
    df,
    x="GrowingSeason(day)",
    y="Radiation(Mj/m2day)",
    z=z_axis,
    color="Condition_Label",
    color_discrete_map={
        "Favorable conds": "green",
        "Unfavorable conds": "red",
        "Intermediate conds": "gray"
    },
    title=f"3D Scatter: Predicted Dry Matter kg/ha vs Growing Season and Radiation ",
    opacity=0.8
)

# Aplicar tamaño de puntos
fig.update_traces(marker=dict(size=point_size))

# Mostrar gráfico
st.plotly_chart(fig, use_container_width=False, width=1200, height=900)
