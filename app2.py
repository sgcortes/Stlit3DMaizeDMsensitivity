# archivo: app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Configurar la página
st.set_page_config(layout="wide", page_title="Maize Yield Surface Fit")

# Leer los datos
df = pd.read_csv("Synthetic_Maize_Predictions.csv", encoding="latin1")
df["Condition"] = df["Condition"].astype(int)

# Mapeo de colores para condiciones
color_map = {1: "green", -1: "red", 0: "gray"}
df["Color"] = df["Condition"].map(color_map)

# Interfaz de control
st.title("🌽 Maize Yield Prediction - 3D Surface Fit & Scatter")
col1, col2 = st.columns([1, 5])

with col1:
    st.subheader("⚙️ Controls")
    point_size = st.slider("Select marker size", 1, 10, 4)
    show_surface = st.checkbox("Show fitted surface", value=True)

with col2:
    # Datos base
    x = df["GrowingSeason(day)"].values
    y = df["Radiation(Mj/m2day)"].values
    z = df["Predicted_kgDM/ha"].values

    # Crear puntos del scatter
    scatter = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=point_size, color=df["Color"]),
        name='Synthetic Samples'
    )

    data = [scatter]

    # Si el usuario quiere mostrar la superficie
    if show_surface:
        # Superficie polinómica (grado 4)
        X_poly = np.column_stack((x, y))
        poly = PolynomialFeatures(degree=4)
        X_poly_transformed = poly.fit_transform(X_poly)

        model = LinearRegression()
        model.fit(X_poly_transformed, z)

        # Crear malla para la superficie
        xi = np.linspace(x.min(), x.max(), 60)
        yi = np.linspace(y.min(), y.max(), 60)
        xi, yi = np.meshgrid(xi, yi)

        mesh_points = np.column_stack((xi.ravel(), yi.ravel()))
        mesh_poly = poly.transform(mesh_points)
        zi = model.predict(mesh_poly).reshape(xi.shape)

        surface = go.Surface(x=xi, y=yi, z=zi, colorscale='Viridis', opacity=0.7, name="Fitted Surface")
        data.insert(0, surface)

    # Crear figura
    fig = go.Figure(data=data)
    fig.update_layout(
        title="3D Scatter + Polynomial Surface (degree 4)",
        scene=dict(
            xaxis_title='Growing Season (days)',
            yaxis_title='Radiation (MJ/m2/day)',
            zaxis_title='Predicted kgDM/ha'
        ),
        width=1100,
        height=800
    )

    st.plotly_chart(fig, use_container_width=True)

