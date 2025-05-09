# archivo: app.py
'''
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
    #title=f"3D Scatter: Predicted Dry Matter kg/ha vs Growing Season and Radiation ",
    opacity=0.8
)

# Aplicar tamaño de puntos
fig.update_traces(marker=dict(size=point_size))

# Mostrar gráfico
st.plotly_chart(fig, use_container_width=False, width=1200, height=2400)
'''
'''
import streamlit as st
import pandas as pd
import plotly.express as px

# Configurar página en modo ancho
st.set_page_config(layout="wide", page_title="Maize Yield 3D Analysis")

# Leer CSV
df = pd.read_csv("Synthetic_Maize_Predictions.csv", encoding="latin1")
df["Condition"] = df["Condition"].astype(int)

# Etiquetas de condición
color_labels = {1: "Favorable conds", -1: "Unfavorable conds", 0: "Intermediate conds"}
df["Condition_Label"] = df["Condition"].map(color_labels)

# Título principal
st.title("🌽 Dry Matter Forage Maize Yield - 3D Interactive Analysis")

# Layout: columna izquierda para controles, derecha para gráfico
col1, col2 = st.columns([1, 4])

with col1:
    st.subheader("⚙️ Controls")
    point_size = st.slider("Select marker size", min_value=1, max_value=10, value=3)

with col2:
    # Gráfico 3D
    fig = px.scatter_3d(
        df,
        x="GrowingSeason(day)",
        y="Radiation(Mj/m2day)",
        z="Predicted_kgDM/ha",
        color="Condition_Label",
        color_discrete_map={
            "Favorable conds": "green",
            "Unfavorable conds": "red",
            "Intermediate conds": "gray"
        },
        title="3D Scatter: Growing Season vs Radiation vs Predicted kgDM/ha",
        opacity=0.8
    )
    fig.update_traces(marker=dict(size=point_size))
    st.plotly_chart(fig, use_container_width=True, height=1400)
    '''
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

