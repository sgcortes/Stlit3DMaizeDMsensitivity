import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Configurar la página
st.set_page_config(layout="wide", page_title="Maize Yield Surface Fit")

# Inicializar la cámara si no existe en el estado
if "camera" not in st.session_state:
    st.session_state.camera = dict(
        eye=dict(x=1.5, y=-1.5, z=1.0)  # Vista desde el frente inferior
    )

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
    poly_degree = st.slider("Polynomial degree", min_value=2, max_value=6, value=4)
    show_surface = st.checkbox("Show fitted surface", value=True)

    # Leyenda visual
    st.markdown("### 🟢 Legend")
    legend_html = f"""
    <div style="line-height: 1.6;">
        <span style="font-size:16px;">
            <span style="display:inline-block; width:{point_size * 2}px; height:{point_size * 2}px;
                  background-color:red; border-radius:50%; margin-right:10px;"></span>
            <span>Unfavorable conds</span><br>
            <span style="display:inline-block; width:{point_size * 2}px; height:{point_size * 2}px;
                  background-color:green; border-radius:50%; margin-right:10px;"></span>
            <span>Favorable conds</span><br>
            <span style="display:inline-block; width:{point_size * 2}px; height:{point_size * 2}px;
                  background-color:gray; border-radius:50%; margin-right:10px;"></span>
            <span>Intermediate conds</span>
        </span>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)

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
        # Superficie polinómica (grado variable)
        X_poly = np.column_stack((x, y))
        poly = PolynomialFeatures(degree=poly_degree)
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
        title=f"3D Scatter + Polynomial Surface (degree {poly_degree})",
        scene=dict(
            xaxis_title='Growing Season (days)',
            yaxis_title='Radiation (MJ/m2/day)',
            zaxis_title='Predicted kgDM/ha'
        ),
        scene_camera=st.session_state.camera,  # mantener la vista de cámara
        width=1100,
        height=800
    )

    # Mostrar gráfico
    st.plotly_chart(fig, use_container_width=True)

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
    poly_degree = st.slider("Polynomial degree", min_value=2, max_value=6, value=4)
    show_surface = st.checkbox("Show fitted surface", value=True)

    # Leyenda visual
    st.markdown("### 🟢 Legend")
    legend_html = f"""
    <div style="line-height: 1.6;">
        <span style="font-size:16px;">
            <span style="display:inline-block; width:{point_size * 2}px; height:{point_size * 2}px;
                  background-color:red; border-radius:50%; margin-right:10px;"></span>
            <span>Unfavorable conds</span><br>
            <span style="display:inline-block; width:{point_size * 2}px; height:{point_size * 2}px;
                  background-color:green; border-radius:50%; margin-right:10px;"></span>
            <span>Favorable conds</span><br>
            <span style="display:inline-block; width:{point_size * 2}px; height:{point_size * 2}px;
                  background-color:gray; border-radius:50%; margin-right:10px;"></span>
            <span>Intermediate conds</span>
        </span>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)

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
        # Superficie polinómica (grado variable)
        X_poly = np.column_stack((x, y))
        poly = PolynomialFeatures(degree=poly_degree)
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
        title=f"3D Scatter + Polynomial Surface (degree {poly_degree})",
        scene=dict(
            xaxis_title='Growing Season (days)',
            yaxis_title='Radiation (MJ/m2/day)',
            zaxis_title='Predicted kgDM/ha'
        ),
        width=1100,
        height=800
    )
    '''

    st.plotly_chart(fig, use_container_width=True)

