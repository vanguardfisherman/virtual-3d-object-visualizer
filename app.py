import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Visualizador 3D", layout="wide")
st.title("Visualizador 3D (demo con un cubo)")

# Cubo unitario centrado
verts = np.array([
    [-0.5, -0.5, -0.5],
    [ 0.5, -0.5, -0.5],
    [ 0.5,  0.5, -0.5],
    [-0.5,  0.5, -0.5],
    [-0.5, -0.5,  0.5],
    [ 0.5, -0.5,  0.5],
    [ 0.5,  0.5,  0.5],
    [-0.5,  0.5,  0.5],
])

# Caras trianguladas (i, j, k) índices en verts
faces = np.array([
    [0,1,2], [0,2,3],  # cara z-
    [4,5,6], [4,6,7],  # cara z+
    [0,1,5], [0,5,4],  # cara y-
    [2,3,7], [2,7,6],  # cara y+
    [1,2,6], [1,6,5],  # cara x+
    [0,3,7], [0,7,4],  # cara x-
])

i, j, k = faces[:,0], faces[:,1], faces[:,2]
x, y, z = verts[:,0], verts[:,1], verts[:,2]

mesh = go.Mesh3d(
    x=x, y=y, z=z,
    i=i, j=j, k=k,
    opacity=1.0,
    color="lightblue",
    lighting=dict(ambient=0.4, diffuse=0.9, specular=0.2),
    flatshading=True
)

fig = go.Figure(data=[mesh])
fig.update_layout(
    scene=dict(
        aspectmode="data",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
    ),
    margin=dict(l=0, r=0, t=0, b=0),
)

st.plotly_chart(fig, use_container_width=True)
