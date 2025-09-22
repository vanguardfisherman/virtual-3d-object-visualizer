import io
import pathlib
import tempfile
from typing import List, Optional, Tuple, Any

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
import trimesh


# ---------- Config básica ----------
st.set_page_config(page_title="Visualizador 3D", layout="wide")
st.title("Visualizador 3D (OBJ/MTL con colores + Exhibición + Colección)")

# Estado para colección
if "collection" not in st.session_state:
    st.session_state.collection = []  # lista de dicts: {"name": str, "mesh": trimesh.Trimesh}


# ---------- Sidebar: controles ----------
with st.sidebar:
    st.header("Opciones de render")
    use_model_colors = st.checkbox("Usar colores del modelo (vertex/face/texture)", True)
    color = st.color_picker("Color fijo si no hay colores", "#87CEFA")
    opacity = st.slider("Opacidad", 0.1, 1.0, 1.0, 0.05)
    flat = st.checkbox("Flat shading", True)
    show_edges = st.checkbox("Mostrar aristas (wireframe)", True)
    show_axes = st.checkbox("Mostrar ejes y grilla", True)
    st.divider()
    st.subheader("Exhibición")
    axis = st.selectbox("Eje de rotación", ["Y", "X", "Z"], index=0)
    direction = st.selectbox("Sentido", ["Horario", "Antihorario"], index=0)
    bob_axis = st.selectbox("Eje de levitación", ["Y", "Z", "X"], index=0)
    exhibit = st.checkbox("Activar modo Exhibición")
    speed = st.slider("Velocidad (°/s)", 5, 60, 15)           # grados por segundo
    bob_ampl = st.slider("Levitación (amplitud)", 0.0, 0.5, 0.1, 0.05)
    fps = st.slider("FPS animación", 5, 60, 20)
    st.caption("Activa 'Exhibición' y usa el ▶ en cada gráfica para reproducir la animación.")
    st.divider()
    st.caption("Sube tu OBJ junto con su MTL y texturas si aplica.")

files = st.file_uploader(
    "Sube archivos 3D (puedes seleccionar varios)",
    type=["obj", "mtl", "stl", "glb", "ply", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)


# ---------- Utilidades de color ----------
def rgba_to_plotly_tuple(c: np.ndarray) -> Tuple[int, int, int, int]:
    if c.shape[-1] == 4:
        r, g, b, a = [int(v) for v in c[:4]]
    else:
        r, g, b = [int(v) for v in c[:3]]
        a = 255
    return (r, g, b, a)

def facecolor_array_to_strings(face_colors: np.ndarray) -> List[str]:
    out = []
    for c in face_colors:
        r, g, b, a = rgba_to_plotly_tuple(c)
        out.append(f"rgba({r},{g},{b},{a/255:.3f})")
    return out


# ---------- Carga robusta (múltiples archivos) ----------
def load_mesh_from_uploads(uploaded_list: List[Any]) -> trimesh.Trimesh:
    """
    Escribe todos los archivos subidos a un directorio temporal y carga el modelo principal.
    Permite que OBJ resuelva su MTL y texturas.
    """
    if not uploaded_list:
        return trimesh.creation.box(extents=[1, 1, 1])

    tmpdir = tempfile.mkdtemp(prefix="st3d_")

    model_path: Optional[pathlib.Path] = None
    model_ext_priority = {".obj": 0, ".glb": 1, ".stl": 2, ".ply": 3}

    for uf in uploaded_list:
        name = pathlib.Path(uf.name).name
        dest = pathlib.Path(tmpdir) / name
        with open(dest, "wb") as f:
            f.write(uf.read())
        ext = dest.suffix.lower()
        if ext in model_ext_priority:
            if model_path is None:
                model_path = dest
            else:
                if model_ext_priority[ext] < model_ext_priority[model_path.suffix.lower()]:
                    model_path = dest

    if model_path is None:
        return trimesh.creation.box(extents=[1, 1, 1])

    geom = trimesh.load(str(model_path), force='mesh')

    if isinstance(geom, trimesh.Scene):
        if len(geom.geometry) == 0:
            return trimesh.creation.box(extents=[1, 1, 1])
        meshes = [m for m in geom.geometry.values()]
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = geom

    return mesh


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.centroid)
    scale = 1.0 / (np.max(mesh.extents) + 1e-9)
    mesh.apply_scale(scale)
    return mesh


def build_base_trace(mesh: trimesh.Trimesh,
                     use_model_colors: bool,
                     fallback_color: str,
                     opacity: float,
                     flat: bool) -> Tuple[go.Mesh3d, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[List[str]]]:
    # Triangulación
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        mesh = mesh.subdivide()

    verts = mesh.vertices.copy()
    faces = mesh.faces.copy()
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]

    # Colores
    mesh_vis = mesh.visual
    vertexcolor = None
    facecolor = None

    if use_model_colors:
        try:
            if hasattr(mesh_vis, "uv") and mesh_vis.uv is not None and hasattr(mesh_vis, "material") and getattr(mesh_vis.material, "image", None) is not None:
                mesh = mesh.copy()
                mesh.visual = mesh.visual.to_color()
                mesh_vis = mesh.visual
                verts = mesh.vertices.copy()
                faces = mesh.faces.copy()
                i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
                x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]

            if hasattr(mesh_vis, "vertex_colors") and mesh_vis.vertex_colors is not None and len(mesh_vis.vertex_colors) == len(verts):
                vc = np.asarray(mesh_vis.vertex_colors)
                vertexcolor = vc[:, :3] if vc.shape[1] >= 3 else vc
            elif hasattr(mesh_vis, "face_colors") and mesh_vis.face_colors is not None and len(mesh_vis.face_colors) == len(faces):
                fc = np.asarray(mesh_vis.face_colors)
                facecolor = facecolor_array_to_strings(fc)
        except Exception:
            vertexcolor = None
            facecolor = None

    kwargs = dict(x=x, y=y, z=z, i=i, j=j, k=k, opacity=opacity, flatshading=flat)
    if vertexcolor is not None:
        kwargs["vertexcolor"] = vertexcolor
    elif facecolor is not None:
        kwargs["facecolor"] = facecolor
    else:
        kwargs["color"] = fallback_color

    mesh3d = go.Mesh3d(**kwargs, lighting=dict(ambient=0.4, diffuse=0.9, specular=0.2))
    return mesh3d, verts, faces, vertexcolor, facecolor


def add_wireframe(fig: go.Figure, verts: np.ndarray, faces: np.ndarray) -> None:
    edges = set()
    for tri in faces:
        e = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for a, b in e:
            edges.add(tuple(sorted((int(a), int(b)))))
    ex, ey, ez = [], [], []
    for a, b in edges:
        ex += [verts[a, 0], verts[b, 0], None]
        ey += [verts[a, 1], verts[b, 1], None]
        ez += [verts[a, 2], verts[b, 2], None]
    fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", line=dict(width=2)))


def rotate_and_bob(verts: np.ndarray, angle_rad: float, bob: float, axis: str, bob_axis: str) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)

    if axis.upper() == "X":
        R = np.array([[1.0, 0.0, 0.0],
                      [0.0,   c, -s ],
                      [0.0,   s,  c ]], dtype=float)
    elif axis.upper() == "Z":
        R = np.array([[  c, -s, 0.0],
                      [  s,  c, 0.0],
                      [0.0, 0.0, 1.0]], dtype=float)
    else:  # "Y"
        R = np.array([[  c, 0.0,  s ],
                      [0.0, 1.0, 0.0],
                      [ -s, 0.0,  c ]], dtype=float)

    v = verts @ R.T

    if bob_axis.upper() == "X":
        v[:, 0] += bob * np.sin(angle_rad)
    elif bob_axis.upper() == "Z":
        v[:, 2] += bob * np.sin(angle_rad)
    else:  # "Y"
        v[:, 1] += bob * np.sin(angle_rad)

    return v


def build_figure(mesh: trimesh.Trimesh,
                 use_model_colors: bool,
                 fallback_color: str,
                 opacity: float,
                 flat: bool,
                 show_edges: bool,
                 show_axes: bool,
                 exhibit: bool,
                 speed_dps: int,
                 bob_amplitude: float,
                 fps: int,
                 axis: str,
                 direction: str,
                 bob_axis: str) -> go.Figure:
    base_trace, base_verts, faces, vertexcolor, facecolor = build_base_trace(
        mesh, use_model_colors, fallback_color, opacity, flat
    )

    fig = go.Figure(data=[base_trace])
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=show_axes, showgrid=show_axes, zeroline=show_axes),
            yaxis=dict(visible=show_axes, showgrid=show_axes, zeroline=show_axes),
            zaxis=dict(visible=show_axes, showgrid=show_axes, zeroline=show_axes),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        uirevision="keep",
    )

    if show_edges:
        add_wireframe(fig, base_verts, faces)

    if not exhibit:
        return fig

    seconds_per_rev = max(1.0, 360.0 / float(speed_dps))
    total_frames = int(fps * seconds_per_rev)
    total_frames = max(30, min(total_frames, 600))

    sign = -1.0 if direction == "Horario" else 1.0
    angles = sign * np.linspace(0.0, 2.0 * np.pi, total_frames, endpoint=False)

    frames = []
    names = []

    for idx, theta in enumerate(angles):
        v = rotate_and_bob(base_verts, theta, bob_amplitude, axis=axis, bob_axis=bob_axis)
        frame_name = f"f{idx}"
        names.append(frame_name)
        frames.append(
            go.Frame(
                name=frame_name,
                data=[go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2])]
            )
        )

    fig.frames = frames

    frame_ms = int(1000 / max(1, fps))
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=True,
            buttons=[
                dict(
                    label="▶ Exhibición",
                    method="animate",
                    args=[names, dict(
                        frame=dict(duration=frame_ms, redraw=True),
                        transition=dict(duration=0),
                        fromcurrent=True,
                        mode="immediate"
                    )]
                ),
                dict(
                    label="⏸️ Pausa",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        transition=dict(duration=0),
                        mode="immediate"
                    )]
                ),
            ],
            x=0.02, y=0.02, xanchor="left", yanchor="bottom"
        )]
    )

    return fig


# ---------- Lógica principal (modelo actual) ----------
if not files:
    st.info("Sube tu OBJ junto con su MTL (y texturas si aplica). Mientras tanto, mostramos un cubo.")
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    current_name = "Cubo de prueba"
else:
    try:
        mesh = load_mesh_from_uploads(files)
        current_name = next((f.name for f in files if f.name.lower().endswith((".obj", ".glb", ".stl", ".ply"))), "Modelo")
    except Exception as e:
        st.error(f"No pude leer el modelo: {e}")
        st.stop()

mesh = normalize_mesh(mesh)

fig = build_figure(
    mesh,
    use_model_colors=use_model_colors,
    fallback_color=color,
    opacity=opacity,
    flat=flat,
    show_edges=show_edges,
    show_axes=show_axes,
    exhibit=exhibit,
    speed_dps=speed,
    bob_amplitude=bob_ampl,
    fps=fps,
    axis=axis,
    direction=direction,
    bob_axis=bob_axis
)
st.plotly_chart(fig, use_container_width=True)

# Botón: Agregar a la colección
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("➕ Agregar a la colección", type="primary"):
        st.session_state.collection.append({
            "name": current_name,
            "mesh": mesh.copy()  # guardamos la malla ya normalizada
        })
        st.success(f"Agregado a la colección: {current_name}")

with col2:
    if st.button("🗑️ Vaciar colección"):
        st.session_state.collection.clear()
        st.warning("Colección vaciada.")


# ---------- Colección ----------
# ---------- Colección (catálogo en grid) ----------
st.subheader("🗂️ Colección")

THUMB_COLS = 3       # cuántas tarjetas por fila (ajústalo a 2, 3, 4...)
THUMB_HEIGHT = 320   # alto del lienzo Plotly por tarjeta (px)
GAP = "medium"       # "small" | "medium" | "large"

if not st.session_state.collection:
    st.info("Tu colección está vacía. Agrega modelos con el botón de arriba.")
else:
    # Recorre la colección en filas de THUMB_COLS columnas
    for start in range(0, len(st.session_state.collection), THUMB_COLS):
        cols = st.columns(THUMB_COLS, gap=GAP)
        row_items = st.session_state.collection[start:start + THUMB_COLS]

        for col, item in zip(cols, row_items):
            with col:
                # Encabezado de la tarjeta
                st.markdown(f"**{item['name']}**")

                # Figura en modo exhibición SIEMPRE (miniatura)
                fig_i = build_figure(
                    item["mesh"],
                    use_model_colors=use_model_colors,
                    fallback_color=color,
                    opacity=opacity,
                    flat=flat,
                    show_edges=False,     # mini más limpio
                    show_axes=False,      # mini sin ejes
                    exhibit=True,         # SIEMPRE en exhibición
                    speed_dps=speed,      # puedes bajar velocidad aquí si quieres (p. ej. 10)
                    bob_amplitude=bob_ampl * 0.8,  # levitación un poco menor en mini
                    fps=fps,
                    axis=axis,
                    direction=direction,
                    bob_axis=bob_axis
                )
                fig_i.update_layout(height=THUMB_HEIGHT)

                # key única por tarjeta
                key = f"collection_plot_{start}_{item['name']}"
                st.plotly_chart(fig_i, use_container_width=True, key=key)



# Exportar PNG (requiere kaleido) de la figura principal
with st.expander("Opcional: Descargar captura PNG de la figura principal"):
    try:
        png_bytes = pio.to_image(fig, format="png", scale=2)
        st.download_button(
            "Descargar captura PNG",
            data=png_bytes,
            file_name="modelo.png",
            mime="image/png",
        )
    except Exception as e:
        st.caption(f"No se pudo generar PNG (¿kaleido instalado?): {e}")
