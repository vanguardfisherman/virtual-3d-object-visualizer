¡Perfecto! Aquí tienes un **README listo para GitHub**. Cópialo tal cual en `README.md`. Abajo también te dejo un `requirements.txt` y un `.gitignore` sugeridos.

---

# Visualizador 3D · Streamlit + Plotly + Trimesh

App web en **Python** para visualizar modelos 3D (**OBJ/MTL/GLB/STL/PLY**), respetar colores/texturas de **MagicaVoxel**, activar **modo Exhibición** (rotación + levitación) y crear un **catálogo** de miniaturas animables.

## Demo

* Ejecuta `streamlit run app.py` y abre `http://localhost:8501`.
* Sube tu **.obj + .mtl** (y texturas si aplica).
* Activa **Exhibición** y usa el botón ▶ en la figura.
* Presiona **“Agregar a la colección”** para crear tu catálogo de “trofeos”.

> Tip MagicaVoxel: exporta `OBJ` junto a su `MTL` y las imágenes de textura; súbelos juntos.

## Características

* Carga de **OBJ/MTL** con colores por vértice/cara y texturas (se hornean a color).
* **Exhibición** configurable: eje de rotación (X/Y/Z), sentido (horario/antihorario), eje de levitación, velocidad, FPS.
* **Catálogo** en grid: tarjetas compactas siempre en modo exhibición.
* Descarga **PNG** de la vista (requiere `kaleido`).
* Normaliza malla (centra y escala).

## Requisitos

* Python 3.10+
* Paquetes:

  ```bash
  pip install -U pip
  pip install -r requirements.txt
  ```

`requirements.txt` (pon este archivo en la raíz):

```
streamlit
plotly
trimesh
numpy
kaleido
```

`.gitignore` (recomendado):

```
venv/
__pycache__/
*.pyc
.vscode/
.env
.DS_Store
```

## Ejecución

```bash
# 1) (opcional) entorno virtual
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

# 2) dependencias
pip install -r requirements.txt

# 3) correr
streamlit run app.py
```

## Uso rápido

1. **Sube el modelo**: selecciona **OBJ + MTL** y texturas (PNG/JPG) juntos.
2. **Exhibición**: activa y ajusta ejes, sentido, velocidad, FPS y levitación.
3. **Colección**: “Agregar a la colección” → abajo verás tus miniaturas en grid.
4. **PNG**: descarga una captura de la figura principal desde el panel “Opcional”.
