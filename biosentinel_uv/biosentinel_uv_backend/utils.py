import os
import glob
import ee
import geemap
from datetime import datetime
import base64
import numpy as np
import torch
from PIL import Image
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import os
import io
import uuid

# === Inicializa Earth Engine (solo una vez) ===
def init_earth_engine():
    project_id = 'biosentineluv'
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)
    print("✅ Earth Engine inicializado correctamente.")

# === Extrae la geometría desde un objeto GeoJSON ===
def extract_geometry(geojson_obj):
    if geojson_obj.get("type") == "FeatureCollection":
        return geojson_obj["features"][0]["geometry"]
    elif geojson_obj.get("type") == "Feature":
        return geojson_obj["geometry"]
    elif geojson_obj.get("type") in ["Polygon", "MultiPolygon", "Point", "LineString"]:
        return geojson_obj
    else:
        raise ValueError("❌ Estructura GeoJSON no reconocida.")

# === Descarga imagen satelital en base a un GeoJSON ===
def descargar_imagen_desde_geojson(geojson_obj, output_dir="downloaded_images"):
    init_earth_engine()

    geometry = extract_geometry(geojson_obj)
    aoi = ee.Geometry(geometry)

    # Configuración
    params = {
        "satellite_collection": "COPERNICUS/S2_SR",
        "satellite_name": "SENTINEL-2",
        "date_range": ("2022-01-01", "2023-01-31"),
        "scale": 50,
        "image_format": "tif",
        "bands": None,  # por defecto usa RGB
        "file_per_band": False
    }

    os.makedirs(output_dir, exist_ok=True)
    for f in glob.glob(os.path.join(output_dir, f"*.{params['image_format']}")):
        os.remove(f)

    collection = (
        ee.ImageCollection(params["satellite_collection"])
        .filterBounds(aoi)
        .filterDate(*params["date_range"])
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
    )

    mosaic = collection.mosaic()
    bands = params["bands"] if params["bands"] else ["B4", "B3", "B2"]
    mosaic = mosaic.select(bands).visualize(min=0, max=3000, bands=bands)

    # Obtener fecha de referencia
    image_date = ee.Date(collection.first().get("system:time_start")).format("YYYY-MM-dd").getInfo()
    band_suffix = "custom" if params["bands"] else "RGB"
    filename = f"{image_date}_{params['satellite_name']}_mosaic_{band_suffix}.{params['image_format']}"
    output_path = os.path.join(output_dir, filename)

    geemap.ee_export_image(
        ee_object=mosaic,
        filename=output_path,
        scale=params["scale"],
        region=aoi,
        file_per_band=params["file_per_band"],
    )

    return output_path

# Carga global (opcional para eficiencia)
model_id = "nvidia/segformer-b0-finetuned-ade-512-512"
feature_extractor = SegformerFeatureExtractor.from_pretrained(model_id)
model = SegformerForSemanticSegmentation.from_pretrained(model_id)
model.eval()

class_labels = model.config.id2label

# Diccionario CSS3 reducido
css3_colors_rgb = {
    'black': (0, 0, 0), 'white': (255, 255, 255), 'red': (255, 0, 0),
    'lime': (0, 255, 0), 'blue': (0, 0, 255), 'yellow': (255, 255, 0),
    'cyan': (0, 255, 255), 'magenta': (255, 0, 255), 'gray': (128, 128, 128),
    'maroon': (128, 0, 0), 'olive': (128, 128, 0), 'green': (0, 128, 0),
    'purple': (128, 0, 128), 'teal': (0, 128, 128), 'navy': (0, 0, 128),
    'orange': (255, 165, 0), 'pink': (255, 192, 203), 'brown': (165, 42, 42)
}

def closest_css3_color_name(rgb_val):
    r1, g1, b1 = rgb_val
    min_dist = float('inf')
    closest_name = None
    for name, (r2, g2, b2) in css3_colors_rgb.items():
        dist = (r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name

def segmentar_con_segformer_b0(image_path):
    import os
    import uuid
    from PIL import Image
    import numpy as np
    import torch

    # 📥 Cargar imagen
    img = Image.open(image_path).convert("RGB")

    # 🔁 Preprocesamiento e inferencia
    inputs = feature_extractor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        upsampled = torch.nn.functional.interpolate(
            logits, size=img.size[::-1], mode="bilinear", align_corners=False
        )
        predicted = upsampled.argmax(dim=1)[0].cpu().numpy()

    # 🎨 Generar paleta aleatoria
    unique_classes = np.unique(predicted)
    np.random.seed(42)
    palette = {i: tuple(np.random.randint(0, 255, 3)) for i in unique_classes}

    # 🖼️ Convertir a imagen RGB directamente (sin modo 'P')
    rgb_array = np.zeros((predicted.shape[0], predicted.shape[1], 3), dtype=np.uint8)
    for class_id, color in palette.items():
        rgb_array[predicted == class_id] = color
    rgb_image = Image.fromarray(rgb_array)

    # 💾 Guardar imagen segmentada
    output_dir = "/tmp/segmentaciones"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"segmentacion_{uuid.uuid4().hex[:8]}.tif"
    output_path = os.path.join(output_dir, output_filename)
    rgb_image.save(output_path)

    # 📊 Conteo de píxeles por clase
    unique, counts = np.unique(predicted, return_counts=True)
    pixels_per_class = {}
    for j, i in enumerate(unique):
        label = class_labels.get(int(i), f"Clase_{i}")
        color_rgb = palette[int(i)]
        hex_color = '#%02x%02x%02x' % color_rgb
        pixels_per_class[label] = {
            "color": hex_color,
            "count": int(counts[j])
        }

    return {
        "classifications": pixels_per_class,
        "image_path": output_path
    }
