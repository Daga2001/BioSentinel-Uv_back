import os
import glob
import ee
import geemap
from datetime import datetime, timedelta
import base64
import numpy as np
import uuid
import rasterio
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import geopandas as gpd
import pandas as pd
from joblib import load
from shapely.geometry import Point
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from functools import lru_cache
# Lazy imports for heavy libraries
_torch = None
_transformers = None
_huggingface_hub = None
_segment_anything = None
_PIL = None

# Devuelve número de cores lógicos disponibles
num_workers = os.cpu_count()  

def _get_torch():
    global _torch
    if _torch is None:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
        _torch = {'torch': torch, 'nn': nn, 'Dataset': Dataset, 'DataLoader': DataLoader}
    return _torch

def _get_transformers():
    global _transformers
    if _transformers is None:
        from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, CLIPSegProcessor, CLIPSegForImageSegmentation
        _transformers = {'SegformerFeatureExtractor': SegformerFeatureExtractor, 'SegformerForSemanticSegmentation': SegformerForSemanticSegmentation, 'CLIPSegProcessor': CLIPSegProcessor, 'CLIPSegForImageSegmentation': CLIPSegForImageSegmentation}
    return _transformers

def _get_PIL():
    global _PIL
    if _PIL is None:
        from PIL import Image
        _PIL = Image
    return _PIL

def _get_sam():
    global _huggingface_hub, _segment_anything
    if _huggingface_hub is None or _segment_anything is None:
        from huggingface_hub import hf_hub_download
        from segment_anything import sam_model_registry, SamPredictor
        _huggingface_hub = hf_hub_download
        _segment_anything = {'sam_model_registry': sam_model_registry, 'SamPredictor': SamPredictor}
    return _huggingface_hub, _segment_anything

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
def descargar_imagen_desde_geojson(geojson_obj, res, model_name=None, output_dir="downloaded_images"):
    init_earth_engine()

    geometry = extract_geometry(geojson_obj)
    aoi = ee.Geometry(geometry)

    # Configuración
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=30*12) # 12 meses atrás

    # Sentinel-2 band names (L2A)
    all_bands = [
        "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A",
        "B9", "B11", "B12"
    ]

    # If model is k-means or mkanet, select all bands
    if model_name in ["k-means", "mkanet"]:
        bands_to_use = all_bands
    else:
        bands_to_use = ["B4", "B3", "B2"]  # RGB

    params = {
        "satellite_collection": "COPERNICUS/S2_SR_HARMONIZED",
        "satellite_name": "SENTINEL-2",
        "date_range": (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
        "scale": res,
        "image_format": "tif",
        "bands": bands_to_use,
        "file_per_band": model_name in ["k-means", "mkanet"]
    }

    # Borrar archivos previos
    os.makedirs(output_dir, exist_ok=True)
    for f in os.listdir(output_dir):
        file_path = os.path.join(output_dir, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

    collection = (
        ee.ImageCollection(params["satellite_collection"])
        .filterBounds(aoi)
        .filterDate(*params["date_range"])
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
    )

    mosaic = collection.mosaic()
    mosaic = mosaic.select(bands_to_use)

    # Exportar imágenes
    if params["file_per_band"]:
        # Exportar cada banda como archivo .tif separado (paralelo)
        def export_band(band):
            band_img = mosaic.select([band]).visualize(min=0, max=3000)
            filename = f"{band}_{params['satellite_name']}_mosaic_{band}.{params['image_format']}"
            output_path = os.path.join(output_dir, filename)

            # Verificar si la banda es demasiado grande para descargar
            try:
                _ = band_img.getDownloadURL({
                    "scale": params["scale"],
                    "region": aoi,
                    "format": "GEO_TIFF"
                })
            except Exception as e:
                error_msg = str(e)
                if "Total request size" in error_msg:
                    raise Exception(f"Band {band} too large for download: {error_msg}")
                else:
                    raise

            # Exportar imagen
            try:
                geemap.ee_export_image(
                    ee_object=band_img,
                    filename=output_path,
                    scale=params["scale"],
                    region=aoi,
                    file_per_band=False,
                )
            except Exception as e:
                error_msg = str(e)
                if "Total request size" in error_msg and "bytes) must be less than" in error_msg:
                    raise Exception(f"Band {band} too large for download: {error_msg}")
                else:
                    raise # vuelve a lanzar el error original
            return output_path
        
        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                band_paths = list(executor.map(export_band, bands_to_use))
            # Verify all files exist
            for path in band_paths:
                if not os.path.exists(path):
                    raise Exception(f"Band export completed but file not found: {path}")
        except Exception as e:
            raise Exception(f"Parallel band export failed: {str(e)}")
        return band_paths
    else:
        # Exportar imagen RGB combinada
        mosaic = mosaic.visualize(min=0, max=3000, bands=bands_to_use)
        try:
            image_date = ee.Date(collection.first().get("system:time_start")).format("YYYY-MM-dd").getInfo()
        except Exception as e:
            raise Exception(f"Error getting image date: {str(e)}")
        
        # Verificar si la imagen es demasiado grande para descargar
        try:
            _ = mosaic.getDownloadURL({
                "scale": params["scale"],
                "region": aoi,
                "format": "GEO_TIFF"
            })
        except Exception as e:
            error_msg = str(e)
            if "Total request size" in error_msg:
                raise Exception(f"Image too large for download: {error_msg}")
            else:
                raise
        
        band_suffix = "custom" if params["bands"] else "RGB"
        filename = f"{image_date}_{params['satellite_name']}_mosaic_{band_suffix}.{params['image_format']}"
        output_path = os.path.join(output_dir, filename)
        
        try:
            geemap.ee_export_image(
                ee_object=mosaic,
                filename=output_path,
                scale=params["scale"],
                region=aoi,
                file_per_band=False,
            )
            # Check if file was actually created
            if not os.path.exists(output_path):
                raise Exception(f"GEE export completed but file not found: {output_path}")
        except Exception as e:
            error_msg = str(e)
            if "Total request size" in error_msg and "bytes) must be less than" in error_msg:
                raise Exception(f"Image too large for download: {error_msg}")
            else:
                raise # vuelve a lanzar el error original
        
        return output_path

# Caché global para modelos
_segformer_cache = {}
_model_lock = threading.Lock()

@lru_cache(maxsize=1)
def get_segformer_model():
    transformers = _get_transformers()
    model_id = "nvidia/segformer-b0-finetuned-ade-512-512"
    feature_extractor = transformers['SegformerFeatureExtractor'].from_pretrained(model_id)
    model = transformers['SegformerForSemanticSegmentation'].from_pretrained(model_id)
    model.eval()
    return feature_extractor, model, model.config.id2label

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

def stack_band_tiffs_to_multiband(band_paths, output_path="downloaded_images/temp_multiband.tif"):
    band_arrays = []
    meta = None
    for band_file in band_paths:
        with rasterio.open(band_file) as src:
            band_arrays.append(src.read(1))
            if meta is None:
                meta = src.meta.copy()
    stack = np.stack(band_arrays)
    meta.update(count=len(band_arrays), dtype=stack.dtype)
    with rasterio.open(output_path, "w", **meta) as dst:
        for i in range(stack.shape[0]):
            dst.write(stack[i], i + 1)
    return output_path

# =========================================================
# ===== Segmentación con Segformer-B0 =====
# =========================================================

def segmentar_con_segformer_b0(image_path):
    # Lazy loading
    PIL_Image = _get_PIL()
    torch_lib = _get_torch()
    feature_extractor, model, class_labels = get_segformer_model()
    
    # 📥 Cargar imagen
    img = PIL_Image.open(image_path).convert("RGB")

    # 🔁 Preprocesamiento e inferencia
    inputs = feature_extractor(images=img, return_tensors="pt")
    with torch_lib['torch'].no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        upsampled = torch_lib['torch'].nn.functional.interpolate(
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
    rgb_image = PIL_Image.fromarray(rgb_array)

    # 💾 Guardar imagen segmentada
    output_dir = "/tmp/segmentaciones"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"segmentacion_{uuid.uuid4().hex[:8]}.png"
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
            "name": label,
            "color": hex_color,
            "count": int(counts[j])
        }

    return {
        "classifications": pixels_per_class,
        "image_path": output_path
    }

# =========================================================
# ===== Segmentación con CLIPSeg y SAM =====
# =========================================================

def segmentar_con_clipseg_sam(image_path):
    # Lazy loading
    PIL_Image = _get_PIL()
    torch_lib = _get_torch()
    transformers = _get_transformers()
    hf_hub_download, sam_lib = _get_sam()
    
    # Descargar checkpoint SAM si no existe
    checkpoint_path = hf_hub_download(
        repo_id="segments-arnaud/sam_vit_b",
        filename="sam_vit_b_01ec64.pth"
    )

    # Cargar imagen
    image = PIL_Image.open(image_path).convert("RGB")
    img_np = np.array(image)

    # Preparar modelo SAM
    device = "cuda" if torch_lib['torch'].cuda.is_available() else "cpu"
    sam = sam_lib['sam_model_registry']["vit_b"](checkpoint=checkpoint_path).to(device)
    predictor = sam_lib['SamPredictor'](sam)
    predictor.set_image(img_np)

    # Preparar modelo CLIPSeg
    processor = transformers['CLIPSegProcessor'].from_pretrained("CIDAS/clipseg-rd64-refined")
    clipseg = transformers['CLIPSegForImageSegmentation'].from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

    # Conceptos ambientales
    concepts = [
        "bosque primario", "vegetación secundaria", "áreas deforestadas",
        "agua", "infraestructura minera", "suelo expuesto",
        "zonas degradadas", "fragmentación ecológica"
    ]

    results = {}
    for concept in concepts:
        inputs = processor(text=concept, images=image, return_tensors="pt").to(device)
        with torch_lib['torch'].no_grad():
            outputs = clipseg(**inputs)
            preds = torch_lib['torch'].sigmoid(outputs.logits).cpu().numpy()[0]
        # Resize mask to match image shape
        mask = (preds > 0.5).astype(np.uint8)
        mask_img = PIL_Image.fromarray(mask * 255)
        mask_resized = mask_img.resize((img_np.shape[1], img_np.shape[0]), resample=PIL_Image.NEAREST)
        mask = np.array(mask_resized) // 255
        pixel_count = int(mask.sum())
        results[concept] = {"mask": mask, "pixels": pixel_count}

    # Crear imagen de máscaras combinadas (opcional: cada clase en un color)
    rgb_array = np.zeros((img_np.shape[0], img_np.shape[1], 3), dtype=np.uint8)
    palette = {i: tuple(np.random.randint(0, 255, 3)) for i in range(len(concepts))}
    for idx, concept in enumerate(concepts):
        rgb_array[results[concept]["mask"] == 1] = palette[idx]
    rgb_image = PIL_Image.fromarray(rgb_array)

    # Guardar imagen segmentada
    output_dir = "/tmp/segmentaciones"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"segmentacion_clipseg_sam_{uuid.uuid4().hex[:8]}.png"
    output_path = os.path.join(output_dir, output_filename)
    rgb_image.save(output_path)

    # Preparar clasificaciones
    pixels_per_class = {}
    for idx, concept in enumerate(concepts):
        color_rgb = palette[idx]
        hex_color = '#%02x%02x%02x' % color_rgb
        pixels_per_class[concept] = {
            "name": concept,
            "color": hex_color,
            "count": results[concept]["pixels"]
        }

    return {
        "classifications": pixels_per_class,
        "image_path": output_path
    }

# =========================================================
# ==== Función para calcular k óptimo ====
# =========================================================

# Global function for multiprocessing
_X_sample_global = None

def _compute_k_metrics(k):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10, init="k-means++").fit(_X_sample_global)
    return kmeans.inertia_, silhouette_score(_X_sample_global, kmeans.labels_)

def calculate_optimal_k(X, k_range=(2, 11), max_samples=10000):
    global _X_sample_global
    
    # Muestra para acelerar en datasets grandes
    if X.shape[0] > max_samples:
        idx = np.random.choice(X.shape[0], max_samples, replace=False)
        _X_sample_global = X[idx]
    else:
        _X_sample_global = X
    
    k_values = list(range(*k_range))
    
    # Función para encontrar Elbow por distancia al segmento
    def find_elbow_by_distance(ks, inertias):
        ks = np.array(ks)
        inertias = np.array(inertias)
        x1, y1 = ks[0], inertias[0]
        x2, y2 = ks[-1], inertias[-1]
        num = np.abs((y2 - y1) * ks - (x2 - x1) * inertias + x2 * y1 - y2 * x1)
        den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distances = num / den
        return int(ks[np.argmax(distances)])
    
    # Calcular métricas secuencialmente para evitar pickle issues
    inertias = []
    silhouettes = []
    for k in k_values:
        inertia, silhouette = _compute_k_metrics(k)
        inertias.append(inertia)
        silhouettes.append(silhouette)
    
    # Selección combinada
    k_sil = k_values[np.argmax(silhouettes)]
    k_elbow = find_elbow_by_distance(k_values, inertias)
    
    # Buscar mejor silhouette dentro de ±1 del elbow
    candidates = [k for k in k_values if abs(k - k_elbow) <= 1]
    sil_by_k = {k: silhouettes[i] for i, k in enumerate(k_values)}
    k_best_sil = max(candidates, key=lambda kk: sil_by_k[kk])
    
    # Umbral de mejora mínima en silhouette
    SIL_DIFF_THRESHOLD = 0.02
    if (sil_by_k[k_best_sil] - sil_by_k[k_elbow]) < SIL_DIFF_THRESHOLD:
        k_optimo = k_elbow
    else:
        k_optimo = k_best_sil
    
    print(f"Elbow sugiere k={k_elbow}, Silhouette sugiere k={k_sil}")
    print(f"➡ K óptimo seleccionado = {k_optimo}")
    
    return k_optimo

# =========================================================
# ==== Segmentación con K-means en imágenes multibanda ====
# =========================================================

def segmentar_con_kmeans(image_path, k=None):
    # Lazy loading
    PIL_Image = _get_PIL()
    
    # Leer el archivo TIFF multibanda
    with rasterio.open(image_path) as src:
        stack = src.read()  # shape: (bands, rows, cols)
    n_bands, h, w = stack.shape

    # Reorganizar para clustering: (n_pixels, n_bands)
    X = stack.reshape(n_bands, -1).T  # shape: (n_pixels, n_bands)

    # Calcular k óptimo si no se proporciona
    if k is None:
        k = calculate_optimal_k(X)

    # Clustering K-means
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    labels = kmeans.labels_.reshape(h, w)

    # Paleta de colores para cada clase
    np.random.seed(42)
    palette = {i: tuple(np.random.randint(0, 255, 3)) for i in range(k)}

    # Crear imagen RGB normalizada (usando bandas Sentinel-2 B04, B03, B02 si existen)
    def normalize_band(band):
        band_min, band_max = np.percentile(band, (2, 98))
        band_norm = np.clip((band - band_min) / (band_max - band_min), 0, 1)
        return (band_norm * 255).astype(np.uint8)

    # Selección de bandas (B04, B03, B02) si existen
    rgb_indices = [3, 2, 1] if n_bands >= 4 else [0, 0, 0]
    rgb = np.stack([normalize_band(stack[i]) for i in rgb_indices], axis=-1)

    # Crear imagen de clases
    rgb_classes = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in palette.items():
        rgb_classes[labels == class_id] = color
    rgb_image = PIL_Image.fromarray(rgb_classes)

    # Guardar imagen segmentada
    output_dir = "/tmp/segmentaciones"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"segmentacion_kmeans_{uuid.uuid4().hex[:8]}.png"
    output_path = os.path.join(output_dir, output_filename)
    rgb_image.save(output_path)

    # Conteo de píxeles por clase
    pixels_per_class = {}
    for class_id in range(k):
        color_rgb = palette[class_id]
        hex_color = '#%02x%02x%02x' % color_rgb
        n_pixels = int(np.sum(labels == class_id))
        pixels_per_class[f"Clase_{class_id}"] = {
            "name": f"Clase_{class_id}",
            "color": hex_color,
            "count": n_pixels
        }

    return {
        "classifications": pixels_per_class,
        "image_path": output_path
    }




class SegDataset:
    def __init__(self, stack, mask):
        self.stack = stack
        self.mask = mask
    def __len__(self): return self.mask.size
    def __getitem__(self, idx):
        torch_lib = _get_torch()
        h, w = divmod(idx, self.stack.shape[2])
        patch = self.stack[:, h, w]
        return torch_lib['torch'].tensor(patch, dtype=torch_lib['torch'].float32), torch_lib['torch'].tensor(self.mask[h, w], dtype=torch_lib['torch'].long)

def create_mkanet_lite(in_bands, n_classes=3):
    torch_lib = _get_torch()
    
    class MKANetLite(torch_lib['nn'].Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch_lib['nn'].Conv2d(in_bands, 16, kernel_size=3, padding=1)
            self.relu = torch_lib['nn'].ReLU()
            self.conv2 = torch_lib['nn'].Conv2d(16, n_classes, kernel_size=1)
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.conv2(x)
            return x
    
    return MKANetLite()

# ===========================================================================================
# mkanet para segmentación de imágenes multibanda
# ===========================================================================================

def segmentar_con_mkanet(image_path, epochs=3):
    # Lazy loading
    torch_lib = _get_torch()
    PIL_Image = _get_PIL()
    
    # Leer el archivo TIFF multibanda
    with rasterio.open(image_path) as src:
        stack = src.read()  # shape: (bands, rows, cols)
    n_bands, h, w = stack.shape

    # Usar K-means para obtener una máscara inicial (etiquetas por píxel)
    X = stack.reshape(n_bands, -1).T
    k = calculate_optimal_k(X)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    mask = kmeans.labels_.reshape(h, w)

    # Dataset y DataLoader optimizado
    dataset = SegDataset(stack, mask)
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.15, random_state=42)
    train_loader = torch_lib['DataLoader'](torch_lib['torch'].utils.data.Subset(dataset, train_idx), batch_size=512, shuffle=True, num_workers=num_workers)
    
    # Modelo
    model = create_mkanet_lite(in_bands=n_bands, n_classes=int(mask.max())+1)
    device = torch_lib['torch'].device("cuda" if torch_lib['torch'].cuda.is_available() else "cpu")
    model.to(device)

    criterion = torch_lib['nn'].CrossEntropyLoss()
    optimizer = torch_lib['torch'].optim.Adam(model.parameters(), lr=1e-3)

    # Entrenamiento optimizado
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.unsqueeze(-1).unsqueeze(-1).to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits.view(len(xb), -1), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    # Inferencia
    model.eval()
    with torch_lib['torch'].no_grad():
        xb_all = torch_lib['torch'].tensor(stack, dtype=torch_lib['torch'].float32).unsqueeze(0).to(device)
        preds = model(xb_all).argmax(1).cpu().squeeze().numpy()

    # Paleta de colores para cada clase
    np.random.seed(42)
    n_classes = int(mask.max())+1
    palette = {i: tuple(np.random.randint(0, 255, 3)) for i in range(n_classes)}

    # Crear imagen de clases
    rgb_classes = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in palette.items():
        rgb_classes[preds == class_id] = color
    rgb_image = PIL_Image.fromarray(rgb_classes)

    # Guardar imagen segmentada
    output_dir = "/tmp/segmentaciones"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"segmentacion_mkanet_{uuid.uuid4().hex[:8]}.png"
    output_path = os.path.join(output_dir, output_filename)
    rgb_image.save(output_path)

    # Conteo de píxeles por clase
    pixels_per_class = {}
    for class_id in range(n_classes):
        color_rgb = palette[class_id]
        hex_color = '#%02x%02x%02x' % color_rgb
        n_pixels = int(np.sum(preds == class_id))
        pixels_per_class[f"Clase_{class_id}"] = {
            "name": f"Clase_{class_id}",
            "color": hex_color,
            "count": n_pixels
        }

    return {
        "classifications": pixels_per_class,
        "image_path": output_path
    }

# =========================================================
# ===== Modelo de predicción de biodiversidad BS-1.0 =====
# =========================================================

def run_bs1_model(lon, lat, taxon, metric, radius_km=50):
    init_earth_engine()

    TAXON = taxon
    MODEL_PATH = f"./model/BS-1.0/models/{TAXON}_model.pkl"
    RESOLUTION = 0.01  # grados (aprox. 1 km)
    OUTPUT_DIR = "./model/BS-1.0/scripts/output"
    DATA_DIR = "./cached_layers"

    if metric == "richness":
        METRIC = "Rel_Species_Richness"
    elif metric == "overlap":
        METRIC = "Biota_Overlap"
    elif metric == "occupancy":
        METRIC = "Rel_Occupancy"
    else:
        raise ValueError("❌ Métrica no válida. Debe ser 'richness', 'overlap' o 'occupancy'.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    def get_bbox_from_point(lon, lat, radius_km=50):
        deg_radius = radius_km / 111
        return {
            "min_lon": lon - deg_radius,
            "max_lon": lon + deg_radius,
            "min_lat": lat - deg_radius,
            "max_lat": lat + deg_radius,
        }

    def download_layer_if_missing(path, gee_image, band, bbox, scale=1000):
        if os.path.exists(path):
            print(f"✅ {os.path.basename(path)} ya descargado.")
            return
        region = ee.Geometry.Rectangle([bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]])
        image = gee_image.select(band).clip(region)
        print(f"⬇️ Descargando {os.path.basename(path)} desde Google Earth Engine...")
        geemap.ee_export_image(
            image,
            filename=path,
            region=region,
            scale=scale,
            file_per_band=False,
            crs="EPSG:4326"
        )

    def get_gee_layers(bbox):
        region = ee.Geometry.Rectangle([bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]])
        ndvi = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterDate("2023-01-01", "2023-12-31") \
            .filterBounds(region) \
            .median() \
            .normalizedDifference(['B8', 'B4']).rename('NDVI')
        lst = ee.ImageCollection("MODIS/061/MOD11A2") \
            .filterDate("2023-01-01", "2023-12-31") \
            .select("LST_Day_1km") \
            .mean() \
            .multiply(0.02).subtract(273.15).rename("LST")
        dem = ee.ImageCollection("COPERNICUS/DEM/GLO30") \
            .mosaic() \
            .select('DEM') \
            .rename('DEM')
        return ndvi, lst, dem

    def generate_grid(bbox, resolution=RESOLUTION):
        lon_vals = np.arange(bbox["min_lon"], bbox["max_lon"], resolution)
        lat_vals = np.arange(bbox["min_lat"], bbox["max_lat"], resolution)
        return [(lon, lat) for lon in lon_vals for lat in lat_vals]

    def extract_raster_values(points, raster_path):
        if not os.path.exists(raster_path):
            raise FileNotFoundError(f"{raster_path} no fue encontrado.")
        with rasterio.open(raster_path) as src:
            values = list(src.sample(points))
            return np.array(values).squeeze()

    def build_geojson(points, predictions_df, output_path):
        gdf = gpd.GeoDataFrame(
            predictions_df,
            geometry=[Point(xy) for xy in points],
            crs="EPSG:4326"
        )
        gdf.to_file(output_path, driver="GeoJSON")
        print(f"📍 GeoJSON guardado en: {output_path}")
        return output_path

    # Main logic
    bbox = get_bbox_from_point(lon, lat, radius_km)
    region_name = f"loc_{lon}_{lat}_{radius_km}km".replace('.', '_').replace('-', 'm')
    ndvi_path = f"{DATA_DIR}/{region_name}_NDVI.tif"
    lst_path = f"{DATA_DIR}/{region_name}_LST.tif"
    dem_path = f"{DATA_DIR}/{region_name}_DEM.tif"
    output_geojson = f"{OUTPUT_DIR}/{region_name}_predictions.geojson"

    ndvi_img, lst_img, dem_img = get_gee_layers(bbox)
    download_layer_if_missing(ndvi_path, ndvi_img, "NDVI", bbox)
    download_layer_if_missing(lst_path, lst_img, "LST", bbox)
    download_layer_if_missing(dem_path, dem_img, "DEM", bbox)

    points = generate_grid(bbox)
    ndvi = extract_raster_values(points, ndvi_path)
    lst = extract_raster_values(points, lst_path)
    dem = extract_raster_values(points, dem_path)

    df = pd.DataFrame({
        "longitude": [pt[0] for pt in points],
        "latitude": [pt[1] for pt in points],
        "NDVI": ndvi,
        "LST_C": lst,
        "DEM": dem
    })

    model = load(MODEL_PATH)
    y_pred = model.predict(df[["NDVI", "LST_C", "DEM", "longitude", "latitude"]])
    
    # Handle single metric selection
    if y_pred.ndim == 1:
        df[METRIC] = y_pred
    else:
        # If model returns multiple columns, select the appropriate one
        metric_columns = ["Biota_Overlap", "Rel_Occupancy", "Rel_Species_Richness"]
        if METRIC in metric_columns:
            metric_index = metric_columns.index(METRIC)
            df[METRIC] = y_pred[:, metric_index]
        else:
            df[METRIC] = y_pred[:, 0]  # Default to first column

    geojson_path = build_geojson(points, df, output_geojson)
    return geojson_path