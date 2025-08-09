import os
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Point
from joblib import load
import pandas as pd
import ee
import geemap

# Inicializa Google Earth Engine
ee.Initialize(project='biosentinel-uv')

# Configuración general
TAXON = "amphibians"
MODEL_PATH = f"./model/BS-1.0/models/{TAXON}_model.pkl"
RESOLUTION = 0.01  # grados (aprox. 1 km)
REGION_NAME = "chiribiquete"
OUTPUT_DIR = "./model/BS-1.0/scripts/output"
DATA_DIR = "./cached_layers"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# BBox para el Parque Nacional Chiribiquete
bbox = {
  "min_lon": -74.728103,
  "max_lon": -71.025195,
  "min_lat": -0.001802,
  "max_lat": 2.911496,
};


NDVI_PATH = f"{DATA_DIR}/{REGION_NAME}_NDVI.tif"
LST_PATH = f"{DATA_DIR}/{REGION_NAME}_LST.tif"
DEM_PATH = f"{DATA_DIR}/{REGION_NAME}_DEM.tif"

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

def get_gee_layers():
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
        .multiply(0.02).subtract(273.15).rename("LST")  # Kelvin a °C

    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30") \
        .mosaic() \
        .select('DEM') \
        .rename('DEM')

    return ndvi, lst, dem

def generate_grid(bbox, resolution=0.01):
    lon_vals = np.arange(bbox["min_lon"], bbox["max_lon"], resolution)
    lat_vals = np.arange(bbox["min_lat"], bbox["max_lat"], resolution)
    return [(lon, lat) for lon in lon_vals for lat in lat_vals]

def extract_raster_values(points, raster_path):
    if not os.path.exists(raster_path):
        raise FileNotFoundError(f"{raster_path} no fue encontrado.")
    with rasterio.open(raster_path) as src:
        values = list(src.sample(points))
        return np.array(values).squeeze()

def build_geojson(points, predictions_df, output_path=f"{OUTPUT_DIR}/predictions.geojson"):
    gdf = gpd.GeoDataFrame(
        predictions_df,
        geometry=[Point(xy) for xy in points],
        crs="EPSG:4326"
    )
    gdf.to_file(output_path, driver="GeoJSON")
    print(f"📍 GeoJSON guardado en: {output_path}")
    return gdf

def main():
    print("🚀 Iniciando predicción de biodiversidad en Chiribiquete...")

    # Paso 1: descargar capas
    ndvi_img, lst_img, dem_img = get_gee_layers()
    download_layer_if_missing(NDVI_PATH, ndvi_img, "NDVI", bbox)
    download_layer_if_missing(LST_PATH, lst_img, "LST", bbox)
    download_layer_if_missing(DEM_PATH, dem_img, "DEM", bbox)

    # Paso 2: generar grilla
    points = generate_grid(bbox, resolution=RESOLUTION)

    # Paso 3: extraer variables
    ndvi = extract_raster_values(points, NDVI_PATH)
    lst = extract_raster_values(points, LST_PATH)
    dem = extract_raster_values(points, DEM_PATH)

    # Paso 4: construir DataFrame
    df = pd.DataFrame({
        "longitude": [pt[0] for pt in points],
        "latitude": [pt[1] for pt in points],
        "NDVI": ndvi,
        "LST_C": lst,
        "DEM": dem
    })

    # Paso 5: cargar modelo y hacer predicción
    print(f"🤖 Cargando modelo para {TAXON}...")
    model = load(MODEL_PATH)
    y_pred = model.predict(df[["NDVI", "LST_C", "DEM", "longitude", "latitude"]])
    df[["Biota_Overlap", "Rel_Occupancy", "Rel_Species_Richness"]] = y_pred

    # Paso 6: exportar GeoJSON
    build_geojson(points, df)

    print("✅ Proceso finalizado.")

if __name__ == "__main__":
    main()
