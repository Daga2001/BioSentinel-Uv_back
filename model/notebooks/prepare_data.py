import os
import pandas as pd
import json

from shapely.geometry import shape

# Carpeta de entrada
input_dir = "../../model/notebooks/EnrichedData/"
# Carpeta de salida
output_dir = "../../docs/training_data/"

# Asegurarse de que exista la carpeta de salida
os.makedirs(output_dir, exist_ok=True)

def extract_centroid(geojson_str):
    try:
        geom_dict = json.loads(geojson_str)
        polygon = shape(geom_dict)
        centroid = polygon.centroid
        return pd.Series({"longitude": centroid.x, "latitude": centroid.y})
    except Exception as e:
        print(f"❌ Error extrayendo centroide: {e}")
        return pd.Series({"longitude": None, "latitude": None})

# Recorrer todos los CSV en la carpeta
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        filepath = os.path.join(input_dir, filename)
        print(f"Procesando {filename}...")

        try:
            df = pd.read_csv(filepath)

            # Extraer coordenadas del centro del polígono
            if ".geo" in df.columns:
                centroid_df = df[".geo"].apply(extract_centroid)
                df = pd.concat([df, centroid_df], axis=1)

            # Convertir temperatura de Kelvin*0.02 a °C
            if "LST_Day_1km" in df.columns:
                df["LST_C"] = df["LST_Day_1km"] * 0.02 - 273.15
                df.drop(columns=["LST_Day_1km"], inplace=True)

            # Eliminar columnas irrelevantes
            df = df.drop(columns=["system:index", ".geo"], errors="ignore")

            # Eliminar filas con valores faltantes
            df.dropna(inplace=True)

            # Reordenar columnas: poner 'id' de primera si existe
            if "id" in df.columns:
                cols = ["id"] + [col for col in df.columns if col != "id"]
                df = df[cols]

            # Extraer nombre base del archivo sin extensión
            label = os.path.splitext(filename)[0].lower()

            # Guardar archivo limpio con extensión .csv
            output_path = os.path.join(output_dir, f"{label}.csv")
            df.to_csv(output_path, index=False)
            print(f"✅ Guardado: {output_path}")

        except Exception as e:
            print(f"❌ Error procesando {filename}: {e}")
