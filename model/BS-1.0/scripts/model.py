import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

# Rutas
input_dir = "/BioSentinel/docs/training_data/"
output_model_dir = "/BioSentinel/model/BS-1.0/models/"


os.makedirs(output_model_dir, exist_ok=True)

# Columnas
features = ["NDVI", "LST_C", "DEM", "longitude", "latitude"]
targets = ["Biota_Overlap", "Rel_Occupancy", "Rel_Species_Richness"]

# Recorrer cada CSV
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        taxa_name = filename.replace(".csv", "")
        print(f"📦 Entrenando modelo para: {taxa_name}")

        try:
            # Leer datos
            df = pd.read_csv(os.path.join(input_dir, filename))

            # Separar X e y
            X = df[features]
            y = df[targets]

            # Entrenar
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Evaluar
            score = r2_score(y_test, model.predict(X_test))
            print(f"   ✅ R² score: {score:.2f}")

            # Guardar modelo
            model_path = os.path.join(output_model_dir, f"{taxa_name}_model.pkl")
            joblib.dump(model, model_path)
            print(f"   💾 Modelo guardado: {model_path}\n")

        except Exception as e:
            print(f"❌ Error entrenando {taxa_name}: {e}")
