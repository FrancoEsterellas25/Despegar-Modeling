import os
from typing import List

# Directorios principales
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SRC_DIR = os.path.join(BASE_DIR, "src")

# Rutas de los datasets (CSV / Parquet)
# Para máxima performance, si existe CSV pesado en src/data/processed, usaremos Polars para cargarlo eficientemente
TRAIN_DATA_PATH = os.path.join(SRC_DIR, "data", "processed", "train_data_clean.csv")
VAL_DATA_PATH = os.path.join(SRC_DIR, "data", "processed", "val_data_clean.csv")

# Rutas de Modelos Originales y ONNX
STACKING_MODEL_PATH = os.path.join(MODELS_DIR, "stacking_model_best.joblib")
STACKING_ONNX_PATH = os.path.join(DATA_DIR, "stacking_model_best.onnx")

# Rutas de Modelos de Mixture of Experts (MoE) - Colega
MOE_DIR = os.path.join(BASE_DIR, "Shiny_Despliegue_Final-master")
MOE_MODELS_DIR = os.path.join(MOE_DIR, "models")

# Variables explicativas clave
FEATURES: List[str] = [
    "position", "adults", "children", "infants", "duration", "anticipation",
    "starRating", "avgRating", "ratio_expectativa",
    "avgRatingCleaning", "avgRatingInternetAccessAndQuality", "avgRatingLocation",
    "avgQualityprice", "avgServicepersonal", "avgService",
    "numberOfRooms", "total_amenities",
    "wday_ci_sine", "wday_ci_cosine",
    "is_carnaval_2024", "is_rock_in_rio_2024", "is_reveillon_2024", "is_carnaval_2025",
    "dist_gig_m", "dist_sdu_m", "dist_metro_m", "dist_favela_m",
    "favela_cercana_size", "favela_densidad_viviendas", "favela_vulnerabilidad_salud",
    "dist_cristo_m", "dist_playa_m", "density_kde",
    "dist_mean_5nn", "dist_mean_10nn", "dist_mean_20nn"
]

TARGET_COL: str = "price_by_night_person"
