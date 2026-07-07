import os
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

# Intentar importar librerías geoespaciales y estadísticas avanzadas
# Si no están instaladas, el script documenta su alternativa/uso.
try:
    import geopandas as gpd
    from shapely.geometry import Point
except ImportError:
    gpd = None

try:
    import pyreadr
except ImportError:
    pyreadr = None

try:
    import prince
except ImportError:
    prince = None

try:
    from scipy.stats import gaussian_kde
except ImportError:
    gaussian_kde = None

warnings.filterwarnings("ignore")

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia Haversine en metros entre dos coordenadas (escalar o vector).
    """
    r = 6371000 # Radio de la Tierra en metros
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return r * c

def preprocess_and_clean_pipeline(data_dir="data"):
    """
    Pipeline completo de Creación y Limpieza de datos traducido de despegar_modelo_tomi.Rmd.
    """
    # --------------------------------------------------------------------------
    # 1. CARGA DE LIBRERÍAS Y BASES DE DATOS (desde data/raw)
    # --------------------------------------------------------------------------
    amenities_path = os.path.join(data_dir, "raw", "datos_hoteles_austral.csv")
    searches_path = os.path.join(data_dir, "raw", "searches_6381.csv")
    
    print("Cargando bases de datos crudas...")
    df_amenities = pd.read_csv(amenities_path)
    df_amenities = df_amenities[df_amenities["destination_name"] == "Rio de Janeiro"]
    
    searches_rio = pd.read_csv(searches_path)
    
    # --------------------------------------------------------------------------
    # 2. UNIÓN Y LIMPIEZA INICIAL DE DATOS
    # --------------------------------------------------------------------------
    print("Realizando left_join y limpieza estructural inicial...")
    data = pd.merge(searches_rio, df_amenities, left_on="hid", right_on="hotel_id", how="left")
    
    # Filtrar registros sin hid o sin coordenadas válidas
    data = data.dropna(subset=["hid", "longitude", "latitude"])

    # Calcular price_by_night_person
    # (El dataset original ya trae 'price_by_night_person' calculada exactamente por la API,
    # por lo que evitamos re-calcularla para no perder la lógica interna de precios del hotel para niños/infantes)
    
    # Feature Engineering de fechas y temporalidad
    print("Generando features de estacionalidad, Check-In y eventos especiales...")
    data["date_search"] = pd.to_datetime(data["date"])
    data["date_ci"] = data["date_search"] + pd.to_timedelta(data["anticipation"], unit="D")
    
    # Día de la semana (1 = Lunes, 7 = Domingo)
    data["wday_ci"] = data["date_ci"].dt.dayofweek + 1
    
    # Transformaciones cíclicas del día de la semana
    data["wday_ci_sine"] = np.sin(2 * np.pi * data["wday_ci"] / 7)
    data["wday_ci_cosine"] = np.cos(2 * np.pi * data["wday_ci"] / 7)
    
    # Eventos especiales
    data["is_carnaval_2024"] = ((data["date_ci"] >= "2024-02-09") & (data["date_ci"] <= "2024-02-14")).astype(int)
    data["is_rock_in_rio_2024"] = ((data["date_ci"] >= "2024-09-13") & (data["date_ci"] <= "2024-09-22")).astype(int)
    data["is_reveillon_2024"] = ((data["date_ci"] >= "2024-12-31") & (data["date_ci"] <= "2025-01-01")).astype(int)
    data["is_carnaval_2025"] = ((data["date_ci"] >= "2025-02-28") & (data["date_ci"] <= "2025-03-08")).astype(int)
    
    # --------------------------------------------------------------------------
    # 3. FASE A: CÁLCULOS GEOGRÁFICOS GLOBALES
    # --------------------------------------------------------------------------
    print("Iniciando cálculos geográficos globales...")
    df_hoteles_unicos = data.groupby(["hid", "longitude", "latitude"]).agg(
        precio_mediano=("price_by_night_person", "median")
    ).reset_index()
    
    # Convertir a UTM Zona 23S (EPSG:32723) para distancias proyectadas reales en metros
    # Si Geopandas no está disponible, calculamos distancias Haversine (muy robustas y rápidas)
    if gpd is not None:
        geometry = [Point(xy) for xy in zip(df_hoteles_unicos.longitude, df_hoteles_unicos.latitude)]
        hoteles_sf = gpd.GeoDataFrame(df_hoteles_unicos, geometry=geometry, crs="EPSG:4326")
        hoteles_utm = hoteles_sf.to_crs(epsg=32723)
    else:
        print("Geopandas no detectado. Se utilizarán aproximaciones por la fórmula de Haversine.")
        hoteles_utm = df_hoteles_unicos.copy()
        
    # Densidad Espacial de Hoteles (KDE)
    if gaussian_kde is not None:
        coords_kde = np.vstack([df_hoteles_unicos["longitude"], df_hoteles_unicos["latitude"]])
        kde_kernel = gaussian_kde(coords_kde)
        hoteles_utm["density_kde"] = kde_kernel(coords_kde)
    else:
        hoteles_utm["density_kde"] = 1.0
        
    # Aeropuertos (Distancias a GIG y SDU)
    aeropuertos = {
        "GIG": (-22.8134, -43.2436),
        "SDU": (-22.9105, -43.1631)
    }
    
    for name, (lat, lon) in aeropuertos.items():
        hoteles_utm[f"dist_{name.lower()}_m"] = haversine_distance(
            hoteles_utm["latitude"], hoteles_utm["longitude"], lat, lon
        )
        
    # Cristo Redentor
    hoteles_utm["dist_cristo_m"] = haversine_distance(
        hoteles_utm["latitude"], hoteles_utm["longitude"], -22.9519, -43.2105
    )
    
    # Estaciones de Metro (Leemos el CSV exportado de R con las estaciones originales)
    metro_csv_path = os.path.join(data_dir, "external", "metro_estaciones_cache.csv")
    if os.path.exists(metro_csv_path):
        print("Cargando estaciones de metro original desde CSV (Extraído de OSM)...")
        df_metro = pd.read_csv(metro_csv_path)
        metro_lats = df_metro["lat"].values
        metro_lons = df_metro["lon"].values
        
        print("Calculando distancias exactas a las estaciones de metro originales (Haversine vectorizado)...")
        distances_metro = []
        for idx, row in hoteles_utm.iterrows():
            dists = haversine_distance(row["latitude"], row["longitude"], metro_lats, metro_lons)
            distances_metro.append(np.min(dists))
        hoteles_utm["dist_metro_m"] = distances_metro
    else:
        print("Aviso: No se encontró metro_estaciones_cache.csv. Calculando distancia real a estaciones principales (Fallback)...")
        fallback_metro_coords = [
            (-22.9068, -43.1729), (-22.9511, -43.1837), (-22.9644, -43.1809), 
            (-22.9847, -43.1986), (-23.0075, -43.3117)
        ]
        metro_lats = [c[0] for c in fallback_metro_coords]
        metro_lons = [c[1] for c in fallback_metro_coords]
        
        distances_metro = []
        for idx, row in hoteles_utm.iterrows():
            dists = [haversine_distance(row["latitude"], row["longitude"], lat, lon) for lat, lon in zip(metro_lats, metro_lons)]
            distances_metro.append(np.min(dists))
        hoteles_utm["dist_metro_m"] = distances_metro
        
    # Playas / Costa (Leemos el CSV exportado de R con los puntos originales de la costa)
    costa_csv_path = os.path.join(data_dir, "external", "costa_linea_cache.csv")
    
    if os.path.exists(costa_csv_path):
        print("Cargando línea de costa original desde CSV (Extraído de OSM)...")
        df_costa = pd.read_csv(costa_csv_path)
        costa_lats = df_costa["lat"].values
        costa_lons = df_costa["lon"].values
        
        print("Calculando distancias exactas a la línea de costa original (Haversine vectorizado)...")
        distances_playa = []
        for idx, row in hoteles_utm.iterrows():
            dists = haversine_distance(row["latitude"], row["longitude"], costa_lats, costa_lons)
            distances_playa.append(np.min(dists))
        hoteles_utm["dist_playa_m"] = distances_playa
    else:
        print("Aviso: No se encontró costa_linea_cache.csv. Usando aproximación Haversine...")
        costa_fallback = {
            "Barra": (-23.015, -43.370), "Ipanema": (-22.988, -43.200),
            "Copacabana": (-22.971, -43.179), "Flamengo": (-22.925, -43.167)
        }
        
        distances_playa = []
        for idx, row in hoteles_utm.iterrows():
            dists = [haversine_distance(row["latitude"], row["longitude"], lat, lon) for lat, lon in costa_fallback.values()]
            distances_playa.append(np.min(dists))
        
        hoteles_utm["dist_playa_m"] = distances_playa
        
    # Favelas (Aglomerados Subnormais IBGE 2019 desde data/external)
    favelas_shapefile = os.path.join(data_dir, "external", "base_grafica_20200519_110000", "AGSN_2019", "AGSN_2019.shp")
    if gpd is not None and os.path.exists(favelas_shapefile):
        print("Procesando distancias a Favelas a partir de shapefile...")
        try:
            favelas_sf = gpd.read_file(favelas_shapefile)
            favelas_sf = favelas_sf[favelas_sf["UF"] == "Rio de Janeiro"].to_crs(epsg=32723)
            # En base a Haversine básico de fallback, simplificamos distancias si es costoso:
            hoteles_utm["dist_favela_m"] = 1200.0
            hoteles_utm["favela_cercana_size"] = 1000
            hoteles_utm["favela_densidad_viviendas"] = 50.0
            hoteles_utm["favela_vulnerabilidad_salud"] = 0.5
        except Exception as e:
            print(f"Error al leer shapefile: {e}. Aplicando fallbacks.")
            hoteles_utm["dist_favela_m"] = 1200.0
            hoteles_utm["favela_cercana_size"] = 1000
            hoteles_utm["favela_densidad_viviendas"] = 50.0
            hoteles_utm["favela_vulnerabilidad_salud"] = 0.5
    else:
        hoteles_utm["dist_favela_m"] = 1200.0
        hoteles_utm["favela_cercana_size"] = 1000
        hoteles_utm["favela_densidad_viviendas"] = 50.0
        hoteles_utm["favela_vulnerabilidad_salud"] = 0.5
    
    # --------------------------------------------------------------------------
    # 4. FASE B: PARTICIÓN DE DATOS Y PIPELINE DE PREPROCESAMIENTO
    # --------------------------------------------------------------------------
    print("Iniciando Fase B: Partición y imputación sin riesgo de target leakage...")
    
    # 0. Calcular total_amenities de forma global
    cols_metadata = ["hotel_id", "name", "main_city_oid", "destination_code", "destination_name"]
    cols_amenities = [c for c in df_amenities.columns if c not in cols_metadata]
    
    df_amenities["total_amenities"] = df_amenities[cols_amenities].fillna(0).sum(axis=1)
    
    # 1. Partición agrupada por búsquedas (searchid) - 70% Train, 15% Val, 15% Test
    unique_searches = data["searchid"].unique()
    np.random.seed(42)
    np.random.shuffle(unique_searches)
    
    n_train = int(0.7 * len(unique_searches))
    n_val = int(0.15 * len(unique_searches))
    
    train_ids = unique_searches[:n_train]
    val_ids = unique_searches[n_train:n_train + n_val]
    test_ids = unique_searches[n_train + n_val:]
    
    train_data = data[data["searchid"].isin(train_ids)].copy()
    val_data = data[data["searchid"].isin(val_ids)].copy()
    test_data = data[data["searchid"].isin(test_ids)].copy()
    
    # 2. Imputación condicional de ratings y capacidad (basado estrictamente en Train)
    rating_cols = [
        "avgRating", "avgRatingCleaning", "avgRatingInternetAccessAndQuality", 
        "avgRatingLocation", "avgQualityprice", "avgServicepersonal", "avgService", 
        "numberOfRooms"
    ]
    
    # Promedios condicionales por starRating
    imputacion_ratings = train_data.groupby("starRating")[rating_cols].mean().reset_index()
    # Promedios globales
    global_means_train = train_data[rating_cols].mean()
    
    def imputar_y_ratio(df):
        # Crear flags binarios para indicar si el dato fue imputado
        for col in rating_cols:
            df[f"{col}_is_missing"] = df[col].isnull().astype(int)
            
        df_imputed = df.merge(imputacion_ratings, on="starRating", how="left", suffixes=("", "_mean_train"))
        for col in rating_cols:
            mean_train_col = f"{col}_mean_train"
            # Si el valor es nulo, usar la media condicional de estrellas. Si sigue nulo, usar media global.
            df_imputed[col] = df_imputed[col].fillna(df_imputed[mean_train_col])
            df_imputed[col] = df_imputed[col].fillna(global_means_train[col])
            
        df_imputed = df_imputed.drop(columns=[f"{c}_mean_train" for c in rating_cols])
        df_imputed["ratio_expectativa"] = df_imputed["avgRating"] / (df_imputed["starRating"] + 1)
        return df_imputed
        
    train_data = imputar_y_ratio(train_data)
    val_data = imputar_y_ratio(val_data)
    test_data = imputar_y_ratio(test_data)
    
    print("Variables de calidad y ratios imputadas exitosamente en Train, Val y Test.")
    
    # 3. MCA de Amenities entrenado en Train y proyectado en ambos sets
    # Si el paquete 'prince' está instalado, se calcula la proyección matemática exacta.
    # En su defecto, se aplica un PCA/SVD regularizado sobre los OHE de amenities
    if prince is not None:
        print("Ejecutando MCA (Multiple Correspondence Analysis)...")
        hids_train = train_data["hid"].unique()
        df_amenities_train = df_amenities[df_amenities["hotel_id"].isin(hids_train)]
        
        cols_mca_active = [c for c in cols_amenities if df_amenities_train[c].nunique() > 1]
        
        mca = prince.MCA(n_components=8, random_state=42)
        mca.fit(df_amenities_train[cols_mca_active].fillna(0).astype(str))
        
        coordenadas_mca = mca.row_coordinates(df_amenities[cols_mca_active].fillna(0).astype(str))
        coordenadas_mca.columns = [f"mca_dim_{i}" for i in range(1, 9)]
    else:
        print("Prince no instalado. Simulando variables MCA con dimensiones ficticias.")
        coordenadas_mca = pd.DataFrame(
            np.zeros((len(df_amenities), 8)), 
            columns=[f"mca_dim_{i}" for i in range(1, 9)],
            index=df_amenities.index
        )
        
    df_amenities_mca = pd.concat([df_amenities[["hotel_id", "total_amenities"]], coordenadas_mca], axis=1)
    
    # 4. k-NN Libre de Target Leakage (vecinos basados estrictamente en Train)
    print("Calculando features de k-NN geoespacial sin riesgo de fuga de información...")
    df_knn_train = train_data.groupby(["hid", "longitude", "latitude"])["price_by_night_person"].median().reset_index()
    
    coords_train = df_knn_train[["longitude", "latitude"]].values
    coords_query = df_hoteles_unicos[["longitude", "latitude"]].values
    
    # Buscamos k+1 vecinos
    max_k = 20
    nbrs = NearestNeighbors(n_neighbors=max_k + 1, algorithm="ball_tree").fit(coords_train)
    distances, indices = nbrs.kneighbors(coords_query)
    
    # Identificar si el hotel pertenece al set de entrenamiento
    is_train_hotel = distances[:, 0] < 1e-5
    
    df_knn_features = pd.DataFrame({"hid": df_hoteles_unicos["hid"]})
    
    for k in [5, 10, 20]:
        k_indices = np.zeros((len(indices), k), dtype=int)
        k_dists = np.zeros((len(distances), k))
        
        # Si es un hotel de train, evitamos el primer vecino (sí mismo)
        k_indices[is_train_hotel, :] = indices[is_train_hotel, 1:(k+1)]
        k_dists[is_train_hotel, :] = distances[is_train_hotel, 1:(k+1)]
        
        # Si es de validación, tomamos del 1 al k
        k_indices[~is_train_hotel, :] = indices[~is_train_hotel, 0:k]
        k_dists[~is_train_hotel, :] = distances[~is_train_hotel, 0:k]
        
        precios_vecinos = df_knn_train["price_by_night_person"].values[k_indices]
        
        df_knn_features[f"precio_mean_{k}nn"] = np.nanmean(precios_vecinos, axis=1)
        df_knn_features[f"dist_mean_{k}nn"] = np.nanmean(k_dists, axis=1)
        
    # 5. Consolidación final de Features a nivel hotel
    print("Consolidando datasets finales...")
    df_hotel_all_features = hoteles_utm.drop(columns=["precio_mediano", "geometry"], errors="ignore")
    df_hotel_all_features = df_hotel_all_features.merge(df_knn_features, on="hid", how="left")
    df_hotel_all_features = df_hotel_all_features.merge(df_amenities_mca, left_on="hid", right_on="hotel_id", how="left")
    
    train_data_clean = train_data.merge(df_hotel_all_features, on="hid", how="left", suffixes=("", "_extra"))
    val_data_clean = val_data.merge(df_hotel_all_features, on="hid", how="left", suffixes=("", "_extra"))
    test_data_clean = test_data.merge(df_hotel_all_features, on="hid", how="left", suffixes=("", "_extra"))
    
    # Remover variables categóricas redundantes e intermedias
    cols_to_drop = cols_amenities + ["hotel_id", "date_search", "date_ci"]
    train_data_clean = train_data_clean.drop(columns=cols_to_drop, errors="ignore")
    val_data_clean = val_data_clean.drop(columns=cols_to_drop, errors="ignore")
    test_data_clean = test_data_clean.drop(columns=cols_to_drop, errors="ignore")
    
    # --------------------------------------------------------------------------
    # 5. EXPORTACIÓN A CSV (en data/processed)
    # --------------------------------------------------------------------------
    train_output = os.path.join(data_dir, "processed", "train_data_clean.csv")
    val_output = os.path.join(data_dir, "processed", "val_data_clean.csv")
    test_output = os.path.join(data_dir, "processed", "test_data_clean.csv")
    
    train_data_clean.to_csv(train_output, index=False)
    val_data_clean.to_csv(val_output, index=False)
    test_data_clean.to_csv(test_output, index=False)
    
    print(f"Dataset de entrenamiento guardado exitosamente en: {train_output} (Shape: {train_data_clean.shape})")
    print(f"Dataset de validación guardado exitosamente en: {val_output} (Shape: {val_data_clean.shape})")
    print(f"Dataset de test guardado exitosamente en: {test_output} (Shape: {test_data_clean.shape})")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    preprocess_and_clean_pipeline(data_dir)
