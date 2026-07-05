import os
import joblib
import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicTransformer

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "src", "data")
    processed_dir = os.path.join(data_dir, "processed")
    models_dir = os.path.join(base_dir, "models")
    
    # Cargar datos procesados
    train_path = os.path.join(processed_dir, "train_data_clean.csv")
    val_path = os.path.join(processed_dir, "val_data_clean.csv")
    
    print("Cargando datasets...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # ELEGIBILIDAD DE FEATURES PARA REGRESIÓN SIMBÓLICA (¿Por qué estas?):
    # Seleccionamos variables continuas clave que representan efectos geográficos (distancias, densidad)
    # y condiciones de búsqueda (anticipación, posición de listado). 
    # El valor del suelo y las tarifas en Río tienen caídas exponenciales y relaciones no lineales complejas 
    # con respecto a las playas y centros turísticos. Symbolic Regression permite descubrir estas interacciones 
    # de forma analítica (ej. decaimientos exponenciales, ratios de densidad/distancia).
    symbolic_input_features = [
        "dist_playa_m", 
        "dist_cristo_m", 
        "dist_metro_m", 
        "density_kde", 
        "anticipation", 
        "position", 
        "total_amenities"
    ]
    
    print(f"Variables seleccionadas para Regresión Simbólica: {symbolic_input_features}")
    
    # Imputar cualquier nulo que haya quedado en estas features antes de pasarlas a gplearn
    # (gplearn no acepta NaNs nativamente)
    medians = train_df[symbolic_input_features].median()
    
    # Tomamos un submuestreo de entrenamiento para acelerar el algoritmo genético (GP es intensivo)
    sample_size = 50000
    if len(train_df) > sample_size:
        train_sample = train_df.sample(n=sample_size, random_state=42)
    else:
        train_sample = train_df
        
    X_train_sub = train_sample[symbolic_input_features].fillna(medians)
    y_train_sub = train_sample["price_by_night_person"]
    
    print("Iniciando evolución genética con SymbolicTransformer...")
    
    # Configuramos el transformador simbólico
    function_set = ["add", "sub", "mul", "div", "log", "sqrt", "sin", "cos"]
    
    transformer = SymbolicTransformer(
        generations=15,             # Número de iteraciones evolutivas
        population_size=1000,       # Cantidad de fórmulas individuales por generación
        hall_of_fame=100,
        n_components=3,             # Queremos generar exactamente las 3 mejores nuevas features simbólicas
        function_set=function_set,
        parsimony_coefficient=0.001, # Penaliza fórmulas extremadamente complejas para evitar sobreajuste
        max_samples=0.9,
        random_state=42,
        n_jobs=-1
    )
    
    # Ajustar transformador
    transformer.fit(X_train_sub, y_train_sub)
    
    print("\nFórmulas matemáticas simbólicas descubiertas:")
    for idx, program in enumerate(transformer._best_programs):
        print(f"  * Feature Simbólica #{idx + 1}: {program}")
        
    # Transformar datasets completos (Train y Val)
    print("\nAplicando transformación a los datasets completos...")
    train_sym_features = transformer.transform(train_df[symbolic_input_features].fillna(medians))
    val_sym_features = transformer.transform(val_df[symbolic_input_features].fillna(medians))
    
    # Convertir a DataFrame
    sym_cols = [f"symbolic_feat_{i+1}" for i in range(3)]
    train_sym_df = pd.DataFrame(train_sym_features, columns=sym_cols, index=train_df.index)
    val_sym_df = pd.DataFrame(val_sym_features, columns=sym_cols, index=val_df.index)
    
    # Forzar conversión a float32 para detectar desbordamientos tempranos y acotar rango
    train_sym_df = train_sym_df.astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0)
    val_sym_df = val_sym_df.astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Limitar valores extremos para que no superen límites de precisión de XGBoost
    train_sym_df = train_sym_df.clip(lower=-1e9, upper=1e9)
    val_sym_df = val_sym_df.clip(lower=-1e9, upper=1e9)
    
    # Concatenar con los datasets originales
    train_augmented = pd.concat([train_df, train_sym_df], axis=1)
    val_augmented = pd.concat([val_df, val_sym_df], axis=1)
    
    # Guardar datasets aumentados
    train_augmented_path = os.path.join(processed_dir, "train_data_symbolic.csv")
    val_augmented_path = os.path.join(processed_dir, "val_data_symbolic.csv")
    
    train_augmented.to_csv(train_augmented_path, index=False)
    val_augmented.to_csv(val_augmented_path, index=False)
    
    # Guardar modelo del transformador
    transformer_model_path = os.path.join(models_dir, "symbolic_transformer.pkl")
    joblib.dump(transformer, transformer_model_path)
    
    print(f"\nProceso finalizado con éxito:")
    print(f"  -> Dataset Train aumentado guardado en: {train_augmented_path}")
    print(f"  -> Dataset Validation aumentado guardado en: {val_augmented_path}")
    print(f"  -> Transformador simbólico guardado en: {transformer_model_path}")

if __name__ == "__main__":
    main()
