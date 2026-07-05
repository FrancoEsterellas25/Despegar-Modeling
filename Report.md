# Reporte Inicial: Modelado de Precios de Alojamiento (Rio de Janeiro)

## 1. Resumen Ejecutivo
Este proyecto tiene como objetivo predecir el precio por noche por persona (`price_by_night_person`) de alojamientos en Rio de Janeiro. Para ello, se ha desarrollado un pipeline de Machine Learning escalable e interpretable que permite al negocio entender los factores que impulsan el precio, utilizando datos enriquecidos (características del hotel, amenities, ubicación, estacionalidad, etc.).

## 2. Arquitecturas de Modelado
Se implementaron y compararon dos enfoques avanzados de ensamble:

1. **Stacking Ensemble:**
   - **Nivel 0 (Base):** Modelos basados en árboles (XGBoost, LightGBM, RandomForest).
   - **Nivel 1 (Meta-modelo):** Regresión Lineal Ordinaria (OLS). Combina las predicciones base para optimizar el error general sin pérdida de linealidad.

2. **Mixture of Experts (MoE):**
   - **Enrutador:** Reglas heurísticas de negocio (basadas en estrellas, cantidad de habitaciones y amenities) que segmentan el inventario en tres grupos: *Lujo/Resort*, *Boutique/Informal* y *Estándar*.
   - **Expertos:** Modelos XGBoost independientes y especializados (Finetuneados en el espacio logarítmico del target) para cada segmento de negocio.

## 3. Explicabilidad y Confianza (SHAP)
Para evitar arquitecturas de "caja negra", se implementó el cálculo de **valores SHAP**. Esto permite auditar las decisiones algorítmicas al nivel de observaciones individuales. Gracias al aislamiento de la lógica de procesamiento (pipeline unificado en cache Parquet), el sistema provee justificaciones instantáneas a nivel unitario, dividiendo el precio pronosticado en contribuciones positivas y negativas por variable (ej. impacto de la distancia a la playa o calificación del alojamiento).

## 4. Próximos Pasos
- Validar las métricas de negocio (RMSE, MAE, R²) directamente con el equipo de Pricing.
- Refinar hiperparámetros si se introducen nuevas variables (ej. clima local o macroeconomía).
- Desplegar el modelo ganador en una API para integración real-time.
