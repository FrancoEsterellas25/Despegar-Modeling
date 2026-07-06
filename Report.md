# Reporte Inicial: Modelado de Precios de Alojamiento (Rio de Janeiro)

## 1. Resumen Ejecutivo
Este proyecto tiene como objetivo predecir el precio por noche por persona (`price_by_night_person`) de alojamientos en Rio de Janeiro. Para ello, se ha desarrollado un pipeline de Machine Learning escalable e interpretable que permite al negocio entender los factores que impulsan el precio, utilizando datos enriquecidos (características del hotel, amenities, ubicación, estacionalidad, etc.).

Al mismo tiempo, probamos dos estrategías de modelado distintas que, a su vez, para el mismo conjunto deciden hacer uso diferentes del conjunto de datos otorgado, dividen en distintos subsets de datos, y entrenan modelos bajo distintos enfoques de agrupamientos. 

## 2. Arquitecturas de Modelado
Se implementaron y compararon dos enfoques avanzados de ensamble, un Stacking de modelos de Boosting (Xgboost y LightGBM), junto a un modelo de Bagging (Random Forest):

1. **Stacking Ensemble:**
   - **Nivel 0 (Base):** Modelos basados en árboles (XGBoost, LightGBM, RandomForest).
   - **Analisís de Correlacion de Predicciones Out of Sample**: Entrenamos modelos y mediante CV (Cross-Validation), predecimos un subset de los datos de entrenamiento, con esas predicciones (para los 3 modelos base), calculamos la correlacion entre los mismos. Esperamos baja correlación entre las predicciones de los modelos, para no tener modelos que, básicamente, estén diciendo lo mismo. 
   - **Nivel 1 (Meta-modelo):** Regresión Lineal Ordinaria (OLS). Combina las predicciones base para optimizar el error general sin pérdida de linealidad. Además de recibir las primeras 2 componentes principales del PCA de los datos de entrenamiento, a forma de contextualizar al modelo, sin romper el principio de que el meta-modelo solo aprenda combinaciones lineales de las predicciones de los modelos base.

2. **Mixture of Experts (MoE):**
   - **Enrutador:** Reglas heurísticas de negocio (basadas en estrellas, cantidad de habitaciones y amenities) que segmentan el inventario en tres grupos: *Lujo/Resort*, *Boutique/Informal* y *Estándar*. 
   - **Expertos:** Modelos XGBoost independientes y especializados (Finetuneados en el espacio logarítmico del target) para cada segmento de negocio.

## 3. Explicabilidad y Confianza (SHAP)
Para evitar arquitecturas de "caja negra", se implementó el cálculo de **valores SHAP**. Esto permite auditar las decisiones algorítmicas al nivel de observaciones individuales. La ventaja que nos provee es entender como cada enfoque de modelado premia a ciertas características sobre otras, además de tener una cantidad de cuanto aporta esa caracteristica con ese valor a la predicción final.

*(Observación: Los valores SHAP del modelo de Stacking, son los promedios ponderados de los valores SHAP de los modelos base, ponderadas por su coeficiente en el modelo de Regresión Lineal.)*

## 4. Métricas de Evaluación
Para auditar de forma justa arquitecturas tan complejas, los modelos se evalúan utilizando las siguientes funciones de pérdida:
- **RMSE (Root Mean Squared Error):** Penaliza de manera cuadrática los errores más grandes (outliers).
- **n-RMSE (Normalized RMSE):** Mide la dispersión del error de forma adimensional ajustándolo a la volatilidad real del mercado (RMSE / Rango de precios).
- **MAE (Mean Absolute Error):** Error promedio en valor absoluto, muy robusto a valores atípicos (desvío promedio en dólares).
- **p-MAE (Percentage MAE):** Mide el error porcentual global dividiendo el MAE por el Precio Medio de la población.
- **R² (Coeficiente de Determinación):** Proporción de la varianza del precio real explicada por el modelo.

## 5. Conclusión y Próximos Pasos
**Conclusión:** Se ha comprobado empíricamente que **ambos enfoques (MoE y Stacking) son fructíferos y competitivos**. Dependiendo del caso individual o segmento de mercado, un enfoque predice puntualmente mejor que el otro, pero en términos agregados ambos ostentan una performance muy similar y altamente satisfactoria para el negocio.

**Próximos Pasos:**
- Agregar modelos basados en otras filosofías al modelado de Stacking (por ejemplo: KNN, basado en distancias)
- Al Mixture of Experts, realizar un sistema objetivo de enrutador de observaciones.
