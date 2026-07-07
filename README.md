# Despegar Modeling: Predicción de Precios de Hoteles en Rio de Janeiro

Este repositorio contiene un pipeline avanzado de Machine Learning diseñado para predecir el precio por noche por persona (`price_by_night_person`) de la oferta hotelera en Rio de Janeiro. 

Se implementó una arquitectura de **Stacking Dinámico con Enrutamiento Orientado al Target**, que combina modelos especialistas entrenados con distintas funciones de pérdida (Tweedie y MAE) segmentados mediante un árbol de decisión espacial.

## 🏆 Métricas Destacadas (Resultados Finales)

Tras evaluar la arquitectura sobre el 100% de la varianza del mundo real (sin eliminación de outliers), los dos modelos que conforman la Frontera de Pareto del proyecto son:

- 🥇 **El Escudo de Varianza (Tweedie Puro + RIDGE):**
  - **$R^2$:** 0.6766
  - **MAE:** 22.87 USD
  - *Ideal para reportes financieros macro y Revenue Management por su altísima resistencia a la varianza y capacidad explicativa.*

- 🥇 **El Francotirador (Tweedie & MAE + HUBER):**
  - **MAE:** 19.42 USD
  - **$R^2$:** 0.6131
  - *Ideal para inferencias precisas de cara al usuario en el Front-End, logrando un error en dólares mínimo.*

---
📉 **Nota de Justificación Arquitectónica (Baseline):** 
Para dimensionar el valor de esta arquitectura, se entrenó un modelo de **Regresión Lineal Simple (Ridge global)** asumiendo un comportamiento unificado para toda la ciudad. Sus resultados empíricos fueron un **MAE de $50.94 USD** y un $R^2$ de **0.1250**. 
Implementar el Enrutamiento Geométrico de 6 clústeres redujo el error promedio en **más de $31 dólares por noche**, demostrando que Río de Janeiro requiere un modelado fuertemente no-lineal y local.

## 🖥️ Interfaz Analítica (Streamlit)
El proyecto incluye un dashboard interactivo para auditar el Feature Engineering, explorar las arquitecturas y revisar empíricamente las predicciones del modelo (hotel por hotel).

Para levantar la interfaz de usuario en tu navegador local, ejecutá:
```bash
streamlit run app.py
```

## 📄 Documentación Detallada

Para una inmersión completa en la metodología, que incluye:
- Ingeniería Espacial y Temporal (Event Flags, Proyección UTM)
- Prevención estricta de Data Leakage y particionamiento por sesión (`searchid`)
- Manejo de nulos condicional e imputación de banderas (`is_missing`)
- Evolución arquitectónica y el experimento con datasets limpios (sin outliers)

Por favor, lea el reporte completo y oficial del proyecto en: 
👉 **[reporte/reporte.pdf](reporte/reporte.pdf)**
