import os
import joblib
import numpy as np
import polars as pl
import onnxruntime as ort
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import config

class OnnxSessionFactory:
    """Factory Pattern for creating and caching ONNX Inference Sessions."""
    
    _sessions: Dict[str, ort.InferenceSession] = {}

    @classmethod
    def get_session(cls, model_path: str) -> ort.InferenceSession:
        """Retrieves or creates an ONNX InferenceSession for the given model path.

        Args:
            model_path (str): Path to the .onnx model file.

        Returns:
            ort.InferenceSession: The initialized ONNX inference session.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        if model_path not in cls._sessions:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Archivo ONNX no encontrado en: {model_path}")
            # Cargamos la sesión de inferencia de ONNX Runtime
            cls._sessions[model_path] = ort.InferenceSession(model_path)
        return cls._sessions[model_path]


class PredictionStrategy(ABC):
    """Strategy Pattern interface for model prediction strategies."""

    @abstractmethod
    def predict(self, features: pl.DataFrame) -> np.ndarray:
        """Executes prediction using the specific strategy.

        Args:
            features (pl.DataFrame): Features to predict on.

        Returns:
            np.ndarray: Predicted values.
        """
        pass


class StackingStrategy(PredictionStrategy):
    """Prediction strategy using the Stacking Model."""

    def __init__(self, use_onnx: bool = True) -> None:
        self.use_onnx = use_onnx
        self.meta_model = None
        self.base_models = {}

    def predict(self, features: pl.DataFrame) -> np.ndarray:
        """Predicts hotel prices using the Stacking Model.

        Args:
            features (pl.DataFrame): Features to predict.

        Returns:
            np.ndarray: Predicted prices.
        """
        X = features.to_numpy().astype(np.float32)
        
        if self.use_onnx and os.path.exists(config.STACKING_ONNX_PATH):
            try:
                session = OnnxSessionFactory.get_session(config.STACKING_ONNX_PATH)
                input_name = session.get_inputs()[0].name
                label_name = session.get_outputs()[0].name
                preds = session.run([label_name], {input_name: X})[0]
                return np.ravel(preds)
            except Exception as e:
                # Fallback controlado si falla la ejecución ONNX
                print(f"Error ejecutando ONNX Stacking: {e}. Usando fallback Joblib.")
        
        # Fallback a Joblib
        if os.path.exists(config.STACKING_MODEL_PATH):
            try:
                data = joblib.load(config.STACKING_MODEL_PATH)
                # Si es un diccionario con modelos entrenados
                meta = data['meta_model']
                bases = data['base_models']
                pca_pipe = data.get('meta_pca_pipeline', None)
                
                # Generar predicciones de nivel 0
                oof_preds = []
                for name, model in bases.items():
                    oof_preds.append(model.predict(X))
                
                oof_matrix = np.column_stack(oof_preds)
                
                # Añadir PCA features si existen
                if pca_pipe is not None:
                    pca_features = pca_pipe.transform(X)
                    oof_matrix = np.column_stack([oof_matrix, pca_features])
                    
                return meta.predict(oof_matrix)
            except Exception as e:
                raise RuntimeError(f"Error crítico ejecutando Stacking Joblib: {e}") from e
        
        print("Modelos de Stacking no encontrados. Usando simulador heurístico.")
        star_idx = features.columns.index("starRating") if "starRating" in features.columns else 0
        stars = X[:, star_idx] if X.shape[1] > star_idx else np.ones(len(X)) * 3
        return 100.0 + (stars * 75.0) + np.random.normal(0, 10, size=len(X))


class MoEStrategy(PredictionStrategy):
    """Prediction strategy using the Mixture of Experts (MoE) Model."""

    def __init__(self, use_onnx: bool = True) -> None:
        self.use_onnx = use_onnx

    def predict(self, features: pl.DataFrame) -> np.ndarray:
        """Predicts hotel prices using the MoE routing rules.

        Args:
            features (pl.DataFrame): Features to predict.

        Returns:
            np.ndarray: Predicted prices.
        """
        import os
        import joblib
        
        moe_path = os.path.join(config.MODELS_DIR, "moe_model_best.joblib")
        if not os.path.exists(moe_path):
            print("Modelos MoE no encontrados, usando mock...")
            return self._mock_predict(features.to_numpy().astype(np.float32), features.columns)
            
        try:
            data = joblib.load(moe_path)
            modelos = data['modelos']
            expected_cols = data.get('columnas', features.columns)
            
            # Alineación dinámica de features: Rellenar con 0 las columnas dummy faltantes
            missing_cols = [c for c in expected_cols if c not in features.columns]
            if missing_cols:
                features = features.with_columns([pl.lit(0).alias(c) for c in missing_cols])
                
            # Seleccionar exactamente las columnas esperadas en el orden correcto
            X = features.select(expected_cols).to_numpy().astype(np.float32)
            
            # Obtener índices para las reglas heurísticas buscando en los expected_cols
            star_idx = expected_cols.index("starRating") if "starRating" in expected_cols else 0
            rooms_idx = expected_cols.index("numberOfRooms") if "numberOfRooms" in expected_cols else 0
            amenities_idx = expected_cols.index("total_amenities") if "total_amenities" in expected_cols else 0
            
            preds = np.zeros(len(X))
            
            for i, row in enumerate(X):
                stars = row[star_idx] if len(row) > star_idx else 3
                rooms = row[rooms_idx] if len(row) > rooms_idx else 50
                amenities = row[amenities_idx] if len(row) > amenities_idx else 5
                
                if stars >= 4.5 or rooms > 300 or amenities > 15:
                    seg = "Lujo_Resort"
                elif stars < 2 or rooms < 50:
                    seg = "Boutique_Informal"
                else:
                    seg = "Estandar"
                
                # El modelo retorna log1p, así que usamos expm1
                row_reshaped = row.reshape(1, -1)
                pred_log = modelos[seg].predict(row_reshaped)[0]
                preds[i] = np.expm1(pred_log)
                
            return preds
            
        except Exception as e:
            # Eliminar la caída silenciosa: Ahora cualquier error explota para no ocultar inconsistencias
            raise RuntimeError(f"Error crítico en MoEStrategy durante la inferencia: {e}") from e
            
    def _mock_predict(self, X: np.ndarray, feature_names: list = None) -> np.ndarray:
        if feature_names is None:
            feature_names = config.FEATURES
            
        star_idx = feature_names.index("starRating") if "starRating" in feature_names else 0
        amenities_idx = feature_names.index("total_amenities") if "total_amenities" in feature_names else 0
        
        preds = []
        for row in X:
            stars = row[star_idx] if len(row) > star_idx else 3
            amenities = row[amenities_idx] if len(row) > amenities_idx else 5
            
            if stars >= 4.5:
                base_val = 250.0 + (amenities * 5.0)
            elif stars >= 3.0 and amenities > 10:
                base_val = 150.0 + (amenities * 4.0)
            else:
                base_val = 80.0 + (amenities * 3.0)
            preds.append(base_val)
        return np.array(preds)


class ModelService:
    """Service layer coordinating model inferences via Strategies."""

    def __init__(self, use_onnx: bool = True) -> None:
        self._use_onnx = use_onnx
        self._strategies: Dict[str, PredictionStrategy] = {
            "Stacking": StackingStrategy(use_onnx=use_onnx),
            "MoE": MoEStrategy(use_onnx=use_onnx)
        }

    def predict(self, model_type: str, features: pl.DataFrame) -> np.ndarray:
        """Executes predictions for a given model type.

        Args:
            model_type (str): Either 'Stacking' or 'MoE'.
            features (pl.DataFrame): Polars DataFrame containing features.

        Returns:
            np.ndarray: Predicted values.

        Raises:
            ValueError: If the model_type is invalid.
        """
        if model_type not in self._strategies:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
        
        return self._strategies[model_type].predict(features)
        
    @staticmethod
    def get_submodel_predictions(features: pl.DataFrame) -> Dict[str, np.ndarray]:
        import streamlit as st
        import joblib
        
        X = features.to_numpy().astype(np.float32)
        
        @st.cache_resource
        def load_unified_models():
            if os.path.exists(config.STACKING_MODEL_PATH):
                try:
                    data = joblib.load(config.STACKING_MODEL_PATH)
                    return data.get('base_models', {})
                except Exception as e:
                    print(f"Error cargando modelos base desde el archivo unificado: {e}")
                    return {}
            return {}

        base_models = load_unified_models()
        preds = {}
        
        # Inferencia paralela extrayendo los submodelos reales del pipeline actual
        if not base_models:
            preds['RandomForest (Fallback)'] = np.zeros(len(X))
            preds['XGBoost (Fallback)'] = np.zeros(len(X))
            preds['LightGBM (Fallback)'] = np.zeros(len(X))
            return preds
            
        for name, model in base_models.items():
            try:
                preds[f"{name} (Nivel 0)"] = model.predict(X)
            except Exception as e:
                preds[f"{name} (Nivel 0 Error)"] = np.zeros(len(X))
                
        return preds
