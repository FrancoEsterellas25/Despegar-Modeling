import numpy as np
import pandas as pd
import shap
import warnings

class ExactStackingExplainer:
    """
    Calcula los valores SHAP exactos y analíticos para un modelo Stacking
    que combina modelos basados en árboles y componentes principales (PCA),
    usando la propiedad de aditividad lineal de Shapley.
    """
    def __init__(self, stacking_data, X_background):
        """
        Inicializa el pipeline explicador.
        
        Args:
            stacking_data (dict): Diccionario cargado desde joblib con 'base_models', 
                                  'meta_model' y 'meta_pca_pipeline'.
            X_background (pd.DataFrame o np.ndarray): Dataset de fondo para 
                                                      TreeExplainer y valores esperados.
        """
        self.meta_model = stacking_data['meta_model']
        self.base_models = stacking_data['base_models']
        self.pca_pipeline = stacking_data['meta_pca_pipeline']
        
        if isinstance(X_background, pd.DataFrame):
            self.feature_names = X_background.columns.tolist()
            self.X_bg_arr = X_background.values
        else:
            self.feature_names = [f"Feature_{i}" for i in range(X_background.shape[1])]
            self.X_bg_arr = X_background
            
        self.num_features = self.X_bg_arr.shape[1]
        
        # 1. Instanciar TreeExplainers para cada modelo base usando X_background
        self.explainers = {}
        # Suprimimos warnings de SHAP sobre la aditividad durante la inicialización
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, model in self.base_models.items():
                est = model
                if hasattr(model, 'steps'):
                    est = model.named_steps['estimator']
                self.explainers[name] = shap.TreeExplainer(est, data=self.X_bg_arr, feature_perturbation='interventional')
            
        # 2. Extraer pesos del metamodelo
        self.meta_weights = self.meta_model.coef_
        self.meta_intercept = self.meta_model.intercept_
        
        self.num_base_models = len(self.base_models)
        self.base_weights = self.meta_weights[:self.num_base_models]
        self.pca_weights = self.meta_weights[self.num_base_models:]
        
        # 3. Extraer componentes y scaler de PCA
        self.scaler = self.pca_pipeline.named_steps['scaler']
        self.pca = self.pca_pipeline.named_steps['pca']
        
        self.alpha = self.pca.components_[0]
        self.beta = self.pca.components_[1]
        
        # 4. Calcular el Valor Esperado Global
        self.mu_bg = np.mean(self.X_bg_arr, axis=0)
        
        self.base_expected_values = []
        for name in self.base_models.keys():
            exp_val = self.explainers[name].expected_value
            if isinstance(exp_val, np.ndarray) or isinstance(exp_val, list):
                exp_val = exp_val[0]
            self.base_expected_values.append(exp_val)
            
        X_bg_scaled = self.scaler.transform(self.mu_bg.reshape(1, -1))
        self.pca_expected_value = self.pca.transform(X_bg_scaled)[0]
        
        self.expected_value = self.meta_intercept
        self.expected_value += np.sum(self.base_weights * np.array(self.base_expected_values))
        self.expected_value += np.sum(self.pca_weights * self.pca_expected_value)

    def explain_instance(self, x):
        """
        Calcula los valores SHAP exactos para una instancia x.
        
        Args:
            x (pd.Series, pd.DataFrame o np.ndarray): Instancia a explicar (1D o 2D con 1 fila).
            
        Returns:
            pd.DataFrame: DataFrame con los nombres de variables y sus valores SHAP exactos.
        """
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x_arr = x.values.reshape(1, -1)
        else:
            x_arr = x.reshape(1, -1)
            
        # 1. SHAP de los Modelos Base
        base_shaps = []
        for name in self.base_models.keys():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sv = self.explainers[name].shap_values(x_arr)
            # Para interventional TreeExplainer en un registro, retorna (1, n_features)
            base_shaps.append(sv[0])
            
        base_shaps = np.array(base_shaps)
        
        total_shap = np.zeros(self.num_features)
        for i in range(self.num_base_models):
            total_shap += self.base_weights[i] * base_shaps[i]
            
        # 2. SHAP de la porción PCA
        pc_loadings = self.pca_weights[0] * self.alpha + self.pca_weights[1] * self.beta
        sigma_scaler = self.scaler.scale_
        pca_shaps = pc_loadings * (x_arr[0] - self.mu_bg) / sigma_scaler
        
        total_shap += pca_shaps
        
        # 3. Verificación del Axioma de Eficiencia (Aditividad)
        base_preds = []
        for name, model in self.base_models.items():
            base_preds.append(model.predict(x_arr)[0])
            
        pca_features = self.pca_pipeline.transform(x_arr)[0]
        
        meta_features = np.concatenate([base_preds, pca_features]).reshape(1, -1)
        actual_prediction = self.meta_model.predict(meta_features)[0]
        
        shap_sum = np.sum(total_shap) + self.expected_value
        
        assert np.isclose(shap_sum, actual_prediction, atol=1e-5), \
            f"Fallo en la aditividad: Suma SHAP + Base ({shap_sum}) != Predicción ({actual_prediction})"
            
        # 4. Formatear salida
        res = pd.DataFrame({
            'Feature': self.feature_names,
            'SHAP_Value': total_shap,
            'Feature_Value': x_arr[0]
        })
        
        return res
