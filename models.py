"""Modèles extraits de V1.ipynb ; normalisation et noyaux conservés.
Le backend GPU optionnel provient de V1_GPU.ipynb.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, RationalQuadratic, WhiteKernel
seed = 2025

class GPRWrapper:
    """
    gère automatiquement la normalisation de l'entrée / sortie
    """
    def __init__(self, kernel, n_restarts=10, device="cpu"):
        self.device = device
        self.kernel = kernel
        self.n_restarts = n_restarts
        self.scaler_x = StandardScaler()

    def fit(self, X, y):
        # Normalisation des entrées 
        X_scaled = self.scaler_x.fit_transform(X)

        # Centrage et réduction de la sortie (y)
        self.y_mean = float(y.mean())
        self.y_std = float(y.std()) if float(y.std()) > 0 else 1.0
        y_scaled = (y - self.y_mean) / self.y_std

        # Optimisation des hyperparamètres par maximisation de la log-vraisemblance marginale.
        regressor = GaussianProcessRegressor
        extra = {}
        if self.device == "cuda":
            from gpu_model import TorchGaussianProcessRegressor
            regressor = TorchGaussianProcessRegressor
            extra["device"] = "cuda"
        self.model = regressor(
            **extra,
            kernel=self.kernel,
            normalize_y=False,
            n_restarts_optimizer=self.n_restarts,
            random_state=seed
        ).fit(X_scaled, y_scaled)

        return self

    def predict(self, X, return_std=False):
        """Prédit après fit, avec retour dans les unités originales de y.

        X : un point [x1, ..., xp] ou une matrice (n_points, p).
        En 1D, passer [[x1], [x2], ...] pour plusieurs points.
        return_std=False : tableau des moyennes (n_points,).
        return_std=True : tuple (moyennes, écarts-types), deux tableaux.
        """
        # Réutilise les statistiques du train ; aucun réapprentissage ici.
        X_scaled = self.scaler_x.transform(np.atleast_2d(X))
        if return_std:
            mean_scaled, std_scaled = self.model.predict(X_scaled, return_std=True)
            # Une translation change la moyenne, mais pas l’écart-type.
            return mean_scaled * self.y_std + self.y_mean, std_scaled * self.y_std

        mean_scaled = self.model.predict(X_scaled)
        return mean_scaled * self.y_std + self.y_mean

def noyaux_candidats(n_dim):
    """Nouveaux objets à chaque appel : amplitude × signal + bruit."""
    signaux = {
        "RBF (ARD)": RBF(np.ones(n_dim), (1e-2, 1e3)),
        "Matérn 3/2 (ARD)": Matern(np.ones(n_dim), (1e-2, 1e3), nu=1.5),
        "Matérn 5/2 (ARD)": Matern(np.ones(n_dim), (1e-2, 1e3), nu=2.5),
        "RationalQuadratic (isotrope)": RationalQuadratic(
            length_scale=1.0, alpha=1.0,
            length_scale_bounds=(1e-2, 1e3), alpha_bounds=(1e-3, 1e3)
        ),
    }
    return {
        nom: ConstantKernel(1.0, (1e-3, 1e3)) * signal
             + WhiteKernel(1e-2, (1e-8, 1e0))
        for nom, signal in signaux.items()
    }

