"""Modèles extraits de V1.ipynb ; normalisation et noyaux conservés.
Le backend GPU optionnel provient de V1_GPU.ipynb.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, RationalQuadratic, WhiteKernel, DotProduct
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

def search_options(options):
    mode = options.get('kernel_search', 'base')
    if mode not in ('base', 'extended'):
        raise ValueError('Choisissez une recherche de noyaux simple ou élargie.')
    if mode == 'extended' and options.get('device') != 'cpu':
        raise ValueError('La recherche élargie de noyaux nécessite le CPU.')
    bounds = options.get('length_bounds', [1e-2, 1e3])
    if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
            or any(isinstance(v, bool) or not isinstance(v, (float, int)) for v in bounds)
            or not np.isfinite(bounds).all() or not 1e-12 <= bounds[0] < bounds[1] <= 1e12):
        raise ValueError('Les bornes des longueurs doivent être positives, croissantes et comprises entre 10⁻¹² et 10¹².')
    return mode, tuple(bounds)


def noyaux_candidats(n_dim, extended=False, length_bounds=(1e-2, 1e3)):
    """Nouveaux objets à chaque appel : amplitude × signal + bruit."""
    initial = float(np.clip(1., *length_bounds))
    signaux = {
        "RBF (ARD)": RBF(np.full(n_dim, initial), length_bounds),
        "Matérn 3/2 (ARD)": Matern(np.full(n_dim, initial), length_bounds, nu=1.5),
        "Matérn 5/2 (ARD)": Matern(np.full(n_dim, initial), length_bounds, nu=2.5),
        "RationalQuadratic (isotrope)": RationalQuadratic(
            length_scale=initial, alpha=1.0,
            length_scale_bounds=length_bounds, alpha_bounds=(1e-3, 1e3)
        ),
    }
    if extended:
        signaux.update({
            'Linéaire': DotProduct(1., (1e-3,1e3)),
            'Quadratique': DotProduct(1., (1e-3,1e3)) ** 2,
            'Tendance linéaire + Matérn 3/2': DotProduct(1., (1e-3,1e3))
                + Matern(np.full(n_dim, initial), length_bounds, nu=1.5),
            'RBF à deux échelles': RBF(np.full(n_dim, initial),length_bounds)
                + ConstantKernel(1., (1e-3,1e3))*RBF(np.full(n_dim,float(np.clip(10.,*length_bounds))),length_bounds),
        })
    return {
        nom: ConstantKernel(1.0, (1e-3, 1e3)) * signal
             + WhiteKernel(1e-2, (1e-8, 1e0))
        for nom, signal in signaux.items()
    }


def bound_diagnostics(kernel, features):
    records, offset = [], 0
    for hp in kernel.hyperparameters:
        if hp.fixed:
            continue
        for index in range(hp.n_elements):
            theta = kernel.theta[offset]
            lower, upper = kernel.bounds[offset]
            tolerance = max(1e-5, .001 * (upper-lower))
            hit = 'lower' if theta-lower <= tolerance else 'upper' if upper-theta <= tolerance else None
            if hp.name.endswith('length_scale'):
                label = 'Longueur de ' + features[index] if hp.n_elements == len(features) else 'Longueur commune'
            else:
                label = {'constant_value':'Variance du signal', 'noise_level':'Variance du bruit',
                         'sigma_0':'Paramètre linéaire', 'alpha':'Mélange des échelles'}.get(hp.name.split('__')[-1], hp.name)
            explanation = ''
            if hit:
                if hp.name.endswith('noise_level') and hit == 'lower':
                    explanation = 'Bruit très faible : cela peut être normal pour une simulation déterministe.'
                elif hp.name.endswith('length_scale') and hit == 'upper':
                    explanation = 'Variation très lente selon cette entrée, ou borne trop basse. À départager par validation.'
                else:
                    explanation = 'Essayer une borne différente puis comparer les erreurs de validation, sans choisir sur le test final.'
            records.append(dict(name=label, parameter=hp.name, value=float(np.exp(theta)),
                                lower=float(np.exp(lower)), upper=float(np.exp(upper)), hit=hit, explanation=explanation))
            offset += 1
    return records
