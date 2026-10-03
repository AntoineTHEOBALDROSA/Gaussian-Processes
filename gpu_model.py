# Extrait de V1_GPU.ipynb. Importé uniquement pour le mode CUDA.
"""Processus gaussien exact : calcul matriciel PyTorch sur CUDA ou CPU explicite.

Les noyaux scikit-learn servent seulement à décrire les paramètres et leurs bornes.
Les matrices, Cholesky, résolutions et gradients sont calculés sur le device choisi.
SciPy pilote L-BFGS-B sur CPU et échange uniquement le petit vecteur de paramètres,
la valeur de l'objectif et son gradient. Les données d'apprentissage restent sur GPU.
"""

import warnings

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process.kernels import (
    ConstantKernel, Matern, Product, RationalQuadratic, RBF, Sum, WhiteKernel,
)


def choisir_device(device="cuda"):
    """Ne remplace jamais silencieusement CUDA par le CPU."""
    result = torch.device(device)
    if result.type not in ("cuda", "cpu"):
        raise ValueError("Choisir 'cuda', 'cuda:0', ... ou explicitement 'cpu'.")
    if result.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Aucun GPU CUDA accessible. Vérifier nvidia-smi et la version CUDA "
                "de PyTorch dans le Python du notebook. Exécuter sur la machine RTX A4000. "
                "Pour un essai CPU volontaire uniquement, choisir DEVICE_GPU = 'cpu'."
            )
        index = torch.cuda.current_device() if result.index is None else result.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"GPU CUDA {index} absent.")
        result = torch.device("cuda", index)
    return result


class TorchGaussianProcessRegressor:
    """API limitée à celle utilisée dans V1 : fit, predict et kernel_.

    Prise en charge : ConstantKernel * (RBF / Matern 1.5 ou 2.5 /
    RationalQuadratic) + WhiteKernel. Calcul en float64 pour les GP peu bruités.
    Le centrage et la réduction de X/y sont assurés par GPRWrapper du notebook.
    """

    def __init__(self, kernel, normalize_y=False, n_restarts_optimizer=10,
                 random_state=2025, device="cuda", alpha=1e-10,
                 maxiter=200, optimizer="L-BFGS-B"):
        if normalize_y:
            raise ValueError("Le notebook normalise déjà y : utiliser normalize_y=False.")
        if optimizer not in ("L-BFGS-B", None):
            raise ValueError("optimizer doit être 'L-BFGS-B' ou None.")
        if not isinstance(n_restarts_optimizer, (int, np.integer)) or n_restarts_optimizer < 0:
            raise ValueError("n_restarts_optimizer doit être un entier >= 0.")
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha doit être strictement positif et fini.")
        if maxiter < 1:
            raise ValueError("maxiter doit être >= 1.")
        self.kernel = kernel
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        self.device = device
        self.alpha = float(alpha)
        self.maxiter = maxiter
        self.optimizer = optimizer

    def _preparer_noyau(self):
        k = self.kernel_
        if not (isinstance(k, Sum) and isinstance(k.k1, Product)
                and isinstance(k.k1.k1, ConstantKernel) and isinstance(k.k2, WhiteKernel)):
            raise ValueError("Structure attendue : ConstantKernel * signal + WhiteKernel.")
        self.signal_ = k.k1.k2
        if isinstance(self.signal_, Matern):
            if self.signal_.nu not in (1.5, 2.5):
                raise ValueError("Seuls les Matérn 3/2 et 5/2 sont pris en charge.")
        elif not isinstance(self.signal_, (RBF, RationalQuadratic)):
            raise ValueError("Signal pris en charge : RBF, Matérn ou RationalQuadratic.")
        longueurs = np.atleast_1d(self.signal_.length_scale)
        if longueurs.size not in (1, self.n_features_in_):
            raise ValueError("Il faut une longueur commune ou une longueur par entrée.")

        # Respecte l'ordre des paramètres de kernel.theta, y compris les paramètres fixes.
        self._slices = {}
        self._fixes = {}
        offset = 0
        params = k.get_params()
        for hp in k.hyperparameters:
            if hp.fixed:
                self._fixes[hp.name] = self._tensor(np.atleast_1d(params[hp.name]))
            else:
                self._slices[hp.name] = slice(offset, offset + hp.n_elements)
                offset += hp.n_elements

    def _tensor(self, values):
        return torch.as_tensor(values, dtype=torch.float64, device=self.device_)

    def _parametres(self, theta):
        values = {key: theta[part].exp() for key, part in self._slices.items()}
        values.update(self._fixes)
        return values

    def _covariance_signal(self, differences_carrees, params):
        amplitude = params["k1__k1__constant_value"][0]
        longueurs = params["k1__k2__length_scale"]
        distance2 = (differences_carrees / longueurs.square()).sum(dim=-1)
        if isinstance(self.signal_, Matern):
            # Évite la dérivée infinie de sqrt(0) sur la diagonale et les doublons.
            distance = distance2.clamp_min(1e-30).sqrt()
            if self.signal_.nu == 1.5:
                r = np.sqrt(3.0) * distance
                correlation = (1.0 + r) * (-r).exp()
            else:
                r = np.sqrt(5.0) * distance
                correlation = (1.0 + r + (5.0 / 3.0) * distance2) * (-r).exp()
        elif isinstance(self.signal_, RationalQuadratic):
            a = params["k1__k2__alpha"][0]
            correlation = (-a * torch.log1p(distance2 / (2.0 * a))).exp()
        else:
            correlation = (-0.5 * distance2).exp()
        return amplitude * correlation

    def _factoriser(self, theta):
        params = self._parametres(theta)
        covariance = self._covariance_signal(self._differences2, params)
        covariance = covariance + (params["k2__noise_level"][0] + self.alpha) * self._identite
        try:
            chol = torch.linalg.cholesky(covariance)
        except torch.linalg.LinAlgError as exc:
            raise RuntimeError(
                "La covariance n'est pas définie positive en float64. "
                "Examiner les doublons, le bruit et le terme alpha."
            ) from exc
        coefficients = torch.cholesky_solve(self._y[:, None], chol)
        return chol, coefficients

    def _objectif(self, theta_numpy):
        theta = self._tensor(theta_numpy).detach().requires_grad_(True)
        chol, coefficients = self._factoriser(theta)
        perte = (
            0.5 * (self._y[:, None] * coefficients).sum()
            + torch.log(torch.diagonal(chol)).sum()
            + 0.5 * len(self._y) * np.log(2.0 * np.pi)
        )
        gradient, = torch.autograd.grad(perte, theta)
        valeur = float(perte.detach().cpu())
        grad = gradient.detach().cpu().numpy().copy()
        if not np.isfinite(valeur) or not np.isfinite(grad).all():
            raise FloatingPointError("Objectif ou gradient non fini pendant l'apprentissage.")
        return valeur, grad

    def fit(self, X, y):
        X = np.array(X, dtype=float, copy=True)
        y = np.array(y, dtype=float, copy=True)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or X.size == 0:
            raise ValueError("X doit être (n_points, n_entrées) et y (n_points,).")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X et y doivent être finis.")
        self.device_ = choisir_device(self.device)
        self.n_features_in_ = X.shape[1]
        self.kernel_ = clone(self.kernel)
        self._preparer_noyau()
        self._X, self._y = self._tensor(X), self._tensor(y)
        self._differences2 = (self._X[:, None, :] - self._X[None, :, :]).square()
        self._identite = torch.eye(len(X), dtype=torch.float64, device=self.device_)
        theta_initial = self.kernel_.theta.copy()
        self.optimization_results_ = []
        if self.optimizer is not None and theta_initial.size:
            bornes = self.kernel_.bounds
            if not np.isfinite(bornes).all():
                raise ValueError("La recherche nécessite des bornes finies.")
            rng = np.random.RandomState(self.random_state)
            departs = [theta_initial] + [
                rng.uniform(bornes[:, 0], bornes[:, 1])
                for _ in range(self.n_restarts_optimizer)
            ]
            for depart in departs:
                resultat = minimize(
                    self._objectif, depart, method="L-BFGS-B", jac=True,
                    bounds=bornes, options={"maxiter": self.maxiter},
                )
                if not resultat.success:
                    warnings.warn(f"L-BFGS-B : {resultat.message}", ConvergenceWarning)
                self.optimization_results_.append(resultat)
            meilleur = min(self.optimization_results_, key=lambda r: r.fun)
            theta_final = meilleur.x
            self.log_marginal_likelihood_value_ = -float(meilleur.fun)
            self.kernel_ = self.kernel_.clone_with_theta(theta_final)
            proches = np.isclose(theta_final, bornes[:, 0]) | np.isclose(theta_final, bornes[:, 1])
            if proches.any():
                warnings.warn(
                    "Un ou plusieurs paramètres sont proches de leurs bornes : "
                    f"{self.kernel_}", ConvergenceWarning,
                )
        else:
            theta_final = theta_initial
        self._theta = self._tensor(theta_final)
        with torch.no_grad():
            self._chol, self._coefficients = self._factoriser(self._theta)
            if not self.optimization_results_:
                self.log_marginal_likelihood_value_ = -float((
                    0.5 * (self._y[:, None] * self._coefficients).sum()
                    + torch.log(torch.diagonal(self._chol)).sum()
                    + 0.5 * len(y) * np.log(2.0 * np.pi)
                ).cpu())
        # Ces tableaux ne servent plus une fois les hyperparamètres ajustés.
        del self._differences2, self._identite
        return self

    def predict(self, X, return_std=False, batch_size=512):
        if not hasattr(self, "_coefficients"):
            raise RuntimeError("Appeler fit avant predict.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError(f"X doit être une matrice finie avec {self.n_features_in_} colonnes.")
        if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError("batch_size doit être un entier >= 1.")
        if len(X) == 0:
            return (np.empty(0), np.empty(0)) if return_std else np.empty(0)
        moyennes, ecarts_types = [], []
        with torch.no_grad():
            params = self._parametres(self._theta)
            for debut in range(0, len(X), batch_size):
                points = self._tensor(X[debut:debut + batch_size])
                differences2 = (points[:, None, :] - self._X[None, :, :]).square()
                covariance = self._covariance_signal(differences2, params)
                moyennes.append((covariance @ self._coefficients).flatten().cpu().numpy())
                if return_std:
                    v = torch.linalg.solve_triangular(self._chol, covariance.T, upper=False)
                    variance = (params["k1__k1__constant_value"][0]
                                + params["k2__noise_level"][0] - v.square().sum(dim=0))
                    if (variance < -1e-8).any():
                        warnings.warn("Variance prédite négative : vérifier le conditionnement.")
                    ecarts_types.append(variance.clamp_min(0).sqrt().cpu().numpy())
        moyenne = np.concatenate(moyennes)
        return (moyenne, np.concatenate(ecarts_types)) if return_std else moyenne