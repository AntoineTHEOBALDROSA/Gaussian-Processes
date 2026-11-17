"""Entrées mixtes et modèles indépendants par sortie, sans code sérialisé."""
import math
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.special import expit
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.preprocessing import StandardScaler


def category(value):
    if pd.isna(value):
        raise ValueError('Une catégorie est manquante.')
    if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
        if not np.isfinite(value):
            raise ValueError('Une catégorie numérique n’est pas finie.')
        return format(float(value), '.15g')
    return str(value).strip()


def numeric_column(values):
    return pd.to_numeric(values.map(lambda v: v.strip().replace(',', '.') if isinstance(v, str) else v), errors='coerce').replace([np.inf, -np.inf], np.nan)


def infer_type(values):
    valid = values.dropna()
    boolean = len(valid) and all(isinstance(v,(bool,np.bool_)) for v in valid)
    return 'numeric' if len(valid) and not boolean and numeric_column(valid).notna().all() else 'categorical'


class InputEncoder:
    def __init__(self, schema):
        self.schema = schema

    def fit(self, X):
        X = np.asarray(X, dtype=object)
        self.schema = [dict(entry) for entry in self.schema]
        self.names = []
        for j, entry in enumerate(self.schema):
            if entry['type'] == 'categorical':
                entry['categories'] = sorted({category(v) for v in X[:, j]})
                self.names.extend(entry['name']+' = '+v for v in entry['categories'])
            else:
                self.names.append(entry['name'])
        # Resource guard based on encoded matrix memory, not an arbitrary input count.
        if len(X) * len(self.names) * 8 > 256 * 1024**2:
            raise ValueError('L’encodage des catégories dépasse 256 Mo. Réduisez les modalités ou les observations.')
        return self

    def transform(self, X, allow_unknown=False):
        X = np.asarray(X, dtype=object)
        if X.ndim != 2 or X.shape[1] != len(self.schema):
            raise ValueError('Le nombre d’entrées ne correspond pas au modèle.')
        blocks = []
        for j, entry in enumerate(self.schema):
            if entry['type'] == 'categorical':
                values = np.array([category(v) for v in X[:, j]])
                if not allow_unknown and not np.isin(values, entry['categories']).all():
                    raise ValueError('Catégorie inconnue pour '+entry['name']+'. Utilisez une modalité de l’apprentissage.')
                blocks.append((values[:, None] == np.array(entry['categories'])[None, :]).astype(float))
            else:
                if any(isinstance(v, (bool, np.bool_)) for v in X[:, j]):
                    raise ValueError('L’entrée '+entry['name']+' doit être numérique.')
                try:
                    values = numeric_column(pd.Series(X[:, j])).to_numpy(dtype=float)
                except (ValueError, OverflowError) as exc:
                    raise ValueError('Valeur numérique invalide pour '+entry['name']+'.') from exc
                if not np.isfinite(values).all():
                    raise ValueError('Valeur numérique non finie ou manquante pour '+entry['name']+'.')
                blocks.append(values[:, None])
        return np.concatenate(blocks, axis=1)


class GaussianClassifier:
    """GP de classification (Laplace), puis intégration logistique par quadrature.

    La même prédiction numérique est utilisée avant et après sauvegarde.
    Un problème multiclasse utilise un GP binaire par classe (un contre tous).
    """
    def __init__(self, kernel=None, restarts=0):
        self.kernel, self.restarts = kernel, restarts
        self.scaler_x = StandardScaler()

    def fit(self, X, y):
        X = self.scaler_x.fit_transform(X)
        classifier = GaussianProcessClassifier(kernel=self.kernel, n_restarts_optimizer=self.restarts,
                                                random_state=2025, max_iter_predict=100, multi_class='one_vs_rest').fit(X, y)
        self.classes = classifier.classes_.tolist()
        base = classifier.base_estimator_
        estimators = [base] if len(self.classes) == 2 else base.estimators_
        self.binaries = [dict(kernel=e.kernel_, x_train=e.X_train_, coefficients=e.y_train_-e.pi_,
                              weights=e.W_sr_, cholesky=e.L_) for e in estimators]
        return self

    def predict_proba(self, X):
        points = self.scaler_x.transform(X)
        nodes, weights = np.polynomial.hermite.hermgauss(32)
        probabilities = []
        for binary in self.binaries:
            covariance = binary['kernel'](binary['x_train'], points)
            mean = covariance.T @ binary['coefficients']
            v = solve_triangular(binary['cholesky'], binary['weights'][:, None]*covariance, lower=True, check_finite=False)
            variance = np.maximum(binary['kernel'].diag(points)-np.einsum('ij,ij->j', v, v), 0.)
            p = expit(mean[:, None] + np.sqrt(2*variance[:, None])*nodes) @ weights / math.sqrt(math.pi)
            probabilities.append(p)
        p = np.column_stack(probabilities)
        if len(self.classes) == 2:
            p = np.column_stack([1-p[:, 0], p[:, 0]])
        p = np.clip(p, 1e-12, 1)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.asarray(self.classes)[np.argmax(self.predict_proba(X), axis=1)]


class ModelBundle:
    def __init__(self, encoder, outputs):
        self.encoder, self.outputs = encoder, outputs

    def predict_all(self, X, allow_unknown=False):
        encoded = self.encoder.transform(X, allow_unknown=allow_unknown)
        output = {}
        for name, entry in self.outputs.items():
            if entry['type'] == 'categorical':
                model = entry['model']
                p = model.predict_proba(encoded)
                output[name] = dict(type='categorical', classes=model.classes, probabilities=p,
                                    labels=np.asarray(model.classes)[p.argmax(axis=1)])
            else:
                mean, std = entry['model'].predict(encoded, return_std=True)
                if not np.isfinite(np.r_[mean, std]).all():
                    raise ValueError('La prédiction n’est pas finie. Vérifiez l’échelle des entrées.')
                output[name] = dict(type='numeric', mean=mean, std=std)
        return output

    def predict(self, X, return_std=False):
        first = next(iter(self.predict_all(X).values()))
        if first['type'] == 'categorical':
            return first['labels']
        return (first['mean'], first['std']) if return_std else first['mean']
