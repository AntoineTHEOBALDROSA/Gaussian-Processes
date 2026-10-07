"""Pipeline local fondé sur V1.ipynb (et V1_GPU.ipynb pour CUDA)."""
import warnings
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from threadpoolctl import threadpool_limits
from models import GPRWrapper, noyaux_candidats, search_options, bound_diagnostics


def prepare_data(frame, target, features):
    if not isinstance(target, str) or target not in frame.columns:
        raise ValueError('Choisissez une variable cible présente dans le fichier.')
    if (not isinstance(features, list) or not features or len(features) > 20
            or any(not isinstance(c, str) or c not in frame.columns for c in features)
            or target in features or len(set(features)) != len(features)):
        raise ValueError('Choisissez entre 1 et 20 entrées distinctes, différentes de la cible.')
    selected = frame[features + [target]].apply(pd.to_numeric, errors='coerce')
    selected = selected.replace([np.inf, -np.inf], np.nan).dropna()
    if len(selected) < 10:
        raise ValueError('Il faut au moins 10 lignes numériques complètes pour ces variables.')
    X = selected[features].to_numpy(dtype=float)
    y = selected[target].to_numpy(dtype=float)
    unique, groups = np.unique(X, axis=0, return_inverse=True)
    if len(unique) < 8:
        raise ValueError('Il faut au moins 8 configurations d’entrée distinctes.')
    # Les doublons restent ensemble, pour le test indépendant comme pour la CV.
    train_groups, test_groups = train_test_split(np.arange(len(unique)), test_size=.2, random_state=2025)
    train = np.flatnonzero(np.isin(groups, train_groups))
    test = np.flatnonzero(np.isin(groups, test_groups))
    return selected, X, y, train, test


def run_analysis(frame, options, progress):
    start = perf_counter()
    features, target = options['features'], options['target']
    selected, X, y, train, test = prepare_data(frame, target, features)
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]
    quick = options['quality'] == 'quick'
    folds, cv_restarts, final_restarts = (3, 0, 1) if quick else (5, 2, 10)
    search, bounds = search_options(options)
    kernels = noyaux_candidats(len(features), extended=search=='extended', length_bounds=bounds)
    unique, groups = np.unique(X_train, axis=0, return_inverse=True)
    partitions = list(KFold(n_splits=folds, shuffle=True, random_state=2025).split(unique))
    ranking, fit_warnings = [], []
    completed, total = 0, len(kernels) * folds + 2
    with threadpool_limits(limits=1):
        for name, kernel in kernels.items():
            scores, messages = [], []
            kernel_start = perf_counter()
            for fold, (g_train, g_valid) in enumerate(partitions):
                progress(int(completed / total * 100), f'Comparaison · {name}', f'Validation croisée : pli {fold + 1} sur {folds}')
                fit = np.flatnonzero(np.isin(groups, g_train))
                valid = np.flatnonzero(np.isin(groups, g_valid))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    model = GPRWrapper(clone(kernel), cv_restarts, options['device']).fit(X_train[fit], y_train[fit])
                messages.extend(str(w.message) for w in caught)
                pred = model.predict(X_train[valid])
                scores.append(float(np.sqrt(mean_squared_error(y_train[valid], pred))))
                completed += 1
            ranking.append(dict(name=name, rmse=float(np.mean(scores)), std=float(np.std(scores)), seconds=perf_counter()-kernel_start, warnings=len(messages)))
            fit_warnings.extend(messages)
        ranking.sort(key=lambda entry: entry['rmse'])
        winner = ranking[0]['name']
        progress(int(completed / total * 100), 'Ajustement du meilleur noyau', winner)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = GPRWrapper(clone(kernels[winner]), final_restarts, options['device']).fit(X_train, y_train)
        fit_warnings.extend(str(w.message) for w in caught)
        pred, std = model.predict(X_test, return_std=True)
        if not np.isfinite(pred).all() or not np.isfinite(std).all():
            raise ValueError('Le modèle produit des valeurs non finies. Vérifiez les données et leur échelle.')
        progress(96, 'Préparation des graphiques', 'Prédictions, coupes et intervalles prédictifs')
        slices = []
        for j, name in enumerate(features):
            grid = np.linspace(X_train[:, j].min(), X_train[:, j].max(), 150)
            grid_X = np.tile(np.median(X_train, axis=0), (len(grid), 1))
            grid_X[:, j] = grid
            mu, sd = model.predict(grid_X, return_std=True)
            slices.append(dict(name=name, x=grid.tolist(), mean=mu.tolist(), lower=(mu-1.96*sd).tolist(), upper=(mu+1.96*sd).tolist(), train_x=X_train[:, j].tolist(), train_y=y_train.tolist()))
    prediction_frame = selected.iloc[test].copy()
    prediction_frame.columns = ['entree_' + c for c in features] + ['valeur_reelle']
    prediction_frame.insert(0, 'ligne_donnees', selected.index[test].to_numpy() + 1)
    prediction_frame['prediction'] = pred
    prediction_frame['sigma'] = std
    prediction_frame['IC95_bas'] = pred - 1.96 * std
    prediction_frame['IC95_haut'] = pred + 1.96 * std
    result = dict(
        target=target, features=features, device=options['device'], quality=options['quality'],
        kernel=winner, optimized_kernel=str(model.model.kernel_), ranking=ranking,
        kernel_search=search, length_bounds=list(bounds), bound_diagnostics=bound_diagnostics(model.model.kernel_, features),
        metrics=dict(mae=float(mean_absolute_error(y_test, pred)), rmse=float(np.sqrt(mean_squared_error(y_test, pred))), coverage=float(np.mean(np.abs(y_test-pred) <= 1.96*std))*100),
        parity=dict(actual=y_test.tolist(), predicted=pred.tolist(), std=std.tolist()),
        input_ranges=[dict(name=name, minimum=float(X_train[:, j].min()), maximum=float(X_train[:, j].max()), median=float(np.median(X_train[:, j]))) for j, name in enumerate(features)],
        slices=slices, data=dict(x=X.tolist(), y=y.tolist(), train=train.tolist(), test=test.tolist()),
        train_count=len(train), test_count=len(test), dropped=len(frame)-len(selected),
        warnings=list(dict.fromkeys(fit_warnings)), seconds=perf_counter()-start,
        folds=folds, seed=2025,
    )
    return result, prediction_frame.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig'), model
