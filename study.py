"""Études mixtes : même partition par configurations, un GP par sortie."""
import warnings
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score, log_loss, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from threadpoolctl import threadpool_limits
from models import GPRWrapper, noyaux_candidats, search_options, bound_diagnostics
from typed_models import InputEncoder, GaussianClassifier, ModelBundle, category, infer_type, numeric_column


def prepare_study(frame, options):
    features, targets = options.get('features'), options.get('targets')
    for names, label in ((features, 'entrée'), (targets, 'sortie')):
        if (not isinstance(names, list) or not names or any(not isinstance(n, str) or n not in frame.columns for n in names)
                or len(set(names)) != len(names)):
            raise ValueError('Choisissez au moins une variable d’'+label+' distincte et présente dans le fichier.')
    if set(features) & set(targets):
        raise ValueError('Une colonne ne peut pas être à la fois entrée et sortie.')
    overrides = options.get('column_types', {})
    if not isinstance(overrides, dict) or any(v not in ('numeric', 'categorical') for v in overrides.values()):
        raise ValueError('Types des variables invalides.')
    types = {name: overrides.get(name, infer_type(frame[name])) for name in features+targets}
    selected = frame[features+targets].copy()
    for name in selected:
        if types[name] == 'numeric':
            selected[name] = numeric_column(selected[name])
        else:
            selected[name] = selected[name].map(lambda v: np.nan if pd.isna(v) or not str(v).strip() else category(v))
    selected = selected.dropna()
    if len(selected) < 10:
        raise ValueError('Il faut au moins 10 lignes complètes pour les entrées et sorties choisies.')
    # Factorize typed row tuples; string and numeric input columns may coexist.
    groups, unique = pd.factorize(pd.MultiIndex.from_frame(selected[features]), sort=False)
    if len(unique) < 8:
        raise ValueError('Il faut au moins 8 configurations d’entrée distinctes.')
    for name in targets:
        if types[name] == 'categorical' and selected[name].nunique() < 2:
            raise ValueError('La sortie catégorielle '+name+' doit contenir au moins deux classes.')
    folds = 3 if options.get('quality') == 'quick' else 5
    categories = [n for n in targets if types[n] == 'categorical']
    classes = {n: set(selected[n]) for n in categories}
    for attempt in range(100):
        split_seed = 2025+attempt
        tg, vg = train_test_split(np.arange(len(unique)), test_size=.2, random_state=split_seed)
        train, test = np.flatnonzero(np.isin(groups, tg)), np.flatnonzero(np.isin(groups, vg))
        partitions = []
        for fit, valid in KFold(n_splits=folds, shuffle=True, random_state=split_seed).split(tg):
            partitions.append((np.flatnonzero(np.isin(groups, tg[fit])), np.flatnonzero(np.isin(groups, tg[valid]))))
        if all(set(selected.iloc[ids][name]) == classes[name] for name in categories for ids in [train]+[p[0] for p in partitions]):
            break
    else:
        raise ValueError('Certaines classes sont présentes dans trop peu de configurations pour la validation croisée. Ajoutez des observations de ces classes ou utilisez le mode rapide.')
    return selected, types, train, test, partitions, split_seed


def run_study(frame, options, progress):
    start = perf_counter()
    selected, types, train, test, partitions, split_seed = prepare_study(frame, options)
    features, targets = options['features'], options['targets']
    raw = selected[features].to_numpy(dtype=object)
    schema = [dict(name=n, type=types[n]) for n in features]
    encoder = InputEncoder(schema).fit(raw[train])
    final_X = encoder.transform(raw[train])
    quick = options['quality'] == 'quick'
    cv_restarts, final_restarts = (0, 1) if quick else (2, 10)
    search, bounds = search_options(options)
    final_bank = noyaux_candidats(final_X.shape[1], search == 'extended', bounds)
    folds = []
    for fit, valid in partitions:
        enc = InputEncoder(schema).fit(raw[fit])
        folds.append((fit, valid, enc.transform(raw[fit]), enc.transform(raw[valid], allow_unknown=True)))
    outputs, models, messages = [], {}, []
    total = len(targets) * (len(final_bank)*len(folds)+1)
    done = 0
    ranges, baseline = [], []
    for j, entry in enumerate(encoder.schema):
        name = entry['name']
        values = selected.iloc[train][name]
        if entry['type'] == 'numeric':
            r = dict(name=name, type='numeric', minimum=float(values.min()), maximum=float(values.max()), median=float(values.median()))
            baseline.append(r['median'])
        else:
            r = dict(name=name, type='categorical', categories=entry['categories'], mode=values.mode().iloc[0])
            baseline.append(r['mode'])
        ranges.append(r)
    with threadpool_limits(limits=1):
        for target in targets:
            categorical = types[target] == 'categorical'
            y = selected[target].to_numpy(dtype=str if categorical else float)
            ranking = []
            classes = sorted(set(y)) if categorical else None
            for kernel_name in final_bank:
                scores, alerts = [], []
                t0 = perf_counter()
                for fold, (fit, valid, xf, xv) in enumerate(folds):
                    progress(int(95*done/total), target+' · '+kernel_name, f'Pli {fold+1}/{len(folds)} · sortie {targets.index(target)+1}/{len(targets)}')
                    kernel = noyaux_candidats(xf.shape[1], search == 'extended', bounds)[kernel_name]
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        if categorical:
                            model = GaussianClassifier(kernel.k1, cv_restarts).fit(xf, y[fit])
                            scores.append(float(log_loss(y[valid], model.predict_proba(xv), labels=classes)))
                        else:
                            model = GPRWrapper(kernel, cv_restarts, options['device']).fit(xf, y[fit])
                            scores.append(float(np.sqrt(mean_squared_error(y[valid], model.predict(xv)))))
                    alerts.extend(str(w.message) for w in caught)
                    done += 1
                row = dict(name=kernel_name, score=float(np.mean(scores)), std=float(np.std(scores)), seconds=perf_counter()-t0, warnings=len(alerts))
                if not categorical:
                    row['rmse'] = row['score']
                ranking.append(row)
                messages.extend(alerts)
            ranking.sort(key=lambda row: row['score'])
            winner = ranking[0]['name']
            progress(int(95*done/total), 'Entraînement final · '+target, winner)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                kernel = final_bank[winner]
                model = (GaussianClassifier(kernel.k1, final_restarts).fit(final_X, y[train]) if categorical
                         else GPRWrapper(kernel, final_restarts, options['device']).fit(final_X, y[train]))
            messages.extend(str(w.message) for w in caught)
            models[target] = dict(type=types[target], model=model)
            done += 1
            test_X = encoder.transform(raw[test], allow_unknown=True)
            output = dict(target=target, type=types[target], kernel=winner, ranking=ranking,
                          score_name='Log-loss' if categorical else 'RMSE', device='cpu' if categorical else options['device'])
            if categorical:
                probabilities = model.predict_proba(test_X)
                prediction = np.asarray(model.classes)[probabilities.argmax(axis=1)]
                output.update(classes=model.classes, metrics=dict(accuracy=float(accuracy_score(y[test], prediction))*100,
                    f1=float(f1_score(y[test], prediction, labels=classes, average='macro', zero_division=0)),
                    log_loss=float(log_loss(y[test], probabilities, labels=classes))),
                    confusion=confusion_matrix(y[test], prediction, labels=classes).tolist(),
                    parity=dict(actual=y[test].tolist(), predicted=prediction.tolist(), probabilities=probabilities.tolist()),
                    optimized_kernel=' ; '.join(str(b['kernel']) for b in model.binaries), bound_diagnostics=[],
                    kernel_variants=[dict(label=model.classes[1] if len(model.binaries)==1 else model.classes[j],
                        bound_diagnostics=bound_diagnostics(binary['kernel'], encoder.names)) for j,binary in enumerate(model.binaries)])
            else:
                prediction, sd = model.predict(test_X, return_std=True)
                output.update(metrics=dict(mae=float(mean_absolute_error(y[test], prediction)),
                    rmse=float(np.sqrt(mean_squared_error(y[test], prediction))), coverage=float(np.mean(np.abs(y[test]-prediction)<=1.96*sd))*100),
                    parity=dict(actual=y[test].tolist(), predicted=prediction.tolist(), std=sd.tolist()),
                    optimized_kernel=str(model.model.kernel_), bound_diagnostics=bound_diagnostics(model.model.kernel_, encoder.names))
            slices = []
            for j, r in enumerate(ranges):
                grid = np.linspace(r['minimum'], r['maximum'], 100).tolist() if r['type'] == 'numeric' else r['categories']
                points = np.tile(np.array(baseline, dtype=object), (len(grid), 1)); points[:, j] = grid
                encoded = encoder.transform(points)
                plot = dict(name=r['name'], type=r['type'], x=grid, train_x=raw[train, j].tolist(), train_y=y[train].tolist())
                if categorical:
                    plot.update(classes=model.classes, probabilities=model.predict_proba(encoded).tolist())
                else:
                    mean, std = model.predict(encoded, return_std=True)
                    plot.update(mean=mean.tolist(), lower=(mean-1.96*std).tolist(), upper=(mean+1.96*std).tolist())
                slices.append(plot)
            output['slices'] = slices
            outputs.append(output)
    bundle = ModelBundle(encoder, models)
    test_predictions = bundle.predict_all(raw[test], allow_unknown=True)
    exported = selected.iloc[test][features].copy()
    exported.insert(0, 'ligne_donnees', selected.index[test].to_numpy()+1)
    for name, prediction in test_predictions.items():
        exported[name+' réel'] = selected.iloc[test][name].to_numpy()
        if prediction['type'] == 'numeric':
            mean, std = prediction['mean'], prediction['std']
            exported[name+' prédit'] = mean; exported[name+' sigma'] = std
            exported[name+' IC95 bas'] = mean-1.96*std; exported[name+' IC95 haut'] = mean+1.96*std
        else:
            exported[name+' prédit'] = prediction['labels']
            for j, label in enumerate(prediction['classes']):
                exported[name+' P('+label+')'] = prediction['probabilities'][:, j]
    unknown = [entry['name'] for j, entry in enumerate(encoder.schema) if entry['type']=='categorical'
               and not set(raw[test, j]) <= set(entry['categories'])]
    if unknown:
        messages.append('Modalités absentes de l’apprentissage dans le test : '+', '.join(unknown)+'. Elles sont encodées par un vecteur nul pour cette évaluation.')
    result = dict(**outputs[0], targets=targets, outputs=outputs, features=features, input_schema=encoder.schema,
        column_types=types, input_ranges=ranges, quality=options['quality'], kernel_search=search, length_bounds=list(bounds),
        data=dict(x=raw.tolist(), y=selected[targets[0]].tolist(), y_by_target={n:selected[n].tolist() for n in targets}, train=train.tolist(), test=test.tolist()),
        train_count=len(train), test_count=len(test), dropped=len(frame)-len(selected), warnings=list(dict.fromkeys(messages)),
        seconds=perf_counter()-start, folds=len(folds), seed=2025, split_seed=split_seed)
    progress(99, 'Préparation des résultats', f'{len(targets)} sortie(s), {len(features)} entrée(s)')
    return result, exported.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig'), bundle
