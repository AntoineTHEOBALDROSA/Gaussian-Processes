"""Enrichissement exploratoire : distance, incertitude du GP et diversité du lot.

Le domaine fourni est une boîte ; sa faisabilité physique doit être vérifiée.
Les sorties proposées sont des estimations, jamais de nouvelles observations.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc, rankdata
from sklearn.gaussian_process.kernels import WhiteKernel, Sum, Product, Exponentiation
from threadpoolctl import threadpool_limits


def noise_diagonal(kernel, X):
    if isinstance(kernel, WhiteKernel):
        return kernel.diag(X)
    if isinstance(kernel, Sum):
        return noise_diagonal(kernel.k1, X) + noise_diagonal(kernel.k2, X)
    if isinstance(kernel, Product):
        d1, d2 = kernel.k1.diag(X), kernel.k2.diag(X)
        return d1*d2 - (d1-noise_diagonal(kernel.k1,X))*(d2-noise_diagonal(kernel.k2,X))
    if isinstance(kernel, Exponentiation):
        diag = kernel.kernel.diag(X)
        return diag**kernel.exponent - (diag-noise_diagonal(kernel.kernel,X))**kernel.exponent
    return np.zeros(len(X))


def latent_predictions(model, X):
    mean, std = model.predict(X, return_std=True)
    if hasattr(model, 'arrays'):
        scaled = (X-model.arrays['input_mean'])/model.arrays['input_scale']
        kernel = model.kernel
    else:
        scaled = model.scaler_x.transform(X)
        kernel = model.model.kernel_
    latent_std = np.sqrt(np.maximum(std**2-noise_diagonal(kernel,scaled)*model.y_std**2, 0))
    return mean, std, latent_std


def legacy_suggestions(model, result, options):
    if not isinstance(options, dict):
        raise ValueError('Paramètres de recherche invalides.')
    count = options.get('count', 10)
    if type(count) is not int or not 1 <= count <= 30:
        raise ValueError('Choisissez entre 1 et 30 nouvelles simulations.')
    domain = options.get('domain')
    features = result['features']
    if not isinstance(domain, list) or len(domain) != len(features):
        raise ValueError('Indiquez les bornes de chaque entrée.')
    low, high, discrete = [], [], []
    for name, entry in zip(features, domain):
        if not isinstance(entry, dict) or entry.get('name') != name:
            raise ValueError('Les variables du domaine doivent correspondre au modèle.')
        a, b, integer = entry.get('minimum'), entry.get('maximum'), entry.get('integer', False)
        if (any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in (a,b))
                or not np.isfinite([a,b]).all() or a > b or type(integer) is not bool):
            raise ValueError('Bornes invalides pour '+name+'.')
        if integer:
            a, b = np.ceil(a), np.floor(b)
            if a > b:
                raise ValueError('Aucun entier dans les bornes de '+name+'.')
        low.append(a); high.append(b); discrete.append(integer)
    low, high = np.array(low), np.array(high)
    U = qmc.LatinHypercube(len(features), seed=2025).random(2048)
    candidates = low + U*(high-low)
    for j, integer in enumerate(discrete):
        if integer:
            candidates[:,j] = np.minimum(high[j], np.floor(low[j]+U[:,j]*(high[j]-low[j]+1)))
    candidates = np.unique(candidates, axis=0)
    known = np.unique(np.asarray(result['data']['x'],dtype=float), axis=0)
    center, scale = known.mean(axis=0), known.std(axis=0)
    scale[scale==0] = 1
    tree = cKDTree((known-center)/scale)
    scaled = (candidates-center)/scale
    distance = tree.query(scaled, k=1)[0]
    mask = distance > 1e-7
    candidates, scaled, distance = candidates[mask], scaled[mask], distance[mask]
    if not len(candidates):
        raise ValueError('Aucune nouvelle configuration dans ce domaine : les points candidats existent déjà.')
    means, stds, latent = [], [], []
    with threadpool_limits(limits=1):
        for start in range(0,len(candidates),128):
            m,s,l = latent_predictions(model, candidates[start:start+128])
            means.extend(m); stds.extend(s); latent.extend(l)
    means, stds, latent = np.asarray(means), np.asarray(stds), np.asarray(latent)
    if not np.isfinite(np.concatenate([means,stds,latent,distance])).all():
        raise ValueError('Ce domaine produit des prédictions non finies. Resserrez les bornes.')
    focus = options.get('target_range')
    if focus is not None:
        if (not isinstance(focus,list) or len(focus)!=2
                or any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in focus)
                or not np.isfinite(focus).all() or focus[0] >= focus[1]):
            raise ValueError('Indiquez deux bornes croissantes pour la plage de sortie souhaitée.')
        mask = (means >= focus[0]) & (means <= focus[1])
        candidates, scaled, distance, means, stds, latent = [v[mask] for v in (candidates,scaled,distance,means,stds,latent)]
        if not len(candidates):
            raise ValueError('Aucun candidat avec une sortie prédite dans cette plage. Élargissez la plage ou le domaine.')
    # Rangs : aucune dépendance aux unités de la cible ou aux échelles d'entrée.
    score = .5*(rankdata(distance)/len(distance) + rankdata(latent)/len(latent))
    diversity = np.full(len(score), np.inf)
    selected = []
    for _ in range(min(count,len(score))):
        priority = score*np.minimum(1.,diversity/(.5*np.sqrt(len(features))))
        if selected:
            priority[selected] = -1
        index = int(np.argmax(priority))
        selected.append(index)
        diversity = np.minimum(diversity, np.linalg.norm(scaled-scaled[index],axis=1))
    ranges = result['input_ranges']
    rows = [dict(values=candidates[i].tolist(), mean=float(means[i]), std=float(stds[i]),
                 latent_std=float(latent[i]), distance=float(distance[i]), score=float(score[i]),
                 extrapolation=any(not r['minimum'] <= candidates[i,j] <= r['maximum'] for j,r in enumerate(ranges)))
            for i in selected]
    return dict(features=features, target=result['target'], rows=rows, candidates=len(candidates),
                known_count=len(known), seed=2025, target_range=focus,
                method='Distance aux observations et incertitude de la fonction, puis diversification du lot.')


def suggestions(model, result, options):
    from typed_models import ModelBundle, InputEncoder
    from design import candidate_pool
    if not isinstance(options, dict):
        raise ValueError('Paramètres de recherche invalides.')
    if not isinstance(model, ModelBundle) and not any(k in options for k in ('constraints', 'seed', 'pool_size', 'acquisition_target')) and not any('distribution' in d for d in options.get('domain', [])):
        return legacy_suggestions(model, result, options)
    count, size = options.get('count', 10), options.get('pool_size', 4096)
    if type(count) is not int or not 1 <= count <= 200 or type(size) is not int or not count <= size <= 32768:
        raise ValueError('Choisissez 1 à 200 simulations et un réservoir de candidats entre ce nombre et 32 768.')
    schema = model.encoder.schema if isinstance(model, ModelBundle) else [dict(name=n, type='numeric') for n in result['features']]
    known = np.array(result['data']['x'], dtype=object)
    candidates, diagnostics = candidate_pool(options, schema, size, known)
    encoder = InputEncoder(schema).fit(known)
    known_e, candidate_e = encoder.transform(known), encoder.transform(candidates)
    center, scale = known_e.mean(0), known_e.std(0); scale[scale==0] = 1
    scaled = (candidate_e-center)/scale
    distance = cKDTree((known_e-center)/scale).query(scaled, k=1)[0]
    valid = distance > 1e-7
    candidates, scaled, distance = candidates[valid], scaled[valid], distance[valid]
    if not len(candidates):
        raise ValueError('Aucune nouvelle configuration distincte des observations dans ce domaine.')
    predictions, uncertainties = {}, {}
    target = options.get('acquisition_target')
    available = list(model.outputs) if isinstance(model, ModelBundle) else [result['target']]
    if target is not None and target not in available:
        raise ValueError('Choisissez une sortie du modèle pour la priorité.')
    with threadpool_limits(limits=1):
        for name in available:
            batches = []
            for start in range(0, len(candidates), 128):
                points = candidates[start:start+128]
                if isinstance(model, ModelBundle):
                    encoded = model.encoder.transform(points)
                    entry = model.outputs[name]
                    if entry['type'] == 'categorical':
                        probabilities = entry['model'].predict_proba(encoded)
                        entropy = -(probabilities*np.log(probabilities)).sum(1)/np.log(probabilities.shape[1])
                        batches.append((entry['type'], probabilities, entropy))
                    else:
                        m, s, latent = latent_predictions(entry['model'], encoded)
                        batches.append(('numeric', m, s, latent))
                else:
                    m, s, latent = latent_predictions(model, np.asarray(points,dtype=float))
                    batches.append(('numeric', m, s, latent))
            if batches[0][0] == 'categorical':
                probs = np.concatenate([b[1] for b in batches]); uncertainty = np.concatenate([b[2] for b in batches])
                classes = model.outputs[name]['model'].classes
                predictions[name] = dict(type='categorical', probabilities=probs, classes=classes,
                                          labels=np.array(classes)[probs.argmax(1)])
            else:
                predictions[name] = dict(type='numeric', mean=np.concatenate([b[1] for b in batches]),
                    std=np.concatenate([b[2] for b in batches]), latent_std=np.concatenate([b[3] for b in batches]))
                uncertainty = predictions[name]['latent_std']
            if not np.isfinite(uncertainty).all():
                raise ValueError('Le domaine produit des incertitudes non finies. Resserrez les bornes.')
            uncertainties[name] = uncertainty
    selected_target = target or available[0]
    focus = options.get('target_range')
    desired = options.get('target_class')
    if focus is not None or desired is not None:
        p = predictions[selected_target]
        if focus is not None:
            from design import finite
            if p['type'] != 'numeric' or not isinstance(focus,list) or len(focus)!=2:
                raise ValueError('Une plage de sortie nécessite une sortie numérique et deux bornes.')
            a,b = [finite(v,'la plage de sortie') for v in focus]
            if a>=b: raise ValueError('Les bornes de sortie doivent être croissantes.')
            mask = (p['mean']>=a)&(p['mean']<=b)
        else:
            if p['type'] != 'categorical' or desired not in p['classes']:
                raise ValueError('Classe de sortie souhaitée invalide.')
            mask = p['labels']==desired
        candidates, scaled, distance = candidates[mask], scaled[mask], distance[mask]
        if not len(candidates): raise ValueError('Aucun candidat avec une sortie prédite dans la cible demandée.')
        for name in available:
            uncertainties[name] = uncertainties[name][mask]
            for key in ('mean','std','latent_std','labels','probabilities'):
                if key in predictions[name]: predictions[name][key] = predictions[name][key][mask]
    normalized = np.mean([rankdata(uncertainties[n])/len(candidates) for n in ([target] if target else available)], axis=0)
    score = .5*(rankdata(distance)/len(candidates)+normalized)
    diversity = np.full(len(candidates), np.inf); selected=[]
    for _ in range(min(count,len(candidates))):
        priority = score*np.minimum(1.,diversity/(.5*np.sqrt(len(schema))))
        priority[selected] = -1
        index = int(np.argmax(priority)); selected.append(index)
        diversity = np.minimum(diversity, np.linalg.norm(scaled-scaled[index],axis=1))
    rows=[]
    for i in selected:
        row = dict(values=candidates[i].tolist(), distance=float(distance[i]), score=float(score[i]), uncertainty=float(normalized[i]), predictions={})
        for name,p in predictions.items():
            if p['type']=='numeric':
                row['predictions'][name] = dict(type='numeric', mean=float(p['mean'][i]), std=float(p['std'][i]), latent_std=float(p['latent_std'][i]))
            else:
                row['predictions'][name] = dict(type='categorical', label=str(p['labels'][i]),
                    probabilities={c:float(p['probabilities'][i,j]) for j,c in enumerate(p['classes'])})
        primary=row['predictions'][selected_target]
        if primary['type']=='numeric': row.update({k:primary[k] for k in ('mean','std','latent_std')})
        row['extrapolation'] = any(r.get('type','numeric')=='numeric' and not r['minimum'] <= float(candidates[i,j]) <= r['maximum'] for j,r in enumerate(result['input_ranges']))
        rows.append(row)
    return dict(features=result['features'], targets=available, target=selected_target, rows=rows, candidates=len(candidates),
                known_count=len({tuple(r) for r in known}), requested=count, acquisition_target=target, **diagnostics,
                warning='Moins de candidats réalisables que le nombre demandé.' if len(rows)<count else '',
                method='Distance et rang moyen des incertitudes (écart-type latent ou entropie), puis diversité.')
