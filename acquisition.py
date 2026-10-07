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


def suggestions(model, result, options):
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
