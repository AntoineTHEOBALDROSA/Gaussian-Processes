"""Plans LHS à marges choisies, contraintes déclaratives et rejet borné."""
import numpy as np
from scipy.stats import qmc, truncnorm
from typed_models import category


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError('Nombre fini attendu pour '+label+'.')
    return float(value)


def validate_domain(domain, schema):
    if not schema or not isinstance(domain, list) or len(domain) != len(schema):
        raise ValueError('Définissez le domaine de chaque entrée.')
    normalized = []
    for entry, column in zip(domain, schema):
        if not isinstance(entry, dict) or entry.get('name') != column['name']:
            raise ValueError('Les variables du domaine ne correspondent pas aux entrées sélectionnées.')
        name = column['name']
        if column['type'] == 'categorical':
            values = entry.get('categories', column.get('categories'))
            if not isinstance(values, list) or not values or len(values) > 3000:
                raise ValueError('Choisissez les modalités autorisées pour '+name+'.')
            values = [category(v) for v in values]
            if any(not v for v in values) or len(set(values)) != len(values):
                raise ValueError('Les modalités doivent être non vides et distinctes pour '+name+'.')
            if column.get('categories') and not set(values) <= set(column['categories']):
                raise ValueError('Modalités inconnues pour '+name+'.')
            probabilities = entry.get('probabilities', [1/len(values)]*len(values))
            if not isinstance(probabilities, list) or len(probabilities) != len(values):
                raise ValueError('Indiquez un poids pour chaque modalité de '+name+'.')
            weights = np.array([finite(p, name) for p in probabilities])
            if (weights < 0).any() or not (weights>0).any():
                raise ValueError('Les poids des modalités doivent être positifs ou nuls, avec au moins un poids positif.')
            weights=weights/weights.max()
            normalized.append(dict(name=name, type='categorical', categories=values, probabilities=(weights/weights.sum()).tolist()))
            continue
        a, b = finite(entry.get('minimum'), name), finite(entry.get('maximum'), name)
        integer = entry.get('integer', False)
        if a > b or type(integer) is not bool:
            raise ValueError('Bornes ou option entiers invalides pour '+name+'.')
        if integer:
            a, b = float(np.ceil(a)), float(np.floor(b))
            if a > b:
                raise ValueError('Aucun entier dans les bornes de '+name+'.')
        law = entry.get('distribution', 'uniform')
        if law not in ('uniform', 'normal', 'loguniform'):
            raise ValueError('Choisissez une loi uniforme, normale tronquée ou log-uniforme pour '+name+'.')
        result = dict(name=name, type='numeric', minimum=a, maximum=b, integer=integer, distribution=law)
        if law == 'normal':
            result['mean'] = finite(entry.get('mean'), name+' moyenne')
            result['std'] = finite(entry.get('std'), name+' écart-type')
            if result['std'] <= 0:
                raise ValueError('L’écart-type doit être strictement positif pour '+name+'.')
        if law == 'loguniform' and a <= 0:
            raise ValueError('La loi log-uniforme nécessite des bornes strictement positives pour '+name+'.')
        normalized.append(result)
    return normalized


def constraint_mask(points, constraints, schema):
    if not isinstance(constraints, list) or len(constraints) > 100:
        raise ValueError('Les contraintes doivent former une liste de 100 conditions au maximum.')
    indices = {c['name']: j for j, c in enumerate(schema)}
    mask = np.ones(len(points), dtype=bool)
    comparisons = {'<': np.less, '<=': np.less_equal, '>': np.greater,
                   '>=': np.greater_equal, '==': np.equal, '!=': np.not_equal}

    def column(name, numeric=False):
        if not isinstance(name, str) or name not in indices:
            raise ValueError('Choisissez une entrée existante pour la contrainte.')
        j = indices[name]
        categorical = schema[j]['type'] == 'categorical'
        if numeric and categorical:
            raise ValueError('Cette contrainte nécessite des entrées numériques.')
        return (points[:, j] if categorical else np.asarray(points[:, j], dtype=float)), categorical

    for rule in constraints:
        if not isinstance(rule, dict):
            raise ValueError('Contrainte invalide.')
        kind = rule.get('type', 'comparison')
        if kind == 'linear':
            terms = rule.get('terms')
            if not isinstance(terms, list) or not terms or len(terms) > len(schema):
                raise ValueError('Ajoutez au moins un terme à la somme pondérée.')
            total = np.zeros(len(points)); names = set(); nonzero = False
            for term in terms:
                if not isinstance(term, dict):
                    raise ValueError('Terme de somme pondérée invalide.')
                name = term.get('feature')
                values, _ = column(name, numeric=True)
                if name in names:
                    raise ValueError('Chaque entrée doit figurer une seule fois dans la somme.')
                names.add(name)
                coefficient = finite(term.get('coefficient'), 'le coefficient')
                nonzero |= coefficient != 0
                total += coefficient * values
            op = rule.get('operator')
            if not nonzero or op not in ('<=', '>=', '=='):
                raise ValueError('Choisissez une somme non nulle et un comparateur valide.')
            threshold = finite(rule.get('value'), 'la limite de somme')
            if op == '==':
                tolerance = finite(rule.get('tolerance', 1e-8), 'la tolérance')
                if tolerance < 0:
                    raise ValueError('La tolérance doit être positive ou nulle.')
                mask &= np.isclose(total, threshold, rtol=0, atol=tolerance)
            else:
                mask &= comparisons[op](total, threshold)
            continue
        if kind not in ('comparison', 'interval', 'membership', 'distance'):
            raise ValueError('Type de contrainte inconnu.')
        a, categorical = column(rule.get('left'))
        op = rule.get('operator')
        if not isinstance(op, str):
            raise ValueError('Choisissez un comparateur valide.')
        if kind == 'interval':
            if categorical or op not in ('inside', 'outside'):
                raise ValueError('Un intervalle nécessite une entrée numérique et le choix inclus/exclu.')
            lower, upper = finite(rule.get('lower'), 'le minimum'), finite(rule.get('upper'), 'le maximum')
            if lower > upper:
                raise ValueError('Le minimum de l’intervalle doit être inférieur ou égal au maximum.')
            inside = (a >= lower) & (a <= upper)
            mask &= inside if op == 'inside' else ~inside
            continue
        if kind == 'membership':
            values = rule.get('values')
            if op not in ('in', 'not_in') or not isinstance(values, list) or not values or len(values) > 3000:
                raise ValueError('Choisissez au moins une valeur autorisée ou exclue.')
            values = [category(v) if categorical else finite(v, 'la valeur autorisée') for v in values]
            if categorical and any(not v for v in values):
                raise ValueError('Une catégorie de contrainte ne peut pas être vide.')
            included = np.isin(a, values)
            mask &= included if op == 'in' else ~included
            continue
        right = rule.get('right')
        if not isinstance(right, dict) or right.get('kind') not in ('feature', 'constant'):
            raise ValueError('Choisissez une variable ou une constante à droite de la contrainte.')
        if right['kind'] == 'feature':
            b, right_categorical = column(right.get('value'))
            if categorical != right_categorical:
                raise ValueError('Une contrainte entre variables doit comparer des types compatibles.')
        else:
            b = category(right.get('value')) if categorical else finite(right.get('value'), 'la constante')
        if kind == 'distance':
            threshold = finite(rule.get('value'), 'l’écart')
            if categorical or op not in ('<=', '>=') or threshold < 0:
                raise ValueError('L’écart nécessite deux valeurs numériques, une distance positive ou nulle et ≤ ou ≥.')
            mask &= comparisons[op](np.abs(a-b), threshold)
        else:
            if op not in comparisons or categorical and op not in ('==', '!='):
                raise ValueError('Comparateur invalide pour ce type de variable.')
            mask &= comparisons[op](a, b)
    return mask


def constraint_description(rule):
    """Description lisible partagée avec les exports, sans interpréter du code."""
    kind = rule.get('type', 'comparison')
    op = rule.get('operator')
    if kind == 'linear':
        expression = ' + '.join(str(t['coefficient'])+' × '+t['feature'] for t in rule['terms'])
        return expression+' '+op+' '+str(rule['value'])+(' (tolérance '+str(rule.get('tolerance', 1e-8))+')' if op == '==' else '')
    if kind == 'interval':
        return rule['left']+(' dans ' if op == 'inside' else ' hors ')+f"[{rule['lower']} ; {rule['upper']}]"
    if kind == 'membership':
        return rule['left']+(' parmi ' if op == 'in' else ' hors de ')+', '.join(map(str, rule['values']))
    right = str(rule['right']['value'])
    if kind == 'distance':
        return '|'+rule['left']+' - '+right+'| '+op+' '+str(rule['value'])
    return rule['left']+' '+op+' '+right


def lhs_sample(domain, count, seed):
    u = np.clip(qmc.LatinHypercube(len(domain), seed=seed).random(count), 1e-12, 1-1e-12)
    result = np.empty((count, len(domain)), dtype=object)
    for j, entry in enumerate(domain):
        if entry['type'] == 'categorical':
            bins = np.cumsum(entry['probabilities']); bins[-1] = 1
            result[:, j] = np.array(entry['categories'])[np.searchsorted(bins, u[:, j], side='right')]
            continue
        a, b = entry['minimum'], entry['maximum']
        if a == b:
            values = np.full(count, a)
        elif entry['distribution'] == 'normal':
            mean, std = entry['mean'], entry['std']
            values = truncnorm.ppf(u[:, j], (a-mean)/std, (b-mean)/std, loc=mean, scale=std)
        elif entry['distribution'] == 'loguniform':
            values = np.exp(np.log(a)+u[:, j]*(np.log(b)-np.log(a)))
        elif entry['integer']:
            values = np.minimum(b, np.floor(a+u[:, j]*(b-a+1)))
        else:
            values = a+u[:, j]*(b-a)
        if entry['integer']:
            values = np.clip(np.rint(values), a, b)
        if not np.isfinite(values).all():
            raise ValueError('La loi choisie produit des valeurs non finies pour '+entry['name']+'. Rapprochez la moyenne du domaine.')
        result[:, j] = values.astype(float)
    return result


def candidate_pool(options, schema, count, known=None, budget=50000):
    domain = validate_domain(options.get('domain'), schema)
    seed = options.get('seed', 2025)
    if type(seed) is not int or not 0 <= seed <= 2**32-1:
        raise ValueError('La graine doit être un entier compris entre 0 et 4 294 967 295.')
    constraints = options.get('constraints', [])
    # Validate even if the feasible set is empty.
    constraint_mask(np.empty((0,len(schema)),dtype=object), constraints, schema)
    seen = {tuple(row) for row in known} if known is not None else set()
    rows, rejected, duplicate, attempted = [], 0, 0, 0
    rounds = 0
    while len(rows) < count and attempted < budget:
        n = min(count if rounds == 0 else max(256, count-len(rows)), budget-attempted)
        points = lhs_sample(domain, n, int(np.random.SeedSequence([seed, rounds]).generate_state(1)[0]))
        mask = constraint_mask(points, constraints, schema)
        rejected += int((~mask).sum()); attempted += n
        for point in points[mask]:
            row = tuple(point)
            if row not in seen:
                rows.append(list(row)); seen.add(row)
            else:
                duplicate += 1
        rounds += 1
        if rounds >= 64:
            break
    if not rows:
        raise ValueError('Aucun point réalisable trouvé. Vérifiez les contraintes, les bornes et les modalités ; le domaine est peut-être déjà entièrement observé.')
    return np.array(rows[:count], dtype=object), dict(seed=seed, attempted=attempted, rejected_constraints=rejected,
        duplicates=duplicate, domain=domain, constraints=constraints, stratification_preserved=rounds==1 and not rejected and not duplicate and all(d['type']=='numeric' and not d['integer'] for d in domain))


def generate_design(options, schema, known=None):
    if not isinstance(options, dict):
        raise ValueError('Paramètres du plan invalides.')
    count = options.get('count', 30)
    if type(count) is not int or not 1 <= count <= 3000:
        raise ValueError('Choisissez entre 1 et 3 000 points.')
    rows, diagnostics = candidate_pool(options, schema, count, known)
    return dict(features=[s['name'] for s in schema], rows=[dict(values=row) for row in rows.tolist()],
                requested=count, **diagnostics,
                warning='Moins de points distincts et réalisables que demandé.' if len(rows)<count else '')
