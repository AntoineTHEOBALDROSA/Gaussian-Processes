"""Archives .gpmodel : JSON et tableaux numériques, sans pickle ni code exécuté.

Le posterior est conservé tel quel ; la réouverture ne lance aucun fit.
La prédiction portable sur CPU utilise le même noyau et la même Cholesky.
"""
import io
import json
import zipfile
import zlib
import numpy as np
from scipy.linalg import solve_triangular
from sklearn.gaussian_process.kernels import (
    ConstantKernel, RBF, Matern, RationalQuadratic, WhiteKernel, Sum, Product,
)

ARRAY_NAMES = ('x_train', 'coefficients', 'cholesky', 'input_mean', 'input_scale')
MEMBERS = {'manifest.json', 'predictions.csv', *(name + '.npy' for name in ARRAY_NAMES)}
MAX_ARCHIVE_BYTES = 160 * 1024 * 1024


def encode_kernel(kernel):
    kind = type(kernel).__name__
    if type(kernel) in (Sum, Product):
        return dict(type=kind, left=encode_kernel(kernel.k1), right=encode_kernel(kernel.k2))
    if type(kernel) is ConstantKernel:
        return dict(type=kind, value=float(kernel.constant_value))
    if type(kernel) is WhiteKernel:
        return dict(type=kind, value=float(kernel.noise_level))
    if type(kernel) in (RBF, Matern, RationalQuadratic):
        if type(kernel) is Matern and kernel.nu not in (.5, 1.5, 2.5):
            raise ValueError('La sauvegarde prend en charge les Matérn 1/2, 3/2 et 5/2.')
        result = dict(type=kind, length_scale=np.asarray(kernel.length_scale).tolist())
        if type(kernel) is Matern:
            result['nu'] = float(kernel.nu)
        if type(kernel) is RationalQuadratic:
            result['alpha'] = float(kernel.alpha)
        return result
    raise ValueError('Ce type de noyau ne peut pas encore être sauvegardé : ' + kind)


def decode_kernel(description, dimensions, depth=0):
    if not isinstance(description, dict) or depth > 8:
        raise ValueError('Description du noyau invalide.')
    kind = description.get('type')
    if kind in ('Sum', 'Product'):
        left = decode_kernel(description['left'], dimensions, depth + 1)
        right = decode_kernel(description['right'], dimensions, depth + 1)
        return left + right if kind == 'Sum' else left * right
    if kind in ('ConstantKernel', 'WhiteKernel'):
        value = float(description['value'])
        if not np.isfinite(value) or value <= 0:
            raise ValueError('Paramètre du noyau invalide.')
        return ConstantKernel(value, 'fixed') if kind == 'ConstantKernel' else WhiteKernel(value, 'fixed')
    if kind not in ('RBF', 'Matern', 'RationalQuadratic'):
        raise ValueError('Type de noyau non reconnu.')
    length = np.asarray(description['length_scale'], dtype=float)
    if length.ndim > 1 or length.size not in (1, dimensions) or not np.isfinite(length).all() or (length <= 0).any():
        raise ValueError('Longueurs de corrélation invalides.')
    length = float(length.reshape(-1)[0]) if length.size == 1 else length
    if kind == 'Matern':
        nu = float(description['nu'])
        if nu not in (.5, 1.5, 2.5):
            raise ValueError('Régularité Matérn non prise en charge.')
        return Matern(length, 'fixed', nu=nu)
    if kind == 'RationalQuadratic':
        alpha = float(description['alpha'])
        if np.ndim(length) or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError('Paramètres RationalQuadratic invalides.')
        return RationalQuadratic(length, alpha, 'fixed', 'fixed')
    return RBF(length, 'fixed')


class SavedModel:
    def __init__(self, kernel, arrays, y_mean, y_std):
        self.kernel = kernel
        self.arrays = arrays
        self.y_mean, self.y_std = y_mean, y_std

    def predict(self, X, return_std=False):
        a = self.arrays
        points = (np.atleast_2d(X) - a['input_mean']) / a['input_scale']
        covariance = self.kernel(points, a['x_train'])
        mean = (covariance @ a['coefficients']) * self.y_std + self.y_mean
        if not return_std:
            return mean
        v = solve_triangular(a['cholesky'], covariance.T, lower=True, check_finite=False)
        variance = self.kernel.diag(points) - np.einsum('ij,ij->j', v, v)
        return mean, np.sqrt(np.maximum(variance, 0)) * self.y_std


def save_model(job):
    model = job['model']
    if isinstance(model, SavedModel):
        arrays, kernel = model.arrays, model.kernel
    else:
        fitted = model.model
        kernel = fitted.kernel_
        arrays = dict(input_mean=model.scaler_x.mean_, input_scale=model.scaler_x.scale_)
        if hasattr(fitted, 'X_train_'):
            arrays.update(x_train=fitted.X_train_, coefficients=fitted.alpha_, cholesky=fitted.L_)
        else:
            arrays.update(x_train=fitted._X.detach().cpu().numpy(),
                          coefficients=fitted._coefficients.detach().cpu().numpy().ravel(),
                          cholesky=fitted._chol.detach().cpu().numpy())
    manifest = dict(format='processus-gaussiens', version=1, kernel=encode_kernel(kernel),
                    y_mean=model.y_mean, y_std=model.y_std,
                    dataset=job['dataset'], options=job['options'], result=job['result'])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, allow_nan=False))
        archive.writestr('predictions.csv', job['csv'])
        for name, array in arrays.items():
            output = io.BytesIO()
            np.save(output, np.asarray(array, dtype='<f8'), allow_pickle=False)
            archive.writestr(name + '.npy', output.getvalue())
    return buffer.getvalue()


def read_array(archive, name):
    raw = archive.read(name + '.npy')
    stream = io.BytesIO(raw)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError('Format numérique non reconnu.')
    if dtype.kind != 'f' or dtype.itemsize != 8 or len(shape) not in (1, 2) or any(n < 1 or n > 3000 for n in shape):
        raise ValueError('Tableau numérique invalide.')
    expected = int(np.prod(shape)) * 8
    if len(raw) - stream.tell() != expected:
        raise ValueError('Tableau numérique incomplet.')
    array = np.frombuffer(raw, dtype=dtype, offset=stream.tell()).reshape(shape, order='F' if fortran else 'C').copy()
    if not np.isfinite(array).all():
        raise ValueError('Valeurs numériques non finies.')
    return array


def validate_study(manifest, n, dimensions):
    r, d, o = manifest['result'], manifest['dataset'], manifest['options']
    features, target = r['features'], r['target']
    if (not isinstance(features, list) or len(features) != dimensions or len(set(features)) != dimensions
            or any(not isinstance(v, str) or not v for v in features)
            or not isinstance(target, str) or target in features):
        raise ValueError('Variables du modèle invalides.')
    columns = [c['name'] for c in d['columns']]
    if (not 2 <= len(columns) <= 50 or len(set(columns)) != len(columns)
            or not set(features + [target]) <= set(columns)
            or not isinstance(d['filename'], str) or not 10 <= d['rows'] <= 3000
            or len(d['records']) != d['rows'] or any(len(row) != len(columns) for row in d['records'])):
        raise ValueError('Données de l’étude invalides.')
    if (o['target'] != target or o['features'] != features or o['quality'] not in ('quick', 'standard')
            or o['device'] not in ('cpu', 'cuda') or r['device'] not in ('cpu', 'cuda')):
        raise ValueError('Configuration de l’étude invalide.')
    count = len(r['data']['x'])
    train, test = r['data']['train'], r['data']['test']
    if (not n < count <= d['rows'] or len(train) != n or len(test) != count - n
            or sorted(train + test) != list(range(count)) or len(r['data']['y']) != count
            or np.asarray(r['data']['x']).shape != (count, dimensions)
            or r['train_count'] != n or r['test_count'] != len(test)):
        raise ValueError('Partitions de l’étude invalides.')
    for field in ('actual', 'predicted', 'std'):
        if len(r['parity'][field]) != len(test) or not np.isfinite(r['parity'][field]).all():
            raise ValueError('Prédictions de test invalides.')
    if len(r['slices']) != dimensions or len(r['input_ranges']) != dimensions:
        raise ValueError('Graphiques du modèle incomplets.')
    for feature, plot, bounds in zip(features, r['slices'], r['input_ranges']):
        if plot['name'] != feature or bounds['name'] != feature or not bounds['minimum'] <= bounds['median'] <= bounds['maximum']:
            raise ValueError('Variables des graphiques invalides.')
        if not 2 <= len(plot['x']) <= 3000 or any(len(plot[key]) != len(plot['x']) for key in ('mean', 'lower', 'upper')):
            raise ValueError('Coupes du modèle invalides.')
        if len(plot['train_x']) != n or len(plot['train_y']) != n:
            raise ValueError('Observations des coupes invalides.')
    if not isinstance(r['kernel'], str) or not isinstance(r['optimized_kernel'], str) or not 1 <= len(r['ranking']) <= 50:
        raise ValueError('Résultats de sélection invalides.')
    for row in r['ranking']:
        if not isinstance(row['name'], str) or not np.isfinite([row[k] for k in ('rmse', 'std', 'seconds', 'warnings')]).all():
            raise ValueError('Classement des noyaux invalide.')
    if (not np.isfinite([r['metrics'][k] for k in ('mae', 'rmse', 'coverage')] + [r['seconds'], r['dropped'], r['folds'], r['seed']]).all()
            or not isinstance(r['warnings'], list) or any(not isinstance(w, str) for w in r['warnings'])):
        raise ValueError('Indicateurs de l’étude invalides.')


def open_model(source):
    try:
        with zipfile.ZipFile(source) as archive:
            entries = archive.infolist()
            if (len(entries) != len(MEMBERS) or {e.filename for e in entries} != MEMBERS
                    or sum(e.file_size for e in entries) > MAX_ARCHIVE_BYTES
                    or any(e.flag_bits & 1 for e in entries)):
                raise ValueError('Archive invalide ou trop volumineuse.')
            manifest = json.loads(archive.read('manifest.json'))
            if manifest.get('format') != 'processus-gaussiens' or manifest.get('version') != 1:
                raise ValueError('Version de sauvegarde non prise en charge.')
            arrays = {name: read_array(archive, name) for name in ARRAY_NAMES}
            csv = archive.read('predictions.csv')
        n, dimensions = arrays['x_train'].shape
        if (not 1 <= dimensions <= 20 or arrays['cholesky'].shape != (n, n)
                or arrays['coefficients'].shape != (n,)
                or arrays['input_mean'].shape != (dimensions,) or arrays['input_scale'].shape != (dimensions,)
                or (arrays['input_scale'] <= 0).any() or (np.diag(arrays['cholesky']) <= 0).any()
                or not np.array_equal(arrays['cholesky'], np.tril(arrays['cholesky']))):
            raise ValueError('Dimensions ou factorisation du modèle invalides.')
        mean, std = float(manifest['y_mean']), float(manifest['y_std'])
        if not np.isfinite([mean, std]).all() or std <= 0:
            raise ValueError('Normalisation de la sortie invalide.')
        validate_study(manifest, n, dimensions)
        kernel = decode_kernel(manifest['kernel'], dimensions)
        model = SavedModel(kernel, arrays, mean, std)
        # Un contrôle numérique léger, sans optimisation ni entraînement.
        value, sigma = model.predict([[b['median'] for b in manifest['result']['input_ranges']]], return_std=True)
        if not np.isfinite([value[0], sigma[0]]).all():
            raise ValueError('Prédiction de contrôle non finie.')
        return manifest, csv, model
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError, RecursionError,
            zipfile.BadZipFile, zlib.error, NotImplementedError, RuntimeError, EOFError, OSError) as exc:
        raise ValueError('Impossible d’ouvrir ce modèle. Le fichier .gpmodel est invalide, incomplet ou incompatible.') from exc
