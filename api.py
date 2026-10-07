"""Fichiers et résultats en mémoire, un seul calcul à la fois."""
import io
import json
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit
import numpy as np
import pandas as pd
from flask import jsonify, request, Response
from werkzeug.exceptions import HTTPException
from analysis import prepare_data, run_analysis
from model_io import save_model, open_model
from pdf_export import make_pdf

MAX_ROWS = 3000


def read_frame(source, filename):
    suffix = Path(filename).suffix.lower()
    if suffix not in ('.csv', '.xlsx'):
        raise ValueError('Format non pris en charge. Choisissez un fichier CSV ou Excel (.xlsx).')
    try:
        if suffix == '.xlsx':
            frame = pd.read_excel(source, nrows=MAX_ROWS + 1)
        else:
            raw = source.read()
            try:
                content = raw.decode('utf-8-sig')
            except UnicodeDecodeError:
                content = raw.decode('cp1252')
            frame = pd.read_csv(io.StringIO(content), sep=None, engine='python', nrows=MAX_ROWS + 1)
            # Les CSV français utilisent généralement ; et une virgule décimale.
            for column in frame.select_dtypes(include=['object', 'str']).columns:
                frame[column] = frame[column].map(lambda v: v.strip().replace(',', '.') if isinstance(v, str) else v)
    except Exception as exc:
        raise ValueError('Impossible de lire ce fichier. Vérifiez son format et la présence des noms de colonnes sur la première ligne.') from exc
    if frame.empty or len(frame.columns) < 2:
        raise ValueError('Le fichier doit contenir des données et au moins deux colonnes.')
    if len(frame) > MAX_ROWS or len(frame.columns) > 50:
        raise ValueError('Cette interface accepte jusqu’à 3 000 lignes et 50 colonnes par fichier.')
    frame.columns = [str(c) for c in frame.columns]
    if len(set(frame.columns)) != len(frame.columns):
        raise ValueError('Les noms de colonnes doivent être distincts.')
    return frame.reset_index(drop=True)


def describe(frame, filename, dataset_id):
    columns, numeric_data = [], {}
    for c in frame.columns:
        converted = pd.to_numeric(frame[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
        numeric_data[c] = json.loads(converted.to_json(orient='values'))
        columns.append(dict(name=c, numeric=bool(converted.notna().any()), invalid=int(converted.isna().sum())))
    numeric = [c['name'] for c in columns if c['numeric']]
    target = 'F' if 'F' in numeric else (numeric[-1] if numeric else None)
    def is_id(name):
        return name.lower().strip() in ('id', 'index', 'num', 'numero', 'numéro', 'n°') or name.lower().startswith('unnamed:')
    features = [c for c in numeric if c != target and not is_id(c)]
    return dict(id=dataset_id, filename=filename, rows=len(frame), columns=columns, target=target,
                features=features[:20], records=json.loads(frame.to_json(orient='values')), numeric_data=numeric_data)


def register_api(app):
    datasets, jobs = OrderedDict(), OrderedDict()
    lock = threading.RLock()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='gp-analysis')
    capability_cache = None

    @app.before_request
    def local_origin():
        request.max_content_length = (100 if request.path in ('/api/models/open', '/api/export/pdf') else 20) * 1024 * 1024
        if request.method == 'POST':
            origin = request.headers.get('Origin')
            if origin and urlsplit(origin).netloc != request.host:
                return jsonify(error='Cette action doit être lancée depuis l’interface locale.'), 403
            if request.headers.get('Sec-Fetch-Site') == 'cross-site':
                return jsonify(error='Requête externe refusée.'), 403

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        response.headers['X-Frame-Options'] = 'DENY'
        if request.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.errorhandler(413)
    def too_large(_):
        limit = 100 if request.path in ('/api/models/open', '/api/export/pdf') else 20
        return jsonify(error=f'Le fichier dépasse la limite de {limit} Mo.'), 413

    @app.errorhandler(ValueError)
    def invalid(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code

    @app.get('/api/capabilities')
    def capabilities():
        nonlocal capability_cache
        if capability_cache is None:
            capability_cache = dict(cuda=False, gpu=None)
            try:
                import torch
                if torch.cuda.is_available():
                    torch.linalg.cholesky(torch.eye(2, device='cuda', dtype=torch.float64))
                    capability_cache = dict(cuda=True, gpu=torch.cuda.get_device_name())
            except Exception:
                pass
        return jsonify(**capability_cache)

    def add_dataset(frame, filename):
        dataset_id = uuid.uuid4().hex
        metadata = describe(frame, filename, dataset_id)
        if sum(c['numeric'] for c in metadata['columns']) < 2:
            raise ValueError('Au moins deux colonnes doivent contenir des valeurs numériques.')
        with lock:
            datasets[dataset_id] = (frame, metadata)
            while len(datasets) > 10:
                datasets.popitem(last=False)
        return jsonify(metadata)

    @app.post('/api/upload')
    def upload():
        uploaded = request.files.get('file')
        if uploaded is None or not uploaded.filename:
            raise ValueError('Choisissez un fichier à importer.')
        filename = uploaded.filename.replace('\\', '/').split('/')[-1]
        return add_dataset(read_frame(uploaded.stream, filename), filename)

    @app.post('/api/example')
    def example():
        path = Path(__file__).resolve().parent.parent / 'data.xlsx'
        if not path.is_file():
            raise ValueError('Le fichier data.xlsx du projet est absent. Importez votre propre fichier.')
        with path.open('rb') as stream:
            return add_dataset(read_frame(stream, path.name), path.name)

    @app.get('/api/datasets/<dataset_id>')
    def dataset(dataset_id):
        with lock:
            entry = datasets.get(dataset_id)
            if entry is None:
                return jsonify(error='Ce fichier n’est plus en mémoire. Importez-le à nouveau.'), 404
            return jsonify(entry[1])

    def worker(job_id, frame, options):
        def progress(percent, title, detail):
            with lock:
                jobs[job_id].update(progress=percent, title=title, detail=detail)
        try:
            result, csv, model = run_analysis(frame, options, progress)
            with lock:
                jobs[job_id].update(status='done', progress=100, title='Analyse terminée', result=result, csv=csv, model=model)
        except Exception as exc:
            app.logger.exception('Échec du calcul local')
            with lock:
                jobs[job_id].update(status='error', error=f'Le calcul n’a pas abouti : {exc}')

    @app.post('/api/analyze')
    def analyze():
        options = request.get_json(silent=True)
        if not isinstance(options, dict):
            raise ValueError('Paramètres d’analyse invalides.')
        if options.get('quality') not in ('quick', 'standard') or options.get('device') not in ('cpu', 'cuda'):
            raise ValueError('Choisissez un mode de recherche et un moteur de calcul valides.')
        dataset_id = options.get('dataset_id')
        if not isinstance(dataset_id, str):
            raise ValueError('Importez un fichier avant de lancer le calcul.')
        if options['device'] == 'cuda':
            if not capabilities().get_json()['cuda']:
                raise ValueError('CUDA n’est pas disponible dans cet environnement Python. Choisissez CPU.')
        with lock:
            if any(j['status'] == 'running' for j in jobs.values()):
                return jsonify(error='Une analyse est déjà en cours. Attendez sa fin avant d’en relancer une.'), 409
            entry = datasets.get(dataset_id)
            if entry is None:
                raise ValueError('Ce fichier n’est plus en mémoire. Importez-le à nouveau.')
            frame = entry[0]
            prepare_data(frame, options.get('target'), options.get('features'))
            job_id = uuid.uuid4().hex
            jobs[job_id] = dict(id=job_id, status='running', progress=0, title='Préparation', detail='Initialisation du modèle', started=time.time(), dataset=entry[1], options=options)
            while len(jobs) > 10:
                jobs.popitem(last=False)
            executor.submit(worker, job_id, frame, options)
        return jsonify(id=job_id), 202

    @app.get('/api/jobs/<job_id>')
    def job(job_id):
        with lock:
            job = jobs.get(job_id)
            if job is None:
                return jsonify(error='Cette analyse n’est plus disponible. Relancez-la depuis vos données.'), 404
            excluded = {'csv', 'model'}
            if request.args.get('details') != '1':
                excluded.add('dataset')
            return jsonify({k: v for k, v in job.items() if k not in excluded})

    @app.get('/api/jobs/<job_id>/predictions.csv')
    def download(job_id):
        with lock:
            job = jobs.get(job_id)
            if job is None or job['status'] != 'done':
                return jsonify(error='Les prédictions ne sont pas encore disponibles.'), 404
            return Response(job['csv'], mimetype='text/csv', headers={'Content-Disposition': 'attachment; filename=predictions.csv'})

    @app.post('/api/jobs/<job_id>/predict')
    def predict(job_id):
        with lock:
            job = jobs.get(job_id)
            if job is None:
                return jsonify(error='Ce modèle n’est plus disponible. Relancez l’analyse.'), 404
            if job['status'] != 'done':
                return jsonify(error='Attendez la fin de l’analyse avant de prédire un point.'), 409
            model, result = job['model'], job['result']
        payload = request.get_json(silent=True)
        values = payload.get('values') if isinstance(payload, dict) else None
        features = result['features']
        if not isinstance(values, dict) or set(values) != set(features):
            raise ValueError('Renseignez exactement les variables d’entrée du modèle : ' + ', '.join(features))
        if any(isinstance(values[c], bool) or not isinstance(values[c], (int, float)) for c in features):
            raise ValueError('Chaque entrée doit être un nombre fini.')
        try:
            point = np.array([[values[c] for c in features]], dtype=float)
        except (OverflowError, TypeError):
            raise ValueError('Chaque entrée doit être un nombre fini.')
        if not np.isfinite(point).all():
            raise ValueError('Chaque entrée doit être un nombre fini.')
        try:
            mean, std = model.predict(point, return_std=True)
        except (ValueError, OverflowError, FloatingPointError):
            raise ValueError('Ces valeurs ne permettent pas un calcul numérique stable. Vérifiez leur échelle.')
        mu, sigma = float(mean[0]), float(std[0])
        if not np.isfinite([mu, sigma, mu - 1.96*sigma, mu + 1.96*sigma]).all():
            raise ValueError('La prédiction n’est pas finie. Vérifiez l’échelle des entrées.')
        outside = [r['name'] for r in result['input_ranges'] if not r['minimum'] <= values[r['name']] <= r['maximum']]
        return jsonify(target=result['target'], mean=mu, std=sigma, lower=mu-1.96*sigma, upper=mu+1.96*sigma, outside=outside)

    @app.get('/api/jobs/<job_id>/model.gpmodel')
    def download_model(job_id):
        with lock:
            entry = jobs.get(job_id)
            if entry is None or entry['status'] != 'done':
                return jsonify(error='Le modèle entraîné n’est pas disponible.'), 404
        return Response(save_model(entry), mimetype='application/octet-stream',
                        headers={'Content-Disposition': 'attachment; filename=modele.gpmodel'})

    @app.post('/api/models/open')
    def restore_model():
        uploaded = request.files.get('file')
        if uploaded is None or not uploaded.filename or not uploaded.filename.lower().endswith('.gpmodel'):
            raise ValueError('Choisissez une sauvegarde de modèle au format .gpmodel.')
        manifest, csv, model = open_model(uploaded.stream)
        data = manifest['dataset']
        frame = pd.DataFrame(data['records'], columns=[c['name'] for c in data['columns']])
        dataset_id, job_id = uuid.uuid4().hex, uuid.uuid4().hex
        metadata = describe(frame, data['filename'], dataset_id)
        options = {**manifest['options'], 'dataset_id': dataset_id, 'device': 'cpu'}
        metadata.update(target=options['target'], features=options['features'])
        entry = dict(id=job_id, status='done', progress=100, title='Modèle rouvert', detail='',
                     started=time.time(), dataset=metadata, options=options, result=manifest['result'],
                     csv=csv, model=model, restored=True)
        with lock:
            if any(j['status'] == 'running' for j in jobs.values()):
                return jsonify(error='Attendez la fin de l’analyse avant d’ouvrir un modèle.'), 409
            datasets[dataset_id] = (frame, metadata)
            jobs[job_id] = entry
            while len(datasets) > 10:
                datasets.popitem(last=False)
            while len(jobs) > 10:
                jobs.popitem(last=False)
        return jsonify({k: v for k, v in entry.items() if k not in ('csv', 'model')})

    @app.post('/api/export/pdf')
    def export_pdf():
        return Response(make_pdf(request.get_json(silent=True)), mimetype='application/pdf',
                        headers={'Content-Disposition': 'attachment; filename=graphiques.pdf'})
