import base64
import io
import json
import time
import unittest
import zipfile
from unittest.mock import patch
import numpy as np
import pandas as pd
from flask import Flask
from PIL import Image
from api import register_api
from models import GPRWrapper, noyaux_candidats
from model_io import open_model, save_model


def new_client():
    app = Flask(__name__)
    app.config['TESTING'] = True
    register_api(app)
    return app.test_client()


def mutate_archive(raw, member, replacement):
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as original, zipfile.ZipFile(output, 'w') as changed:
        for name in original.namelist():
            changed.writestr(name, replacement if name == member else original.read(name))
    return output.getvalue()


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x = np.linspace(-2, 2, 24)
        cls.frame = pd.DataFrame({'ID':range(24), 'X':x, 'X2':x**2+.1*x, 'F':np.sin(x)+x*x})
        cls.client = new_client()
        uploaded = cls.client.post('/api/upload', data={'file':(io.BytesIO(cls.frame.to_csv(index=False).encode()), 'essai.csv')}).get_json()
        cls.options = dict(dataset_id=uploaded['id'], features=['X','X2'], target='F', quality='quick', device='cpu')
        response = cls.client.post('/api/analyze', json=cls.options)
        cls.job_id = response.get_json()['id']
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            cls.job = cls.client.get('/api/jobs/'+cls.job_id+'?details=1').get_json()
            if cls.job['status'] != 'running':
                break
            time.sleep(.05)
        assert cls.job['status'] == 'done', cls.job
        cls.archive = cls.client.get('/api/jobs/'+cls.job_id+'/model.gpmodel').data

    def test_reopen_after_restart_without_training_and_resave(self):
        # A fresh server has neither the old jobs nor the old dataset.
        restarted = new_client()
        self.assertEqual(restarted.get('/api/jobs/'+self.job_id).status_code, 404)
        with patch('models.GPRWrapper.fit', side_effect=AssertionError('No fit on reopening')), \
             patch('sklearn.gaussian_process.GaussianProcessRegressor.fit', side_effect=AssertionError('No fit')):
            response = restarted.post('/api/models/open', data={'file':(io.BytesIO(self.archive),'essai.gpmodel')})
            self.assertEqual(response.status_code, 200, response.get_json())
            restored = response.get_json()
            self.assertEqual(restored['result'], self.job['result'])
            self.assertEqual(restored['dataset']['records'], self.job['dataset']['records'])
            self.assertEqual(restored['options']['features'], ['X','X2'])
            self.assertEqual(restored['options']['target'], 'F')
            self.assertTrue(restored['restored'])
            self.assertNotEqual(restored['id'], self.job_id)
            for values in ({'X':.13,'X2':.7},{'X':10.,'X2':20.}):
                before = self.client.post('/api/jobs/'+self.job_id+'/predict', json={'values':values}).get_json()
                after = restarted.post('/api/jobs/'+restored['id']+'/predict', json={'values':values}).get_json()
                for field in ('mean', 'std', 'lower', 'upper'):
                    self.assertAlmostEqual(before[field], after[field], places=10)
                self.assertEqual(before['outside'], after['outside'])
            csv = restarted.get('/api/jobs/'+restored['id']+'/predictions.csv')
            self.assertEqual(csv.data, self.client.get('/api/jobs/'+self.job_id+'/predictions.csv').data)
            saved_again = restarted.get('/api/jobs/'+restored['id']+'/model.gpmodel')
            self.assertEqual(saved_again.status_code, 200)
            third = new_client().post('/api/models/open', data={'file':(io.BytesIO(saved_again.data),'copie.gpmodel')})
            self.assertEqual(third.status_code, 200, third.get_json())

    def test_all_four_kernels_preserve_mean_and_uncertainty(self):
        manifest, csv, _ = open_model(io.BytesIO(self.archive))
        train = manifest['result']['data']['train']
        X = self.frame[['X','X2']].to_numpy()[train]
        y = self.frame['F'].to_numpy()[train]
        points = np.array([[.123,.456],[2.1,4.3],[-2.,3.8]])
        for name, kernel in noyaux_candidats(2).items():
            with self.subTest(kernel=name):
                model = GPRWrapper(kernel, 0).fit(X, y)
                fixture = dict(model=model, csv=csv, dataset=manifest['dataset'], options=manifest['options'], result=manifest['result'])
                _, _, restored = open_model(io.BytesIO(save_model(fixture)))
                before = model.predict(points, return_std=True)
                after = restored.predict(points, return_std=True)
                np.testing.assert_allclose(after[0], before[0], atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(after[1], before[1], atol=1e-10, rtol=1e-9)

    def test_corrupted_unsupported_and_object_archives_are_rejected(self):
        with zipfile.ZipFile(io.BytesIO(self.archive)) as archive:
            manifest = json.loads(archive.read('manifest.json'))
        manifest['version'] = 999
        unsupported = mutate_archive(self.archive, 'manifest.json', json.dumps(manifest))
        objects = io.BytesIO(); np.save(objects, np.array([{'unsafe':'object'}], dtype=object))
        bad_array = mutate_archive(self.archive, 'coefficients.npy', objects.getvalue())
        for raw in (b'not a model', unsupported, bad_array, self.archive[:100]):
            response = new_client().post('/api/models/open', data={'file':(io.BytesIO(raw),'broken.gpmodel')})
            self.assertEqual(response.status_code, 400, response.get_json())
        self.assertEqual(new_client().post('/api/models/open').status_code, 400)
        self.assertEqual(self.client.get('/api/jobs/missing/model.gpmodel').status_code, 404)
        self.assertEqual(new_client().post('/api/models/open', data={'file':(io.BytesIO(self.archive),'model.pkl')}).status_code, 400)

    def test_pdf_contains_multiple_landscape_pages_and_rejects_bad_images(self):
        buffer = io.BytesIO()
        Image.new('RGB', (1200, 700), '#316bd1').save(buffer, format='PNG')
        image = 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()
        pages = [{'title':'Valeurs réelles et prédites', 'description':'Test indépendant · F', 'image':image},
                 {'title':'Matrice des entrées', 'description':'Moyenne et variance', 'image':image}]
        response = self.client.post('/api/export/pdf', json={'pages':pages})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'application/pdf')
        self.assertTrue(response.data.startswith(b'%PDF-'))
        self.assertEqual(response.data.count(b'/Type /Page\n'), 2)
        self.assertIn(b'attachment;', response.headers['Content-Disposition'].encode())
        for payload in ({}, {'pages':[]}, {'pages':[{'title':'Bad','image':'https://example.org'}]},
                        {'pages':[{'title':'Bad','image':'data:image/png;base64,not-an-image'}]}):
            self.assertEqual(self.client.post('/api/export/pdf', json=payload).status_code, 400)
        self.assertEqual(self.client.post('/api/export/pdf', json={'pages':pages}, headers={'Origin':'https://example.org'}).status_code, 403)


if __name__ == '__main__':
    unittest.main()
