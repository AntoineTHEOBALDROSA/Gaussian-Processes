import io
import time
import unittest
import numpy as np
import pandas as pd
from app import app
from analysis import prepare_data
from unittest.mock import patch

class LocalWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def upload(self, text, filename='observations.csv'):
        return self.client.post('/api/upload', data={'file': (io.BytesIO(text.encode()), filename)})

    def test_rejects_bad_files_and_external_requests(self):
        self.assertEqual(self.upload('hello', 'test.txt').status_code, 400)
        self.assertEqual(self.upload('x,y\na,b\nc,d').status_code, 400)
        self.assertEqual(self.client.post('/api/example', headers={'Origin':'https://example.org'}).status_code, 403)
        self.assertEqual(self.client.get('/', headers={'Host':'malicious.test'}).status_code, 400)

    def test_french_csv_and_numeric_metadata(self):
        response = self.upload('ID;X;F\n1;1,5;2,5\n2;2,5;\n3;3,5;5,5')
        self.assertEqual(response.status_code, 200)
        data=response.get_json()
        self.assertEqual(data['features'], ['X'])
        self.assertEqual(data['target'], 'F')
        self.assertEqual(data['columns'][2]['invalid'], 1)
        self.assertEqual(data['records'][0][1], '1.5')
        self.assertEqual(data['numeric_data']['X'], [1.5, 2.5, 3.5])
        self.assertEqual(data['numeric_data']['F'], [2.5, None, 5.5])

    def test_duplicate_groups_never_cross_train_test(self):
        frame=pd.DataFrame({'x':np.repeat(np.arange(15),2),'y':np.repeat(np.arange(15)**2,2),'unused':[None]*30})
        clean,X,y,train,test=prepare_data(frame,'y',['x'])
        self.assertEqual(len(clean),30)
        self.assertFalse(set(X[train,0]) & set(X[test,0]))
        with self.assertRaises(ValueError):
            prepare_data(frame,'y',['y'])

    def test_complete_import_compute_and_export(self):
        x=np.linspace(-2,2,24)
        csv=pd.DataFrame({'ID':range(len(x)),'X':x,'F':np.sin(x)+x*x}).to_csv(index=False)
        imported=self.upload(csv).get_json()
        self.assertEqual(len(imported['records']), 24)
        self.assertAlmostEqual(imported['records'][-1][1], 2.0)
        options=dict(dataset_id=imported['id'], target='F',features=['X'],quality='quick',device='cpu')
        response=self.client.post('/api/analyze',json=options)
        self.assertEqual(response.status_code,202,response.get_json())
        job_id=response.get_json()['id']
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            job=self.client.get('/api/jobs/'+job_id).get_json()
            if job['status']!='running':break
            time.sleep(.1)
        self.assertEqual(job['status'],'done',job)
        result=job['result']
        self.assertEqual(len(result['ranking']),4)
        self.assertEqual(result['train_count']+result['test_count'],24)
        self.assertLess(result['metrics']['rmse'],.3)
        self.assertTrue(all(np.isfinite(result['parity']['predicted'])))
        self.assertEqual(len(result['slices']),1)
        exported=self.client.get('/api/jobs/'+job_id+'/predictions.csv')
        self.assertEqual(exported.status_code,200)
        self.assertIn('IC95_bas',exported.data.decode('utf-8-sig'))
        point=result['data']['x'][result['data']['test'][0]][0]
        with patch('models.GPRWrapper.fit', side_effect=AssertionError('No retraining')):
            prediction=self.client.post('/api/jobs/'+job_id+'/predict',json={'values':{'X':point}})
        self.assertEqual(prediction.status_code,200,prediction.get_json())
        values=prediction.get_json()
        self.assertAlmostEqual(values['mean'],result['parity']['predicted'][0],places=9)
        self.assertAlmostEqual(values['std'],result['parity']['std'][0],places=8)
        self.assertLessEqual(values['lower'],values['mean'])
        self.assertGreaterEqual(values['upper'],values['mean'])
        extrapolated=self.client.post('/api/jobs/'+job_id+'/predict',json={'values':{'X':20}}).get_json()
        self.assertEqual(extrapolated['outside'],['X'])
        for bad in [{'X':True},{'X':'abc'},{'X':float('inf')},{},{'X':0,'Y':1}]:
            self.assertEqual(self.client.post('/api/jobs/'+job_id+'/predict',json={'values':bad}).status_code,400)
        self.assertEqual(self.client.post('/api/jobs/missing/predict',json={'values':{'X':0}}).status_code,404)
        restored=self.client.get('/api/jobs/'+job_id+'?details=1').get_json()
        self.assertEqual(len(restored['dataset']['records']),24)
        self.assertNotIn('model',restored)
        self.assertNotIn('dataset',job)
        options['features']=['missing']
        self.assertEqual(self.client.post('/api/analyze',json=options).status_code,400)

if __name__=='__main__': unittest.main()
