import io
import json
import time
import unittest
import warnings
import zipfile
from unittest.mock import patch
import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from scipy.stats import truncnorm
from api import describe
from design import generate_design, lhs_sample, validate_domain
from study import run_study, prepare_study
from model_io import save_model, open_model
from acquisition import suggestions
from excel_prediction import predict_workbook
from report_export import make_report
from test_exports import new_client, mutate_archive


class MixedStudiesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x=np.linspace(-3,3,36)
        cls.frame=pd.DataFrame(dict(X=x,Z=np.cos(x),Matiere=['acier','alu']*18,
            Force=np.sin(x)+np.tile([.1,0],18),Etat=np.where(x<-1,'faible',np.where(x>1,'fort','moyen')),Rendement=x*x))
        cls.options=dict(features=['X','Z','Matiere'],targets=['Force','Etat','Rendement'],target='Force',quality='quick',device='cpu')
        cls.result,cls.csv,cls.model=run_study(cls.frame,cls.options,lambda *args:None)
        cls.dataset=describe(cls.frame,'mixte.csv','fixture')
        cls.job=dict(result=cls.result,csv=cls.csv,model=cls.model,dataset=cls.dataset,options=cls.options)
        cls.archive=save_model(cls.job)

    def test_three_outputs_include_real_classification_and_regression(self):
        r=self.result
        self.assertEqual(r['targets'],['Force','Etat','Rendement'])
        self.assertEqual(len(r['outputs']),3)
        classifier=r['outputs'][1]
        self.assertEqual(classifier['type'],'categorical')
        self.assertEqual(len(classifier['classes']),3)
        self.assertEqual(len(classifier['kernel_variants']),3)
        self.assertTrue(all(v['bound_diagnostics'] for v in classifier['kernel_variants']))
        p=np.array(classifier['parity']['probabilities'])
        np.testing.assert_allclose(p.sum(1),1,atol=1e-12)
        self.assertTrue(np.isfinite(p).all());self.assertTrue(((p>=0)&(p<=1)).all())
        self.assertGreater(classifier['metrics']['accuracy'],50)
        self.assertEqual(len(classifier['confusion']),3)
        self.assertTrue(all(len(o['ranking'])==4 for o in r['outputs']))

    def test_no_fixed_limit_on_input_columns(self):
        f=pd.DataFrame({f'X{i}':np.arange(20)*(i+1) for i in range(55)})
        f['Y']=np.arange(20)**2
        options=dict(features=list(f.columns[:-1]),targets=['Y'],quality='quick')
        selected,_,_,_,_,_=prepare_study(f,options)
        self.assertEqual(len(selected.columns),56)
        self.assertEqual(len(describe(f,'large.csv','id')['features']),55)
        response=new_client().post('/api/upload',data={'file':(io.BytesIO(f.to_csv(index=False).encode()),'large.csv')})
        self.assertEqual(response.status_code,200,response.get_json())

    def test_types_and_duplicates_do_not_leak_across_the_split(self):
        frame=pd.concat([self.frame,self.frame],ignore_index=True)
        selected,types,train,test,partitions,_=prepare_study(frame,self.options)
        self.assertEqual(types['Matiere'],'categorical')
        self.assertEqual(types['Etat'],'categorical')
        def configurations(ids):return {tuple(row) for row in selected.iloc[ids][self.options['features']].to_numpy()}
        self.assertFalse(configurations(train)&configurations(test))
        for a,b in partitions:self.assertFalse(configurations(a)&configurations(b))
        for change in (dict(targets=['Force','Force']),dict(features=['Force']),dict(column_types={'Etat':'unknown'})):
            with self.assertRaises(ValueError):prepare_study(self.frame,dict(self.options,**change))

    def test_numeric_category_codes_and_all_categorical_inputs(self):
        f=pd.DataFrame({'A':np.tile([1,2,3,4],12),'B':np.repeat([1,2,3],16),'C':np.tile(['oui','non'],24)})
        options=dict(features=['A','B'],targets=['C'],column_types={'A':'categorical','B':'categorical'},quality='quick',device='cpu')
        r,_,m=run_study(f,options,lambda *a:None)
        self.assertEqual(r['input_schema'][0]['categories'],['1','2','3','4'])
        self.assertIn(m.predict_all([[1,2]])['C']['labels'][0],['oui','non'])

    def test_portable_mixed_archive_never_retrains(self):
        points=np.array([[.3,.5,'acier'],[4.,.3,'alu']],dtype=object)
        before=self.model.predict_all(points)
        with patch('typed_models.GaussianClassifier.fit',side_effect=AssertionError('No fitting')),patch('models.GPRWrapper.fit',side_effect=AssertionError('No fitting')):
            manifest,_,model=open_model(io.BytesIO(self.archive));after=model.predict_all(points)
        self.assertEqual(manifest['version'],2)
        for name,p in before.items():
            if p['type']=='numeric':
                np.testing.assert_allclose(p['mean'],after[name]['mean'],rtol=1e-9,atol=1e-9)
                np.testing.assert_allclose(p['std'],after[name]['std'],rtol=1e-6,atol=1e-8)
            else:
                np.testing.assert_allclose(p['probabilities'],after[name]['probabilities'],atol=1e-12)
                self.assertEqual(p['labels'].tolist(),after[name]['labels'].tolist())
        _,_,third=open_model(io.BytesIO(save_model(dict(self.job,model=model))))
        np.testing.assert_allclose(third.predict_all(points)['Etat']['probabilities'],after['Etat']['probabilities'])
        with self.assertRaisesRegex(ValueError,'inconnue'):model.predict_all([[.2,.3,'plastique']])

    def test_mixed_api_predicts_all_outputs_and_reports_invalid_points(self):
        client=new_client();opened=client.post('/api/models/open',data={'file':(io.BytesIO(self.archive),'model.gpmodel')})
        self.assertEqual(opened.status_code,200,opened.get_json());job=opened.get_json();url='/api/jobs/'+job['id']+'/predict'
        response=client.post(url,json={'values':{'X':.3,'Z':.5,'Matiere':'acier'}})
        self.assertEqual(response.status_code,200,response.get_json());self.assertEqual(set(response.get_json()['outputs']),set(self.options['targets']))
        for value in ({'X':True,'Z':.5,'Matiere':'acier'},{'X':.3,'Z':.5,'Matiere':'inconnu'},{'X':.3,'Z':float('inf'),'Matiere':'acier'}):
            self.assertEqual(client.post(url,json={'values':value}).status_code,400)
        pdf=client.get('/api/jobs/'+job['id']+'/report.pdf')
        self.assertEqual(pdf.status_code,200);self.assertTrue(pdf.data.startswith(b'%PDF-'));self.assertIn(b'/Outlines',pdf.data)

    def test_excel_outputs_classes_and_numbers_and_preserves_filled_cells(self):
        book=Workbook();ws=book.active;ws.title='Points';ws.append(['X','Z','Matiere','Force','Etat','Rendement'])
        ws.append([.3,.5,'acier',123,None,None]);ws.append([.4,.6,'inconnu',None,None,None]);ws.append([.8,.2,'alu',None,None,None])
        book.create_sheet('Conserver')['A1']='=1+2';raw=io.BytesIO();book.save(raw)
        options=dict(sheet='Points',header_row=1,first_row=2,last_row=4,mapping={'X':1,'Z':2,'Matiere':3},outputs={'Force':dict(column=4),'Etat':dict(column=5),'Rendement':dict(column=6)})
        output,summary=predict_workbook(raw.getvalue(),self.model,self.options['features'],options)
        saved=load_workbook(io.BytesIO(output));ws=saved['Points']
        self.assertEqual(ws['D2'].value,123);self.assertIsInstance(ws['E2'].value,str);self.assertIsInstance(ws['F2'].value,float)
        self.assertIsNone(ws['E3'].value);self.assertEqual(summary['outputs']['Etat']['invalid_rows'],[3])
        self.assertEqual(saved['Conserver']['A1'].value,'=1+2')
        options['outputs']['Etat']['column']=4
        with self.assertRaises(ValueError):predict_workbook(raw.getvalue(),self.model,self.options['features'],options)

    def test_active_acquisition_respects_constraints_and_all_output_types(self):
        options=dict(count=6,pool_size=128,domain=[dict(name='X',minimum=-4,maximum=4,distribution='normal',mean=0,std=2),dict(name='Z',minimum=-1,maximum=1),dict(name='Matiere',categories=['acier','alu'])],constraints=[dict(left='X',operator='>',right=dict(kind='feature',value='Z'))],seed=33)
        a=suggestions(self.model,self.result,options);b=suggestions(self.model,self.result,options)
        self.assertEqual(a,b);self.assertEqual(len(a['rows']),6)
        self.assertTrue(all(r['values'][0]>r['values'][1] for r in a['rows']))
        self.assertEqual(set(a['rows'][0]['predictions']),set(self.options['targets']))
        self.assertEqual(a['rows'][0]['predictions']['Etat']['type'],'categorical')

    def test_v2_rejects_non_numeric_arrays_and_unexpected_members(self):
        stream=io.BytesIO();np.save(stream,np.array([{'unsafe':1}],dtype=object),allow_pickle=True)
        raw=mutate_archive(self.archive,'models/0/coefficients.npy',stream.getvalue())
        with self.assertRaises(ValueError):open_model(io.BytesIO(raw))
        output=io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(self.archive)) as old,zipfile.ZipFile(output,'w') as new:
            for name in old.namelist():new.writestr(name,old.read(name))
            new.writestr('unexpected.txt','ignored?')
        with self.assertRaises(ValueError):open_model(io.BytesIO(output.getvalue()))

    def test_report_contains_all_outputs_plan_and_original_records(self):
        job=dict(self.job,design=dict(features=['X','Z','Matiere'],targets=['Force','Etat','Rendement'],rows=[dict(values=[4,.5,'acier'])],seed=12,constraints=[dict(type='interval',left='X',operator='outside',lower=-1,upper=1),dict(type='linear',terms=[dict(feature='X',coefficient=1)],operator='<=',value=5)]))
        raw=make_report(job)
        self.assertTrue(raw.startswith(b'%PDF-'));self.assertIn(b'/Outlines',raw);self.assertIn(b'/Subtype /Link',raw)
        self.assertGreater(raw.count(b'/Type /Page\n'),8)
        from pathlib import Path
        Path('/tmp/gp-rapport-mixte.pdf').write_bytes(raw)


class DesignTests(unittest.TestCase):
    def test_upload_preserves_category_codes_and_punctuation(self):
        client=new_client()
        csv='Code;Matiere;Y\n001;acier, mat;1\n01;alu;2\n1;acier, mat;3\n'
        data=client.post('/api/upload',data={'file':(io.BytesIO(csv.encode()),'codes.csv')}).get_json()
        self.assertEqual([r[0] for r in data['records']],['001','01','1'])
        self.assertEqual(data['records'][0][1],'acier, mat')
        from typed_models import infer_type
        self.assertEqual(infer_type(pd.Series([True,False],dtype=object)),'categorical')

    def test_uniform_and_truncated_normal_preserve_quantile_strata(self):
        schema=[dict(name='X',type='numeric')]
        for entry in (dict(name='X',minimum=0,maximum=1),dict(name='X',minimum=-.3,maximum=.8,distribution='normal',mean=.2,std=.3),dict(name='X',minimum=.1,maximum=10,distribution='loguniform')):
            domain=validate_domain([entry],schema);points=lhs_sample(domain,100,2025)[:,0].astype(float)
            if entry.get('distribution')=='normal':q=truncnorm.cdf(points,(-.3-.2)/.3,(.8-.2)/.3,loc=.2,scale=.3)
            elif entry.get('distribution')=='loguniform':q=(np.log(points)-np.log(.1))/(np.log(10)-np.log(.1))
            else:q=points
            self.assertEqual(sorted(np.floor(q*100).astype(int).tolist()),list(range(100)))

    def test_constraints_categories_weights_and_reproducibility(self):
        schema=[dict(name='X',type='numeric'),dict(name='Z',type='numeric'),dict(name='M',type='categorical',categories=['a','b'])]
        options=dict(count=25,seed=45,domain=[dict(name='X',minimum=0,maximum=10,integer=True),dict(name='Z',minimum=0,maximum=10),dict(name='M',categories=['a','b'],probabilities=[1,3])],constraints=[dict(left='X',operator='<',right=dict(kind='feature',value='Z'))])
        a=generate_design(options,schema);b=generate_design(options,schema)
        self.assertEqual(a,b);self.assertEqual(len(a['rows']),25);self.assertFalse(a['stratification_preserved'])
        self.assertTrue(all(r['values'][0]<r['values'][1] and r['values'][0].is_integer() for r in a['rows']))
        options['constraints']=[dict(left='X',operator='<',right=dict(kind='constant',value=-100))]
        with self.assertRaisesRegex(ValueError,'réalisable'):generate_design(options,schema)
        options['constraints']=['__import__("os").system("bad")']
        with self.assertRaises(ValueError):generate_design(options,schema)

    def test_design_api_works_before_any_model_is_trained(self):
        client=new_client();f=pd.DataFrame({'X':np.arange(12),'Z':np.arange(12)**2,'Y':np.arange(12)})
        dataset=client.post('/api/upload',data={'file':(io.BytesIO(f.to_csv(index=False).encode()),'initial.csv')}).get_json()
        with patch('models.GPRWrapper.fit',side_effect=AssertionError('No model')):
            response=client.post('/api/design',json=dict(dataset_id=dataset['id'],features=['X','Z'],targets=['Y'],count=15,domain=[dict(name='X',minimum=0,maximum=12),dict(name='Z',minimum=20,maximum=200)],constraints=[]))
        self.assertEqual(response.status_code,200,response.get_json());self.assertEqual(len(response.get_json()['rows']),15)
        pdf=client.get('/api/designs/'+response.get_json()['id']+'/report.pdf')
        self.assertEqual(pdf.status_code,200);self.assertTrue(pdf.data.startswith(b'%PDF-'))
        bad=client.post('/api/design',json=dict(dataset_id=dataset['id'],features=[[]]))
        self.assertEqual(bad.status_code,400)

    def test_invalid_laws_and_incompatible_constraints_are_rejected(self):
        schema=[dict(name='X',type='numeric')]
        for entry in (dict(name='X',minimum=0,maximum=1,distribution='normal',mean=0,std=0),dict(name='X',minimum=0,maximum=1,distribution='loguniform'),dict(name='X',minimum=True,maximum=2),dict(name='X',minimum=0,maximum=1,distribution='executable')):
            with self.assertRaises(ValueError):validate_domain([entry],schema)


if __name__=='__main__':unittest.main()
