import builtins
import importlib.util
import types
import unittest
import warnings
from unittest.mock import patch
import numpy as np
from hardware import gpu_capabilities


class HardwareTests(unittest.TestCase):
    def test_missing_and_cpu_only_torch_have_explicit_reasons(self):
        original=builtins.__import__
        def without_torch(name,*args,**kwargs):
            if name=='torch':raise ImportError('not installed')
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=without_torch):
            result=gpu_capabilities()
        self.assertFalse(result['cuda']);self.assertEqual(result['code'],'torch_missing');self.assertTrue(result['python'])
        fake=types.SimpleNamespace(__version__='test',version=types.SimpleNamespace(cuda=None))
        with patch.dict('sys.modules',{'torch':fake}):result=gpu_capabilities()
        self.assertEqual(result['code'],'torch_cpu')

    def test_inaccessible_remote_device_is_not_reported_as_ready(self):
        fake=types.SimpleNamespace(__version__='test',version=types.SimpleNamespace(cuda='12'),cuda=types.SimpleNamespace(is_available=lambda:False))
        with patch.dict('sys.modules',{'torch':fake}):result=gpu_capabilities()
        self.assertFalse(result['cuda']);self.assertEqual(result['code'],'cuda_hidden');self.assertIn('allocation',result['reason'])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch optionnel absent de ce Python')
    def test_tensor_backend_matches_cpu_reference_for_all_four_kernels(self):
        from gpu_model import TorchGaussianProcessRegressor
        from sklearn.gaussian_process import GaussianProcessRegressor
        from models import noyaux_candidats
        x=np.linspace(-1,1,12);X=np.column_stack([x,x*x]);y=np.sin(x)
        points=np.array([[.2,.4],[-1,.5]])
        for name,k in noyaux_candidats(2).items():
            with self.subTest(kernel=name),warnings.catch_warnings():
                warnings.simplefilter('ignore')
                a=GaussianProcessRegressor(kernel=k,optimizer=None).fit(X,y)
                b=TorchGaussianProcessRegressor(kernel=k,optimizer=None,device='cpu',n_restarts_optimizer=0).fit(X,y)
                for v,w in zip(a.predict(points,return_std=True),b.predict(points,return_std=True)):
                    np.testing.assert_allclose(v,w,atol=1e-8,rtol=1e-8)

    def test_real_cuda_when_allocated(self):
        if not gpu_capabilities()['cuda']:
            self.skipTest('Aucune carte CUDA accessible pour un essai matériel')
        from gpu_model import TorchGaussianProcessRegressor
        from models import noyaux_candidats
        x=np.linspace(-1,1,10)[:,None]
        m=TorchGaussianProcessRegressor(noyaux_candidats(1)['RBF (ARD)'],n_restarts_optimizer=0,maxiter=3).fit(x,np.sin(x[:,0]))
        mean,std=m.predict(x,return_std=True)
        self.assertEqual(m.device_.type,'cuda');self.assertTrue(np.isfinite(mean).all());self.assertTrue(np.isfinite(std).all())


if __name__=='__main__':unittest.main()
