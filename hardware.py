"""Diagnostic du Python serveur, jamais du navigateur ou du client SSH."""
import os
import socket
import sys


def gpu_capabilities():
    result = dict(cuda=False, gpu=None, host=socket.gethostname(), python=sys.executable,
                  torch_version=None, cuda_runtime=None, visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  reason='', code='unavailable')
    try:
        import torch
    except ImportError:
        return dict(result, code='torch_missing', reason='PyTorch n’est pas installé dans le Python qui lance ce serveur.')
    except Exception as exc:
        return dict(result, code='torch_import', reason='PyTorch ne peut pas être chargé : '+str(exc))
    result.update(torch_version=torch.__version__, cuda_runtime=torch.version.cuda)
    if not torch.version.cuda:
        return dict(result, code='torch_cpu', reason='Cette installation de PyTorch est une version CPU, sans CUDA.')
    if not torch.cuda.is_available():
        return dict(result, code='cuda_hidden', reason='Aucun GPU CUDA accessible au serveur. Vérifiez le pilote NVIDIA, l’allocation GPU du serveur et CUDA_VISIBLE_DEVICES.')
    try:
        # Exercise the real backend, including the likelihood gradient, not just detection.
        import numpy as np
        from gpu_model import TorchGaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
        kernel = ConstantKernel(1.) * RBF(1.) + WhiteKernel(.01)
        model = TorchGaussianProcessRegressor(kernel, optimizer=None, n_restarts_optimizer=0)
        x = np.linspace(-1, 1, 8)[:, None]
        model.fit(x, np.sin(x[:, 0]))
        model._identite = torch.eye(len(x), device=model.device_, dtype=torch.float64)
        loss, gradient = model._objectif(kernel.theta)
        del model._identite
        mean, std = model.predict(x[:2], return_std=True)
        torch.cuda.synchronize()
        if not np.isfinite(np.r_[loss, gradient, mean, std]).all():
            raise RuntimeError('Résultat numérique non fini.')
        result.update(cuda=True, gpu=torch.cuda.get_device_name(), code='ready',
                      reason='Calcul GP float64, factorisation, gradient et prédiction CUDA vérifiés.')
    except Exception as exc:
        result.update(code='cuda_compute', reason='GPU détecté mais le calcul GP échoue : '+str(exc))
    return result
