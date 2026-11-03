"""Format v2 : encodage et postérieurs numériques, jamais pickle/joblib."""
import io
import json
import zipfile
import numpy as np
from typed_models import InputEncoder, GaussianClassifier, ModelBundle


def regression_state(model):
    from model_io import SavedModel
    if isinstance(model, SavedModel):
        return model.kernel, model.arrays
    fitted = model.model
    arrays = dict(input_mean=model.scaler_x.mean_, input_scale=model.scaler_x.scale_)
    if hasattr(fitted, 'X_train_'):
        arrays.update(x_train=fitted.X_train_, coefficients=fitted.alpha_, cholesky=fitted.L_)
    else:
        arrays.update(x_train=fitted._X.detach().cpu().numpy(), coefficients=fitted._coefficients.detach().cpu().numpy().ravel(),
                      cholesky=fitted._chol.detach().cpu().numpy())
    return fitted.kernel_, arrays


def save_bundle(job):
    from model_io import encode_kernel
    model = job['model']; states, arrays = [], {}
    for i, (target, entry) in enumerate(model.outputs.items()):
        prefix = f'models/{i}/'
        m = entry['model']
        state = dict(target=target, type=entry['type'])
        if entry['type'] == 'numeric':
            kernel, a = regression_state(m)
            state.update(kernel=encode_kernel(kernel), y_mean=m.y_mean, y_std=m.y_std)
            arrays.update({prefix+name:value for name,value in a.items()})
        else:
            state.update(classes=m.classes, binaries=[])
            arrays.update({prefix+'input_mean':m.scaler_x.mean_, prefix+'input_scale':m.scaler_x.scale_})
            for j, binary in enumerate(m.binaries):
                state['binaries'].append(dict(kernel=encode_kernel(binary['kernel'])))
                arrays.update({prefix+f'binary/{j}/'+name:binary[name] for name in ('x_train','coefficients','weights','cholesky')})
        states.append(state)
    manifest = dict(format='processus-gaussiens', version=2, encoder=model.encoder.schema, models=states,
                    dataset=job['dataset'], options=job['options'], result=job['result'])
    output=io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, allow_nan=False))
        archive.writestr('predictions.csv', job['csv'])
        for name, value in arrays.items():
            stream=io.BytesIO(); np.save(stream, np.asarray(value, dtype='<f8'), allow_pickle=False)
            archive.writestr(name+'.npy', stream.getvalue())
    return output.getvalue()


def check_cholesky(array, n):
    if array.shape != (n,n) or (np.diag(array)<=0).any() or not np.array_equal(array,np.tril(array)):
        raise ValueError('Factorisation invalide.')


def load_bundle(archive, manifest):
    from model_io import read_array, decode_kernel, SavedModel, ARRAY_NAMES, MAX_ARCHIVE_BYTES
    entries = archive.infolist()
    if sum(e.file_size for e in entries)>MAX_ARCHIVE_BYTES or any(e.flag_bits & 1 for e in entries):
        raise ValueError('Archive invalide ou trop volumineuse.')
    schema, states, result = manifest['encoder'], manifest['models'], manifest['result']
    if not isinstance(schema,list) or not schema or not isinstance(states,list) or not states:
        raise ValueError('Schéma du modèle incomplet.')
    names=[]; encoded=[]
    for entry in schema:
        if not isinstance(entry,dict) or not isinstance(entry.get('name'),str) or not entry['name'] or entry.get('type') not in ('numeric','categorical'):
            raise ValueError('Schéma des entrées invalide.')
        names.append(entry['name'])
        if entry['type']=='categorical':
            cats=entry.get('categories')
            if not isinstance(cats,list) or not cats or any(not isinstance(c,str) or not c for c in cats) or len(set(cats))!=len(cats):
                raise ValueError('Modalités des entrées invalides.')
            encoded.extend(entry['name']+' = '+c for c in cats)
        else: encoded.append(entry['name'])
    if len(set(names))!=len(names) or len(encoded)>30000:
        raise ValueError('Dimensions d’encodage invalides.')
    expected={'manifest.json','predictions.csv'}
    outputs={}; n=None
    for i,state in enumerate(states):
        prefix=f'models/{i}/'; target=state['target']
        if not isinstance(target,str) or not target or target in outputs or target in names:
            raise ValueError('Sorties invalides.')
        if state['type']=='numeric':
            a={name:read_array(archive,prefix+name) for name in ARRAY_NAMES}
            expected.update(prefix+name+'.npy' for name in ARRAY_NAMES)
            count,dim=a['x_train'].shape
            check_cholesky(a['cholesky'],count)
            if a['coefficients'].shape!=(count,): raise ValueError('Coefficients invalides.')
            mean, std=float(state['y_mean']),float(state['y_std'])
            if not np.isfinite([mean,std]).all() or std<=0: raise ValueError('Normalisation invalide.')
            model=SavedModel(decode_kernel(state['kernel'],len(encoded)),a,mean,std)
            input_mean,input_scale=a['input_mean'],a['input_scale']
        elif state['type']=='categorical':
            classes=state['classes']; binaries=state['binaries']
            if (not isinstance(classes,list) or len(classes)<2 or len(set(classes))!=len(classes)
                    or any(not isinstance(c,str) for c in classes) or len(binaries)!=(1 if len(classes)==2 else len(classes))):
                raise ValueError('Classes de sortie invalides.')
            model=GaussianClassifier(); model.classes=classes; model.binaries=[]
            input_mean,input_scale=[read_array(archive,prefix+k) for k in ('input_mean','input_scale')]
            expected.update(prefix+k+'.npy' for k in ('input_mean','input_scale'))
            count=dim=None
            for j,b in enumerate(binaries):
                keys=('x_train','coefficients','weights','cholesky')
                a={key:read_array(archive,prefix+f'binary/{j}/'+key) for key in keys}
                expected.update(prefix+f'binary/{j}/'+key+'.npy' for key in keys)
                rows,cols=a['x_train'].shape
                check_cholesky(a['cholesky'],rows)
                if (count is not None and (rows,cols)!=(count,dim)) or a['coefficients'].shape!=(rows,) or a['weights'].shape!=(rows,) or (a['weights']<0).any():
                    raise ValueError('Posterior de classification invalide.')
                count,dim=rows,cols
                model.binaries.append(dict(kernel=decode_kernel(b['kernel'],len(encoded)),**a))
            model.scaler_x.mean_=input_mean; model.scaler_x.scale_=input_scale; model.scaler_x.n_features_in_=dim
        else: raise ValueError('Type de sortie inconnu.')
        if (dim!=len(encoded) or input_mean.shape!=(dim,) or input_scale.shape!=(dim,) or (input_scale<=0).any()
                or not 1<=count<=3000 or (n is not None and count!=n)):
            raise ValueError('Dimensions du modèle invalides.')
        n=count; outputs[target]=dict(type=state['type'],model=model)
    if len(entries)!=len(expected) or {e.filename for e in entries}!=expected:
        raise ValueError('Membres d’archive inattendus ou dupliqués.')
    data,options=manifest['dataset'],manifest['options']
    columns=[c['name'] for c in data['columns']]
    raw=result['data']; train,test=raw['train'],raw['test']; length=len(raw['x'])
    if (names!=result['features'] or list(outputs)!=result['targets'] or options['features']!=names or options['targets']!=list(outputs)
            or len(train)!=n or not n<length<=3000 or sorted(train+test)!=list(range(length))
            or any(len(row)!=len(names) for row in raw['x']) or len(data['records'])!=data['rows']
            or any(len(row)!=len(columns) for row in data['records']) or not set(names+list(outputs))<=set(columns)
            or [o['target'] for o in result['outputs']]!=list(outputs)):
        raise ValueError('Description de l’étude incohérente.')
    encoder=InputEncoder(schema); encoder.names=encoded
    bundle=ModelBundle(encoder,outputs)
    bundle.predict_all(np.array(raw['x'],dtype=object)[train[:1]])
    return manifest, archive.read('predictions.csv'), bundle
