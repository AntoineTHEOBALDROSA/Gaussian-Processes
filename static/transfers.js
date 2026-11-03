'use strict';

function transferStatus(message) {
  $('transfer-status').textContent = message || '';
  show('transfer-status', Boolean(message));
}
function downloadName(suffix) {
  const base = (state.dataset?.filename || 'modele').replace(/\.[^.]+$/, '').replace(/[^\p{L}\p{N}._-]/gu, '-').slice(0,80);
  return `${base}-${suffix}`;
}
async function downloadResponse(path, options, filename) {
  let response;
  try { response = await fetch(path, options); }
  catch { throw new Error('Le serveur local ne répond plus. Vérifiez qu’il est démarré.'); }
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || 'L’export n’a pas abouti. Réessayez.');
  }
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement('a'); link.href = url; link.download = filename;
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
  return response.headers;
}
document.getElementById('open-model').addEventListener('click', () => {
  if (!state.busy) $('model-input').click();
});
document.getElementById('model-input').addEventListener('change', async event => {
  const file = event.target.files[0];
  if (!file || state.busy) return;
  if (!/\.gpmodel$/i.test(file.name) || file.size > 100 * 1024 * 1024) {
    error('Choisissez un fichier .gpmodel de moins de 100 Mo.'); event.target.value = ''; return;
  }
  const form = new FormData(); form.append('file', file);
  busy(true); error(null); transferStatus('Ouverture du modèle…');
  try {
    const job = await api('/api/models/open', {method:'POST', body:form});
    clearTimeout(state.poll);
    setDataset(job.dataset); applyConfiguration(job.options);
    state.job = job.id; state.result = job.result; state.restored = true;
    remember(job.id); busy(false); renderResults();
    transferStatus('Modèle rouvert. Les résultats et les prédictions sont disponibles sans nouvel entraînement.');
  } catch (e) { error(e.message); transferStatus(null); }
  finally { busy(false); $('model-input').value = ''; }
});
document.getElementById('save-model').addEventListener('click', async () => {
  if (state.busy || !state.result) return;
  busy(true); error(null); transferStatus('Préparation de la sauvegarde…');
  try {
    await downloadResponse(`/api/jobs/${state.job}/model.gpmodel`, {}, downloadName('modele.gpmodel'));
    transferStatus('Sauvegarde téléchargée. Vous pouvez la rouvrir avec « Ouvrir un modèle ».');
  } catch (e) { error(e.message); transferStatus(null); }
  finally { busy(false); }
});

document.getElementById('report-pdf').addEventListener('click',async()=>{
  if(state.busy||(!state.result&&!state.designId))return;
  busy(true);error(null);transferStatus('Création du rapport complet avec sommaire…');
  try{
    const path=state.result?`/api/jobs/${state.job}/report.pdf`:`/api/designs/${state.designId}/report.pdf`;
    await downloadResponse(path,{},downloadName('rapport-complet.pdf'));transferStatus('Rapport complet téléchargé : sommaire, données, sorties et dernier plan généré.');
  }catch(e){error(e.message);transferStatus(null);}finally{busy(false);}
});
