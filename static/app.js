'use strict';
const $ = (id) => document.getElementById(id);
const state = {dataset:null, result:null, job:null, busy:false, trained:false, step:1, tab:'parity', poll:null, restored:false, chartReady:Promise.resolve()};
const number = (value, digits=4) => Number(value).toLocaleString('fr-FR', {maximumSignificantDigits:digits});
const show = (id, visible) => { $(id).hidden = !visible; };
function error(message) { $('error').textContent = message || ''; show('error', Boolean(message)); }
async function api(path, options={}) {
  let response;
  try { response = await fetch(path, options); }
  catch { throw new Error('Le serveur local ne répond plus. Vérifiez qu’il est démarré, puis réessayez.'); }
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'La requête a échoué.');
  return data;
}
function remember(id) { try { id ? sessionStorage.setItem('gp-job',id) : sessionStorage.removeItem('gp-job'); } catch {} }
function busy(value) {
  state.busy = value;
  document.querySelectorAll('#config input, #config select, #file-input, #example, #clear-file, #model-input, #open-model, #input-matrix-pdf, [data-tab], [data-input-tab], #slice-variable, #predict-button, #prediction-inputs input, #excel-card input, #excel-card select, #excel-card button, #coverage-view input, #coverage-view select, #coverage-view button').forEach(el => el.disabled = value);
  // CUDA remains unavailable unless the local capability check succeeded.
  $('device').querySelector('[value=cuda]').disabled = !$('device').dataset.cuda;
  $('dropzone').classList.toggle('disabled',value);
  updateSteps();
  updateRun();
}
function selectedFeatures() { return [...document.querySelectorAll('#features input:checked')].map(el=>el.value); }
function updateRun() { $('run').disabled = state.busy || !state.dataset || !selectedFeatures().length || !$('target').value; }
function updateSteps() {
  const navigation = document.querySelector('.workflow-nav');
  navigation.classList.toggle('model-navigation', state.trained);
  navigation.setAttribute('aria-label', state.trained ? 'Vues du modèle' : 'Étapes de l’analyse');
  show('step1', !state.trained);
  show('new-model', state.trained);
  show('clear-file', !state.trained);
  $('new-model').disabled = state.busy;
  show('save-model', Boolean(state.result));
  $('save-model').disabled = state.busy || !state.result;
  $('export-pdf').disabled = state.busy || !state.result;
  document.querySelectorAll('[data-step]').forEach(button => {
    const step = Number(button.dataset.step);
    button.disabled = state.busy || (step === 1 && state.trained) || (step === 2 && !state.dataset) || (step === 3 && !state.result);
    button.querySelector('span').hidden = state.trained;
    if (state.trained) {
      button.removeAttribute('aria-current');
      button.setAttribute('aria-pressed', String(step === state.step));
    } else {
      button.removeAttribute('aria-pressed');
      if (step === state.step) button.setAttribute('aria-current', 'step');
      else button.removeAttribute('aria-current');
    }
    button.title = state.busy ? 'Attendez la fin de l’opération en cours.' : '';
  });
}
function stage(step) {
  state.step = step;
  show('input-panel', step !== 3);
  document.querySelector('.workspace').classList.toggle('results-mode', step === 3);
  $('panel-number').textContent = String(step).padStart(2, '0');
  $('panel-title').textContent = step === 3 ? (state.busy ? 'Entraînement' : 'Résultats') : 'Données';
  document.querySelector('.results-panel').setAttribute('aria-label', step === 3 ? 'Espace de résultats' : 'Exploration des données');
  [1,2,3].forEach(i=> { $('step'+i).classList.toggle('active',i===step); $('step'+i).classList.toggle('done',i<step); });
  show('dropzone', step === 1);
  show('example', step === 1);
  show('config', Boolean(state.dataset) && step !== 1);
  show('run', step !== 1);
  updateSteps();
}
function navigateStep(step) {
  if (state.busy || (step === 1 && state.trained) || (step === 2 && !state.dataset) || (step === 3 && !state.result)) return;
  const scrollPosition = {top:window.scrollY, left:window.scrollX, behavior:'instant'};
  // Preserve the scroll range even when the destination panel is shorter.
  document.querySelector('main').style.minHeight = `${window.innerHeight + scrollPosition.top}px`;
  error(null);
  stage(step);
  if (step === 3) {
    display('results');
    $('result-state').textContent = state.restored ? 'Modèle rouvert' : 'Analyse terminée';
    renderChart();
  } else {
    display(state.dataset ? 'preview' : 'empty');
    $('result-state').textContent = state.dataset ? 'Données importées' : 'En attente de données';
  }
  // Keep focus on the clicked step; changing panels must not move the viewport.
  if (step !== 3) refreshInputMatrix();
  window.scrollTo(scrollPosition);
  requestAnimationFrame(() => window.scrollTo(scrollPosition));
}
document.querySelectorAll('[data-step]').forEach(button => {
  button.addEventListener('click', () => navigateStep(Number(button.dataset.step)));
});
stage(1);
function display(view) { ['empty','preview','progress-view','results'].forEach(id=>show(id,id===view)); }
function renderFeatures(chosen) {
  const target = $('target').value;
  $('features').replaceChildren();
  state.dataset.columns.filter(c=>c.numeric && c.name!==target).forEach(column=>{
    const label = document.createElement('label');
    const input = document.createElement('input'); input.type='checkbox'; input.value=column.name;
    input.checked = chosen.includes(column.name); input.addEventListener('change', configurationChanged);
    label.append(input,document.createTextNode(column.name)); $('features').append(label);
  });
  updateDataNote();
  updateRun();
}
function updateDataNote() {
  const target = $('target').value;
  const invalid = state.dataset.columns.filter(c=>[target,...selectedFeatures()].includes(c.name)).reduce((n,c)=>n+c.invalid,0);
  $('data-note').textContent = invalid ? 'Certaines valeurs sont manquantes ou non numériques. Les lignes concernées seront exclues du calcul ; leur nombre sera indiqué dans les résultats.' : 'Les colonnes d’identification sont exclues par défaut. Vérifiez votre sélection.';
  updateRun();
}
function configurationChanged() {
  updateDataNote();
  refreshInputMatrix();
  if (state.result) { state.result=null; display('preview'); $('result-state').textContent='Configuration modifiée'; remember(null); stage(2); }
  saveConfiguration();
  updateRun();
}
function fillTable(container, columns, rows) {
  const table = container.tagName==='TABLE' ? container : document.createElement('table');
  table.replaceChildren(); const head=document.createElement('thead'), tr=document.createElement('tr');
  columns.forEach(c=>{ const th=document.createElement('th'); th.textContent=c; th.scope='col'; tr.append(th); }); head.append(tr); table.append(head);
  const body=document.createElement('tbody');
  rows.forEach(row=>{const tr=document.createElement('tr');row.forEach(value=>{const td=document.createElement('td');td.textContent=value===null ? '—' : typeof value==='number' ? number(value,6) : String(value);tr.append(td);});body.append(tr);});
  table.append(body); if(container!==table) container.replaceChildren(table);
}
function setDataset(data) {
  state.dataset=data; state.restored=false; state.result=null; state.trained=false; error(null); show('dropzone',false); show('example',false); show('file-summary',true); show('config',true);
  $('file-name').textContent=data.filename; $('file-meta').textContent=`${data.rows} lignes · ${data.columns.length} colonnes`;
  document.querySelector('.file-icon').textContent=data.filename.toLowerCase().endsWith('.xlsx')?'XLSX':'CSV';
  $('target').replaceChildren();
  data.columns.filter(c=>c.numeric).forEach(c=>$('target').add(new Option(c.name,c.name)));
  $('target').value=data.target; renderFeatures(data.features);
  renderDataTable(data);
  $('preview-count').textContent=`${data.rows} lignes · ${data.columns.length} colonnes`;
  selectInputTab('table');
  display('preview'); $('result-state').textContent='Données importées'; stage(2); updateRun(); saveConfiguration();
}
async function importFile(file) {
  if(state.busy || state.trained || !file) return;
  if(!/\.(csv|xlsx)$/i.test(file.name)) { error('Choisissez un fichier CSV ou Excel (.xlsx).'); return; }
  if(file.size>20*1024*1024) { error('Le fichier dépasse la limite de 20 Mo.'); return; }
  const form=new FormData(); form.append('file',file); busy(true); error(null); $('result-state').textContent='Lecture du fichier…';
  try {setDataset(await api('/api/upload',{method:'POST',body:form}));remember(null);}
  catch(e){error(e.message);$('result-state').textContent='Import non abouti';}
  finally {busy(false);$('file-input').value='';}
}
$('file-input').addEventListener('change',e=>importFile(e.target.files[0]));
['dragenter','dragover'].forEach(type=>$('dropzone').addEventListener(type,e=>{e.preventDefault();$('dropzone').classList.add('dragover');}));
['dragleave','drop'].forEach(type=>$('dropzone').addEventListener(type,e=>{e.preventDefault();$('dropzone').classList.remove('dragover');}));
$('dropzone').addEventListener('drop',e=>{
  if(e.dataTransfer.files.length!==1) return error('Déposez un seul fichier à la fois.');
  importFile(e.dataTransfer.files[0]);
});
// Prevent the browser from navigating to files dropped outside the drop area.
window.addEventListener('dragover',e=>e.preventDefault());
window.addEventListener('drop',e=>e.preventDefault());
$('example').addEventListener('click',async()=>{
  if(state.busy || state.trained) return;
  busy(true);error(null);$('result-state').textContent='Lecture du fichier…';
  try{setDataset(await api('/api/example',{method:'POST'}));remember(null);}catch(e){error(e.message);$('result-state').textContent='Import non abouti';}finally{busy(false);}
});
function resetModel() {
  if (state.busy) return;
  clearTimeout(state.poll);
  state.dataset=null;state.result=null;state.job=null;state.trained=false;state.poll=null;state.restored=false;
  remember(null);saveConfiguration();error(null);
  $('transfer-status').hidden=true;
  $('file-input').value='';$('quality').value='standard';$('device').value='cpu';$('kernel-search').value='base';$('length-min').value='0.01';$('length-max').value='1000';updateQualityNote();
  ['file-summary','config'].forEach(id=>show(id,false));
  selectInputTab('table');
  display('empty');$('result-state').textContent='En attente de données';stage(1);updateRun();
}
$('clear-file').addEventListener('click',resetModel);
$('new-model').addEventListener('click',resetModel);
function updateQualityNote() {
  $('quality-note').textContent=$('quality').value==='quick'?'Validation croisée : 3 plis · sans relance par pli, 1 à l’ajustement final.':'Validation croisée : 5 plis · 2 relances par pli, 10 à l’ajustement final.';
}
function saveConfiguration() {
  try {
    if (!state.dataset) { sessionStorage.removeItem('gp-configuration'); return; }
    sessionStorage.setItem('gp-configuration', JSON.stringify({dataset_id:state.dataset.id, target:$('target').value, features:selectedFeatures(), quality:$('quality').value, device:$('device').value, kernel_search:$('kernel-search').value, length_bounds:[Number($('length-min').value),Number($('length-max').value)], trained:state.trained}));
  } catch {}
}
function applyConfiguration(options) {
  $('target').value=options.target;renderFeatures(options.features);
  $('quality').value=options.quality;$('device').value=options.device;$('kernel-search').value=options.kernel_search || 'base';const bounds=options.length_bounds || [0.01,1000];$('length-min').value=bounds[0];$('length-max').value=bounds[1];updateQualityNote();
}
$('target').addEventListener('change',()=>{const old=selectedFeatures(); renderFeatures([...old,...state.dataset.features]);configurationChanged();});
$('quality').addEventListener('change',()=>{updateQualityNote();configurationChanged();});
$('device').addEventListener('change',configurationChanged);
['kernel-search','length-min','length-max'].forEach(id=>$(id).addEventListener('change',configurationChanged));
async function startAnalysis() {
  if(state.busy || !state.dataset) throw new Error('Importez un fichier et attendez la fin du calcul en cours.');
  const options={dataset_id:state.dataset.id,target:$('target').value,features:selectedFeatures(),quality:$('quality').value,device:$('device').value,kernel_search:$('kernel-search').value,length_bounds:[Number($('length-min').value),Number($('length-max').value)]};
  if(!options.features.length) throw new Error('Sélectionnez au moins une entrée.');
  error(null);busy(true);
  try {
    const job=await api('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(options)});
    state.result=null;state.restored=false;state.job=job.id;remember(job.id);display('progress-view');stage(3);$('result-state').textContent='Calcul en cours';
    $('progress').value=0;$('progress-percent').textContent='0 %';$('progress-title').textContent='Préparation de l’analyse…';$('elapsed').textContent='0 s';
    if(window.matchMedia('(max-width:760px)').matches) $('progress-view').scrollIntoView({block:'start'});
    await pollJob(job.id); return {id:job.id,status:'started'};
  } catch(e) {busy(false);throw e;}
}
$('run').addEventListener('click',()=>startAnalysis().catch(e=>error(e.message)));
async function pollJob(id) {
  clearTimeout(state.poll);
  try {
    const job=await api('/api/jobs/'+id);error(null);
    if(job.status==='running') {
      $('progress').value=job.progress;$('progress-percent').textContent=job.progress+' %';$('progress-title').textContent=job.title;$('progress-detail').textContent=job.detail;
      const seconds=Math.max(0,Math.floor(Date.now()/1000-job.started));$('elapsed').textContent=seconds<60?`${seconds} s`:`${Math.floor(seconds/60)} min ${seconds%60} s`;
      state.poll=setTimeout(()=>pollJob(id),1200);
    } else if(job.status==='done') {
      state.result=job.result;busy(false);renderResults();
    } else {
      busy(false);display('preview');$('result-state').textContent='Analyse interrompue';stage(2);error(job.error);remember(null);
    }
  } catch(e) {
    error(e.message);
    // Keep tracking the same job across transient disconnects, never launch a duplicate.
    if(e.message.includes('plus disponible')) {busy(false);display('preview');remember(null);}
    else state.poll=setTimeout(()=>pollJob(id),4000);
  }
}
function renderResults() {
  state.trained=true;saveConfiguration();
  const r=state.result;display('results');stage(3);$('result-state').textContent=state.restored ? 'Modèle rouvert' : 'Analyse terminée';
  $('metrics').replaceChildren();
  [['Erreur RMSE',number(r.metrics.rmse),'Sur les données de test'],['Erreur MAE',number(r.metrics.mae),'Sur les données de test'],['Couverture à 95 %',number(r.metrics.coverage,3)+' %','Points dans l’intervalle']].forEach(([label,value,sub])=>{
    const div=document.createElement('div');div.className='metric';
    [['metric-label',label],['metric-value',value],['metric-sub',sub]].forEach(([cls,text])=>{const el=document.createElement('div');el.className=cls;el.textContent=text;div.append(el);});$('metrics').append(div);
  });
  $('best-kernel').textContent='Noyau retenu : '+r.kernel;
  $('download').href='/api/jobs/'+state.job+'/predictions.csv';
  $('slice-variable').replaceChildren();r.features.forEach((name,i)=>$('slice-variable').add(new Option(name,String(i))));
  $('analysis-info').textContent=`${r.train_count} points d’apprentissage · ${r.test_count} points de test · ${r.dropped} ligne(s) exclue(s) · ${r.folds} plis · ${r.device.toUpperCase()} · ${number(r.seconds,3)} s. Les configurations identiques restent dans le même groupe. Graine : ${r.seed}.${state.restored ? ' Modèle rouvert sans entraînement ; les nouvelles prédictions sont calculées sur CPU.' : ''}`;
  show('warnings',r.warnings.length>0);$('warnings').querySelector('summary').textContent=`${r.warnings.length} avertissement(s) d’optimisation — consulter les détails`;
  const list=$('warnings').querySelector('ul');list.replaceChildren();r.warnings.forEach(message=>{const li=document.createElement('li');li.textContent=message;list.append(li);});
  setupPrediction();
  setupResearch();
  chooseTab('parity');
}
const blue='#316bd1', gray='#9eafc6', teal='#339e8f';
function layout(xTitle,yTitle) {
  return {font:{family:'Inter, Segoe UI, sans-serif',size:12,color:'#677a96'},paper_bgcolor:'#fff',plot_bgcolor:'#fff',margin:{l:66,r:24,t:20,b:56},autosize:true,hovermode:'closest',legend:{orientation:'h',y:1.14,x:0,font:{size:11}},xaxis:{title:{text:xTitle,standoff:14},gridcolor:'#edf1f6',zeroline:false,linecolor:'#dce3ee',automargin:true},yaxis:{title:{text:yTitle,standoff:12},gridcolor:'#edf1f6',zeroline:false,linecolor:'#dce3ee',automargin:true}};
}
function chooseTab(tab) {
  state.tab=tab;
  document.querySelectorAll('[data-tab]').forEach(btn=>{const selected=btn.dataset.tab===tab;btn.setAttribute('aria-selected',String(selected));btn.tabIndex=selected?0:-1;});
  $('chart-panel').setAttribute('aria-labelledby','tab-'+tab);renderChart();
}
document.querySelectorAll('[data-tab]').forEach((btn,index,all)=>{
  btn.addEventListener('click',()=>chooseTab(btn.dataset.tab));
  btn.addEventListener('keydown',e=>{if(!['ArrowRight','ArrowLeft','Home','End'].includes(e.key))return;e.preventDefault();let next=e.key==='Home'?0:e.key==='End'?all.length-1:(index+(e.key==='ArrowRight'?1:-1)+all.length)%all.length;all[next].focus();chooseTab(all[next].dataset.tab);});
});
$('slice-variable').addEventListener('change',renderChart);
function renderChart() {
  const r=state.result;if(!r)return;
  const tab=state.tab;let traces=[],plotLayout;
  show('coverage-view',tab==='coverage');show('chart-heading',tab!=='coverage');show('kernel-diagnostics',tab==='kernels' && Boolean(r.bound_diagnostics));show('export-pdf',tab!=='coverage');
  show('slice-variable',tab==='slices');show('kernel-table',tab==='kernels');show('chart',tab!=='kernels' && tab!=='matrix' && tab!=='coverage');show('result-matrix',tab==='matrix');
  if(tab==='coverage'){renderCoverage();return;}
  if(tab==='matrix'){
    $('chart-title').textContent='Matrice des entrées du modèle';
    $('chart-description').textContent='Toutes les lignes importées · axe horizontal : variable de la colonne ; axe vertical : variable de la ligne. Diagonale : moyenne et variance (division par N).';
    renderInputMatrix('result-matrix', state.dataset, r.features);
    return;
  }
  if(tab==='parity'){
    $('chart-title').textContent='Valeurs réelles et prédites';$('chart-description').textContent=`Test indépendant · ${r.target} · Les barres représentent l’intervalle prédictif nominal à 95 %.`;
    const p=r.parity,lo=Math.min(...p.actual,...p.predicted),hi=Math.max(...p.actual,...p.predicted);
    traces=[{x:[lo,hi],y:[lo,hi],mode:'lines',name:'Prédiction parfaite',line:{color:gray,width:1.5,dash:'dash'},hoverinfo:'skip'},{x:p.actual,y:p.predicted,mode:'markers',name:'Prédictions',marker:{color:blue,size:7,opacity:.85},error_y:{type:'data',array:p.std.map(s=>1.96*s),color:'#afc7ef',thickness:1,width:2},hovertemplate:'Réel : %{x:.5g}<br>Prédit : %{y:.5g}<extra></extra>'}];
    plotLayout=layout('Valeur réelle ('+r.target+')','Valeur prédite ('+r.target+')');
  } else if(tab==='slices') {
    const s=r.slices[Number($('slice-variable').value)||0];
    $('chart-title').textContent='Réponse du modèle selon '+s.name;$('chart-description').textContent='Les autres entrées sont fixées à leur médiane d’apprentissage. Les observations sont projetées sur cet axe.';
    traces=[{x:s.x,y:s.lower,mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip'},{x:s.x,y:s.upper,mode:'lines',line:{width:0},fill:'tonexty',fillcolor:'rgba(49,107,209,.14)',name:'Intervalle à 95 %',hoverinfo:'skip'},{x:s.train_x,y:s.train_y,mode:'markers',marker:{color:gray,size:5,opacity:.5},name:'Apprentissage'},{x:s.x,y:s.mean,mode:'lines',line:{color:blue,width:2.5},name:'Prédiction'}];plotLayout=layout(s.name,r.target);
  } else if(tab==='data') {
    $('chart-title').textContent='Répartition des données';
    const dims=r.features.length;const is3d=dims>=3;
    $('chart-description').textContent=is3d?'Les trois premières entrées sont représentées. Faites glisser pour tourner la vue ; la couleur indique la cible.':dims===2?'Les deux entrées sont représentées ; la couleur indique la cible.':'Une entrée et la valeur cible, avec séparation apprentissage / test.';
    traces=['train','test'].map((part,index)=>{const ids=r.data[part];const trace={type:is3d?'scatter3d':'scatter',mode:'markers',name:index?'Test':'Apprentissage',x:ids.map(i=>r.data.x[i][0]),y:ids.map(i=>dims>=2?r.data.x[i][1]:r.data.y[i]),marker:{size:is3d?4:7,symbol:index?'diamond':'circle',opacity:.85,color:ids.map(i=>r.data.y[i]),colorscale:[[0,'#a0d9d2'],[.5,'#4b8fbf'],[1,'#2448a6']],cmin:Math.min(...r.data.y),cmax:Math.max(...r.data.y),showscale:!index,colorbar:{title:{text:r.target},thickness:10,len:.75}}};if(is3d)trace.z=ids.map(i=>r.data.x[i][2]);return trace;});
    plotLayout=layout(r.features[0],dims>=2?r.features[1]:r.target);
    plotLayout.margin.r=70;
    if(is3d)plotLayout.scene={xaxis:{title:{text:r.features[0]}},yaxis:{title:{text:r.features[1]}},zaxis:{title:{text:r.features[2]}},camera:{eye:{x:1.5,y:1.5,z:1}}};
  } else {
    $('chart-title').textContent=`Comparaison des ${r.ranking.length} noyaux`;$('chart-description').textContent='Classement par erreur moyenne de validation croisée. Une RMSE plus faible est préférable. Le test final reste indépendant.';
    fillTable($('kernel-table'),['Noyau','RMSE CV','Écart-type','Durée (s)','Alertes'],r.ranking.map(k=>[k.name,k.rmse,k.std,k.seconds,k.warnings]));return;
  }
  state.chartReady = Plotly.react('chart',traces,plotLayout,{responsive:true,displaylogo:false,scrollZoom:false,modeBarButtonsToRemove:['lasso2d','select2d','sendDataToCloud','sendChartToCloud'],toImageButtonOptions:{format:'png',filename:'atelier-gp-'+tab,scale:2}}).catch(e=>error('Le graphique n’a pas pu être affiché : '+e.message));
}
async function initialize() {
  try{const capabilities=await api('/api/capabilities');if(capabilities.cuda){$('device').dataset.cuda='true';const option=$('device').querySelector('[value=cuda]');option.disabled=false;option.textContent='GPU · '+capabilities.gpu;}}catch(e){error(e.message);}
  let id, saved;
  try { id=sessionStorage.getItem('gp-job'); saved=JSON.parse(sessionStorage.getItem('gp-configuration') || 'null'); } catch {}
  if(id){
    try {
      const job=await api('/api/jobs/'+id+'?details=1');
      setDataset(job.dataset);applyConfiguration(job.options);state.job=id;state.restored=Boolean(job.restored);
      state.trained=Boolean(saved?.trained && saved.dataset_id===job.dataset.id);
      if(job.status==='done'){state.result=job.result;renderResults();}
      else if(job.status==='running'){busy(true);display('progress-view');stage(3);$('result-state').textContent='Calcul en cours';pollJob(id);}
      else{error(job.error);remember(null);stage(2);}
      saveConfiguration();
    } catch {remember(null);id=null;}
  }
  if(!id && saved){
    try {
      const data=await api('/api/datasets/'+saved.dataset_id);
      setDataset(data);applyConfiguration(saved);state.trained=Boolean(saved.trained);stage(2);saveConfiguration();
    } catch {try{sessionStorage.removeItem('gp-configuration');}catch{}}
  }
  // Optional native browser agent access, sharing the same visible application state.
  if(document.modelContext?.registerTool){
    const lifecycle=new AbortController();window.addEventListener('pagehide',()=>lifecycle.abort(),{once:true});
    try{await document.modelContext.registerTool({name:'read_analysis_state',description:'Lire le fichier sélectionné, le statut du calcul et les métriques affichées.',inputSchema:{type:'object',properties:{},additionalProperties:false},annotations:{readOnlyHint:true,untrustedContentHint:true},execute:async input=>{if(!input||Object.keys(input).length)throw new Error('Aucun paramètre attendu.');return{filename:state.dataset?.filename||null,running:state.busy,metrics:state.result?.metrics||null};}},{signal:lifecycle.signal});}catch(e){console.debug('WebMCP indisponible',e);}
  }
}
initialize();
