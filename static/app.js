'use strict';
const $ = (id) => document.getElementById(id);
const state = {dataset:null, result:null, job:null, busy:false, trained:false, step:1, tab:'parity', poll:null, restored:false, chartReady:Promise.resolve(), configuration:{}, activeTarget:null, capabilities:null, designId:null};
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
  state.busy=value;
  document.querySelectorAll('#config input, #config select, #config button, #file-input, #clear-file, #model-input, #open-model, [data-tab], [data-input-tab], #slice-variable, #result-target, #predict-button, #prediction-inputs input, #prediction-inputs select, #excel-card input, #excel-card select, #excel-card button, #planner input, #planner select, #planner button').forEach(el=>el.disabled=value);
  updateDevice();if(typeof updateDesignFocus==='function')updateDesignFocus();$('dropzone').classList.toggle('disabled',value);updateSteps();updateRun();
}
function selectedFeatures(){return state.dataset ? state.dataset.columns.filter(c=>state.configuration[c.name]?.role==='input').map(c=>c.name) : [];}
function selectedTargets(){return state.dataset ? state.dataset.columns.filter(c=>state.configuration[c.name]?.role==='output').map(c=>c.name) : [];}
function selectedTypes(){return Object.fromEntries(Object.entries(state.configuration).map(([n,c])=>[n,c.type]));}
function currentResult(){
  const r=state.result;if(!r)return null;
  const output=r.outputs?.find(o=>o.target===state.activeTarget)||r.outputs?.[0];
  return output ? {...r,...output,data:{...r.data,y:r.data.y_by_target[output.target]}} : {...r,type:'numeric'};
}
function updateRun(){$('run').disabled=state.busy||!state.dataset||!selectedFeatures().length||!selectedTargets().length;}
function updateSteps() {
  const navigation = document.querySelector('.workflow-nav');
  navigation.classList.toggle('model-navigation', state.trained);
  navigation.setAttribute('aria-label', state.trained ? 'Vues du modèle' : 'Étapes de l’analyse');
  show('step1', !state.trained);show('step5',Boolean(state.result));
  show('new-model', state.trained);
  show('clear-file', !state.trained);
  $('new-model').disabled = state.busy;
  show('save-model', Boolean(state.result));
  $('save-model').disabled = state.busy || !state.result;
  show('report-pdf',Boolean(state.result||state.designId));$('report-pdf').disabled=state.busy;
  document.querySelectorAll('[data-step]').forEach(button => {
    const step = Number(button.dataset.step);
    button.disabled = state.busy || (step === 1 && state.trained) || (step === 2 && !state.dataset) || ([3,5].includes(step) && !state.result) || (step===4 && (!state.dataset || !selectedFeatures().length));
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
  show('input-panel', step < 3);
  document.querySelector('.workspace').classList.toggle('results-mode', step >= 3);
  $('panel-number').textContent = String(step).padStart(2, '0');
  $('panel-title').textContent = step===5 ? 'Prédire' : step===4 ? 'Plan d’expériences' : step === 3 ? (state.busy ? 'Entraînement' : 'Résultats') : 'Données';
  document.querySelector('.results-panel').setAttribute('aria-label', step===5 ? 'Prédire' : step===4 ? 'Plan d’expériences' : step === 3 ? 'Espace de résultats' : 'Exploration des données');
  [1,2,3,4,5].forEach(i=> { $('step'+i).classList.toggle('active',i===step); $('step'+i).classList.toggle('done',i<step); });
  show('dropzone', step === 1);
  show('config', Boolean(state.dataset) && step === 2);
  show('run', step === 2);
  updateSteps();
}
function navigateStep(step) {
  if (state.busy || (step === 1 && state.trained) || (step === 2 && !state.dataset) || ([3,5].includes(step) && !state.result) || (step===4 && (!state.dataset || !selectedFeatures().length))) return;
  const scrollPosition = {top:window.scrollY, left:window.scrollX, behavior:'instant'};
  // Preserve the scroll range even when the destination panel is shorter.
  document.querySelector('main').style.minHeight = `${window.innerHeight + scrollPosition.top}px`;
  error(null);
  stage(step);
  if(step===4){display('planner');setupPlanner();$('result-state').textContent=state.result?'Modèle disponible':'Plan sans modèle';}
  else if(step===5){display('predictions-view');$('result-state').textContent='Modèle disponible';}
  else if (step === 3) {
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
function display(view) { ['empty','preview','progress-view','results','planner','predictions-view'].forEach(id=>show(id,id===view)); }
function renderVariables(){
  $('variables').replaceChildren();
  state.dataset.columns.forEach(column=>{
    const config=state.configuration[column.name],row=document.createElement('div');row.className='variable-row';row.dataset.name=column.name;
    const title=document.createElement('strong');title.textContent=column.name;title.title=column.name;
    const role=document.createElement('select');role.setAttribute('aria-label','Rôle de '+column.name);
    [['ignore','Ignorer'],['input','Entrée'],['output','Sortie']].forEach(([v,t])=>role.add(new Option(t,v)));role.value=config.role;
    const type=document.createElement('select');type.setAttribute('aria-label','Type de '+column.name);
    [['numeric','Numérique'],['categorical','Catégorie']].forEach(([v,t])=>type.add(new Option(t,v)));type.value=config.type;
    role.addEventListener('change',()=>{config.role=role.value;row.dataset.role=role.value;type.hidden=role.value==='ignore';configurationChanged();});
    type.addEventListener('change',()=>{config.type=type.value;configurationChanged();});
    row.dataset.role=config.role;type.hidden=config.role==='ignore';row.append(title,role,type);$('variables').append(row);
  });filterVariables();updateDataNote();
}
function filterVariables(){const term=$('variable-search').value.trim().toLocaleLowerCase('fr');document.querySelectorAll('.variable-row').forEach(row=>row.hidden=!row.dataset.name.toLocaleLowerCase('fr').includes(term));}
$('variable-search').addEventListener('input',filterVariables);
$('all-inputs').addEventListener('click',()=>{Object.values(state.configuration).forEach(c=>{if(c.role!=='output')c.role='input';});renderVariables();configurationChanged();});
$('clear-inputs').addEventListener('click',()=>{Object.values(state.configuration).forEach(c=>{if(c.role==='input')c.role='ignore';});renderVariables();configurationChanged();});
function updateDataNote(){
  const features=selectedFeatures(),targets=selectedTargets();
  $('selection-count').textContent=`${features.length} entrée(s) · ${targets.length} sortie(s)`;
  const invalid=state.dataset.columns.filter(c=>[...features,...targets].includes(c.name)).some(c=>state.configuration[c.name].type==='numeric'?c.invalid:c.missing);
  $('data-note').textContent=invalid?'Les lignes incomplètes pour les variables sélectionnées seront exclues du calcul.':'Chaque sortie aura son propre modèle.';
  updateDevice();updateSteps();updateRun();
}
function configurationChanged() {
  updateDataNote();
  refreshInputMatrix();
  if (state.result) { state.result=null;state.job=null; display('preview'); $('result-state').textContent='Configuration modifiée'; remember(null); stage(2); }
  state.designId=null;invalidatePlanner();saveConfiguration();
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
  state.dataset=data; state.restored=false; state.result=null; state.trained=false; error(null); show('dropzone',false);  show('file-summary',true); show('config',true);
  $('file-name').textContent=data.filename; $('file-meta').textContent=`${data.rows} lignes · ${data.columns.length} colonnes`;
  document.querySelector('.file-icon').textContent=data.filename.toLowerCase().endsWith('.xlsx')?'XLSX':'CSV';
  state.activeTarget=null;state.designId=null;state.configuration=Object.fromEntries(data.columns.map(c=>[c.name,{role:c.name===data.target?'output':data.features.includes(c.name)?'input':'ignore',type:c.type||(c.numeric?'numeric':'categorical')}]));
  $('variable-search').value='';renderVariables();invalidatePlanner();
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
function resetModel() {
  if (state.busy) return;
  clearTimeout(state.poll);
  state.dataset=null;state.result=null;state.job=null;state.configuration={};state.designId=null;state.trained=false;state.poll=null;state.restored=false;
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
    sessionStorage.setItem('gp-configuration', JSON.stringify({dataset_id:state.dataset.id, target:selectedTargets()[0], targets:selectedTargets(), column_types:selectedTypes(), features:selectedFeatures(), quality:$('quality').value, device:$('device').value, kernel_search:$('kernel-search').value, length_bounds:[Number($('length-min').value),Number($('length-max').value)], trained:state.trained}));
  } catch {}
}
function applyConfiguration(options) {
  const targets=options.targets||[options.target];state.configuration=Object.fromEntries(state.dataset.columns.map(c=>[c.name,{role:targets.includes(c.name)?'output':options.features.includes(c.name)?'input':'ignore',type:options.column_types?.[c.name]||c.type||(c.numeric?'numeric':'categorical')}]));renderVariables();
  $('quality').value=options.quality;$('device').value=options.device;$('kernel-search').value=options.kernel_search || 'base';const bounds=options.length_bounds || [0.01,1000];$('length-min').value=bounds[0];$('length-max').value=bounds[1];updateQualityNote();
}
$('quality').addEventListener('change',()=>{updateQualityNote();configurationChanged();});
$('device').addEventListener('change',configurationChanged);
['kernel-search','length-min','length-max'].forEach(id=>$(id).addEventListener('change',configurationChanged));
async function startAnalysis() {
  if(state.busy || !state.dataset) throw new Error('Importez un fichier et attendez la fin du calcul en cours.');
  const options={dataset_id:state.dataset.id,target:selectedTargets()[0],targets:selectedTargets(),column_types:selectedTypes(),features:selectedFeatures(),quality:$('quality').value,device:$('device').value,kernel_search:$('kernel-search').value,length_bounds:[Number($('length-min').value),Number($('length-max').value)]};
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
function renderResults(){
  state.trained=true;state.activeTarget=state.result.targets?.includes(state.activeTarget)?state.activeTarget:state.result.target;saveConfiguration();
  display('results');stage(3);$('result-state').textContent=state.restored?'Modèle rouvert':'Analyse terminée';
  $('result-target').replaceChildren();(state.result.outputs||[state.result]).forEach(o=>$('result-target').add(new Option(o.target+(o.type==='categorical'?' · classification':' · régression'),o.target)));
  $('result-target').value=state.activeTarget;
  $('download').href='/api/jobs/'+state.job+'/predictions.csv';
  const r=state.result;$('analysis-info').textContent=`${r.features.length} entrées · ${(r.targets||[r.target]).length} sorties · ${r.train_count} points d’apprentissage · ${r.test_count} points de test · ${r.dropped} ligne(s) exclue(s) · ${r.folds} plis · ${number(r.seconds,3)} s. Modèles indépendants par sortie. Graine : ${r.split_seed||r.seed}.${state.restored?' Modèle rouvert sans entraînement ; prédictions sur CPU.':''}`;
  show('warnings',r.warnings.length>0);$('warnings').querySelector('summary').textContent=`${r.warnings.length} avertissement(s) — consulter les détails`;
  const list=$('warnings').querySelector('ul');list.replaceChildren();r.warnings.forEach(message=>{const li=document.createElement('li');li.textContent=message;list.append(li);});
  setupPrediction();setupResearch();renderResultOutput();
}
function renderResultOutput(){
  const r=currentResult();$('metrics').replaceChildren();
  const metrics=r.type==='categorical'?[['Exactitude',number(r.metrics.accuracy)+' %','Sur les données de test'],['F1 macro',number(r.metrics.f1),'Moyenne sur les classes'],['Log-loss',number(r.metrics.log_loss),'Plus faible = meilleur']]:[['Erreur RMSE',number(r.metrics.rmse),'Sur les données de test'],['Erreur MAE',number(r.metrics.mae),'Sur les données de test'],['Couverture à 95 %',number(r.metrics.coverage,3)+' %','Points dans l’intervalle']];
  metrics.forEach(([label,value,sub])=>{const div=document.createElement('div');div.className='metric';[['metric-label',label],['metric-value',value],['metric-sub',sub]].forEach(([cls,text])=>{const el=document.createElement('div');el.className=cls;el.textContent=text;div.append(el);});$('metrics').append(div);});
  $('best-kernel').textContent=r.target+' · noyau retenu : '+r.kernel+' · '+r.device.toUpperCase();
  $('slice-variable').replaceChildren();r.features.forEach((name,i)=>$('slice-variable').add(new Option(name,String(i))));
  for(const id of ['data-x-axis','data-y-axis','data-z-axis']){
    const select=$(id);select.replaceChildren();if(id==='data-z-axis')select.add(new Option('Aucun',''));
    r.features.forEach((name,j)=>select.add(new Option(name,String(j))));if(id==='data-y-axis')select.add(new Option(r.target+' (sortie)','target'));
  }
  $('data-y-axis').value=r.features.length>1?'1':'target';renderBounds();chooseTab(state.tab);
}
$('result-target').addEventListener('change',()=>{state.activeTarget=$('result-target').value;renderResultOutput();});
['data-x-axis','data-y-axis','data-z-axis'].forEach(id=>$(id).addEventListener('change',renderChart));
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
  const r=currentResult();if(!r)return;
  const tab=state.tab;let traces=[],plotLayout;
  show('chart-heading',true);show('distribution-ranges',false);show('kernel-diagnostics',tab==='kernels' && Boolean(r.bound_diagnostics));show('data-axes',tab==='data');
  show('slice-variable',tab==='slices');show('kernel-table',tab==='kernels');show('chart',tab!=='kernels' && tab!=='matrix' && tab!=='coverage');show('result-matrix',tab==='matrix');
  if(tab==='matrix'){
    $('chart-title').textContent='Matrice des entrées du modèle';
    $('chart-description').textContent='';
    renderInputMatrix('result-matrix', state.dataset, r.features);
    return;
  }
  if(tab==='parity' && r.type==='categorical'){
    $('chart-title').textContent='Matrice de confusion · '+r.target;$('chart-description').textContent='Test indépendant · lignes : classe réelle ; colonnes : classe prédite.';
    traces=[{type:'heatmap',x:r.classes,y:r.classes,z:r.confusion,colorscale:[[0,'#f1f5fc'],[1,blue]],text:r.confusion,texttemplate:'%{text}',hovertemplate:'Réel : %{y}<br>Prédit : %{x}<br>Effectif : %{z}<extra></extra>'}];plotLayout=layout('Classe prédite','Classe réelle');plotLayout.yaxis.autorange='reversed';
  } else if(tab==='parity'){
    $('chart-title').textContent='Valeurs réelles et prédites';$('chart-description').textContent=`${r.target} · Les barres représentent l’intervalle prédictif nominal à 95 %.`;
    const p=r.parity,lo=Math.min(...p.actual,...p.predicted),hi=Math.max(...p.actual,...p.predicted);
    traces=[{x:[lo,hi],y:[lo,hi],mode:'lines',name:'Prédiction parfaite',line:{color:gray,width:1.5,dash:'dash'},hoverinfo:'skip'},{x:p.actual,y:p.predicted,mode:'markers',name:'Prédictions',marker:{color:blue,size:7,opacity:.85},error_y:{type:'data',array:p.std.map(s=>1.96*s),color:'#afc7ef',thickness:1,width:2},hovertemplate:'Réel : %{x:.5g}<br>Prédit : %{y:.5g}<extra></extra>'}];
    plotLayout=layout('Valeur réelle ('+r.target+')','Valeur prédite ('+r.target+')');
  } else if(tab==='slices') {
    const s=r.slices[Number($('slice-variable').value)||0];
    $('chart-title').textContent='Réponse du modèle selon '+s.name;$('chart-description').textContent='Les autres entrées sont fixées à leur médiane numérique ou à leur catégorie la plus fréquente dans l’apprentissage.';
    if(r.type==='categorical'){
      traces=s.classes.map((name,j)=>({x:s.x,y:s.probabilities.map(p=>p[j]),mode:s.type==='categorical'?'lines+markers':'lines',name}));plotLayout=layout(s.name,'Probabilité de classe');plotLayout.yaxis.range=[0,1];
    }else if(s.type==='categorical'){
      traces=[{x:s.x,y:s.mean,type:'bar',marker:{color:blue},error_y:{type:'data',array:s.upper.map((v,i)=>v-s.mean[i])},name:'Prédiction'}];plotLayout=layout(s.name,r.target);
    }else{
      traces=[{x:s.x,y:s.lower,mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip'},{x:s.x,y:s.upper,mode:'lines',line:{width:0},fill:'tonexty',fillcolor:'rgba(49,107,209,.14)',name:'Intervalle à 95 %',hoverinfo:'skip'},{x:s.train_x,y:s.train_y,mode:'markers',marker:{color:gray,size:5,opacity:.5},name:'Apprentissage'},{x:s.x,y:s.mean,mode:'lines',line:{color:blue,width:2.5},name:'Prédiction'}];plotLayout=layout(s.name,r.target);
    }
    if(s.type==='categorical')plotLayout.xaxis.type='category';
  } else if(tab==='data') {
    $('chart-title').textContent='Répartition des données';$('chart-description').textContent='Choisissez les entrées à représenter. La couleur indique la sortie sélectionnée.';
    const ix=Number($('data-x-axis').value),yv=$('data-y-axis').value,iz=$('data-z-axis').value,is3d=iz!=='';
    const output=r.type==='categorical'?r.data.y.map(v=>r.classes.indexOf(v)):r.data.y;
    traces=['train','test'].map((part,index)=>{const ids=r.data[part];const trace={type:is3d?'scatter3d':'scatter',mode:'markers',name:index?'Test':'Apprentissage',x:ids.map(i=>r.data.x[i][ix]),y:ids.map(i=>yv==='target'?r.data.y[i]:r.data.x[i][Number(yv)]),marker:{size:is3d?4:7,symbol:index?'diamond':'circle',opacity:.8,color:ids.map(i=>output[i]),colorscale:[[0,'#a0d9d2'],[1,'#2448a6']],cmin:Math.min(...output),cmax:Math.max(...output),showscale:!index,colorbar:{title:{text:r.target},thickness:10,len:.75,...(r.type==='categorical'?{tickvals:r.classes.map((_,i)=>i),ticktext:r.classes}:{})}}};if(is3d)trace.z=ids.map(i=>r.data.x[i][Number(iz)]);return trace;});
    const kind=j=>r.input_schema?.[j]?.type||'numeric',xcat=kind(ix)==='categorical',ycat=yv==='target'?r.type==='categorical':kind(Number(yv))==='categorical';
    plotLayout=layout(r.features[ix],yv==='target'?r.target:r.features[Number(yv)]);plotLayout.margin.r=90;
    if(xcat)plotLayout.xaxis.type='category';if(ycat)plotLayout.yaxis.type='category';
    if(is3d)plotLayout.scene={xaxis:{title:{text:r.features[ix]},type:xcat?'category':'linear'},yaxis:{title:{text:yv==='target'?r.target:r.features[Number(yv)]},type:ycat?'category':'linear'},zaxis:{title:{text:r.features[Number(iz)]},type:kind(Number(iz))==='categorical'?'category':'linear'}};
  } else if(tab==='distribution'){
    $('chart-title').textContent='Distribution de '+r.target;
    if(r.type==='categorical'){
      const counts=r.classes.map(c=>r.data.y.filter(v=>v===c).length);traces=[{type:'bar',x:r.classes,y:counts,marker:{color:blue}}];$('chart-description').textContent='Effectifs des classes dans toutes les observations retenues.';
    }else{
      const lo=Math.min(...r.data.y),hi=Math.max(...r.data.y),width=(hi-lo)/12||1,counts=Array(12).fill(0);r.data.y.forEach(v=>counts[Math.min(11,Math.floor((v-lo)/width))]++);
      traces=[{type:'bar',x:counts.map((_,i)=>lo+(i+.5)*width),y:counts,width:width*.9,marker:{color:blue}}];const rare=counts.map((n,i)=>({n,i})).sort((a,b)=>a.n-b.n).slice(0,3);$('chart-description').textContent='';show('distribution-ranges',true);$('distribution-ranges').replaceChildren();const title=document.createElement('h4');title.textContent='Plages les moins représentées';const list=document.createElement('ol');rare.forEach(({n,i})=>{const item=document.createElement('li'),range=document.createElement('span'),count=document.createElement('strong');range.textContent=`[${number(lo+i*width,4)} ; ${number(lo+(i+1)*width,4)}]`;count.textContent=n+' point'+(n>1?'s':'');item.append(range,count);list.append(item);});$('distribution-ranges').append(title,list);
    }plotLayout=layout(r.target,'Nombre d’observations');
  } else {
    $('chart-title').textContent=`Comparaison des ${r.ranking.length} noyaux`;$('chart-description').textContent='Classement par score moyen de la validation croisée. Un score plus faible est préférable.';
    fillTable($('kernel-table'),['Noyau',(r.score_name||'RMSE')+' CV','Écart-type','Durée (s)','Alertes'],r.ranking.map(k=>[k.name,k.score??k.rmse,k.std,k.seconds,k.warnings]));document.querySelectorAll('#kernel-table tbody tr').forEach((row,i)=>{if(r.ranking[i].name===r.kernel){row.classList.add('selected-kernel-row');const badge=document.createElement('span');badge.className='selected-kernel-badge';badge.textContent='Retenu';row.cells[0].append(badge);}});return;
  }
  state.chartReady = Plotly.react('chart',traces,plotLayout,{responsive:true,displaylogo:false,scrollZoom:false,modeBarButtonsToRemove:['lasso2d','select2d','sendDataToCloud','sendChartToCloud'],toImageButtonOptions:{format:'png',filename:'atelier-gp-'+tab,scale:2}}).catch(e=>error('Le graphique n’a pas pu être affiché : '+e.message));
}
function updateDevice(){
  const option=$('device').querySelector('[value=cuda]'), cap=state.capabilities;
  const outputs=selectedTargets(),onlyClasses=outputs.length>0&&outputs.every(n=>state.configuration[n].type==='categorical');
  option.disabled=!cap?.cuda||onlyClasses||$('kernel-search').value==='extended';
  option.textContent=cap?.cuda?'GPU · '+cap.gpu:'GPU · indisponible sur ce serveur';
  if(option.disabled&&$('device').value==='cuda')$('device').value='cpu';
  $('device-note').textContent=(cap?cap.host+' : '+cap.reason:'Vérification du serveur…')+(onlyClasses?' La classification GP est calculée sur CPU.':outputs.some(n=>state.configuration[n].type==='categorical')?' Les sorties numériques peuvent utiliser CUDA ; les classes restent sur CPU.':'');
}
async function refreshCapabilities(refresh=true){
  try{state.capabilities=await api('/api/capabilities'+(refresh?'?refresh=1':''));const c=state.capabilities;$('gpu-details').textContent=`Serveur : ${c.host}\nPython : ${c.python}\nPyTorch : ${c.torch_version||'absent'}\nCUDA PyTorch : ${c.cuda_runtime||'absent'}\nGPU : ${c.gpu||'non accessible'}\nCUDA_VISIBLE_DEVICES : ${c.visible_devices??'non défini'}\nDiagnostic : ${c.reason}\n\nCommande sur le serveur :\n${c.python} tools/diagnose_gpu.py`;updateDevice();}catch(e){$('device-note').textContent=e.message;}
}
$('refresh-gpu').addEventListener('click',()=>refreshCapabilities());
async function initialize() {
  await refreshCapabilities(false);
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
window.addEventListener('DOMContentLoaded',initialize,{once:true});
