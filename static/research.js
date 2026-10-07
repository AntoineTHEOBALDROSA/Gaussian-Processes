'use strict';
const research = {workbook:null, info:null, proposals:null, distributionKey:null};

function decimal(text, label) {
  const raw = String(text).trim().replace(',', '.');
  if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(raw) || !Number.isFinite(Number(raw))) throw new Error('Saisissez un nombre fini pour '+label+'.');
  return Number(raw);
}
function setupResearch() {
  research.workbook=null;research.info=null;research.proposals=null;research.distributionKey=null;
  ['excel-settings','excel-mapping-panel','coverage-results'].forEach(id=>show(id,false));
  $('excel-file').value='';$('excel-filename').textContent='';$('excel-message').textContent='';$('coverage-message').textContent='';
  $('excel-overwrite').checked=false;$('coverage-focus').checked=false;show('coverage-focus-fields',false);
  $('excel-output-name').value=state.result.target+'_prédit';
  $('coverage-domain').replaceChildren();
  state.result.features.forEach((name,j)=>{
    const values=state.result.data.x.map(row=>row[j]);
    const group=document.createElement('fieldset'),legend=document.createElement('legend');legend.textContent=name;group.append(legend);
    for (const [label,value,key] of [['Minimum',Math.min(...values),'minimum'],['Maximum',Math.max(...values),'maximum']]) {
      const field=document.createElement('label'),input=document.createElement('input');field.textContent=label;
      input.type='text';input.inputMode='decimal';input.value=String(value);input.dataset.bound=key;field.append(input);group.append(field);
    }
    const label=document.createElement('label'),check=document.createElement('input');label.className='check-label';
    check.type='checkbox';check.checked=values.every(Number.isInteger);check.dataset.bound='integer';label.append(check,document.createTextNode('Entiers'));group.append(label);
    group.dataset.feature=name;$('coverage-domain').append(group);
  });
  ['coverage-x','coverage-y'].forEach(id=>{$(id).replaceChildren();state.result.features.forEach((f,j)=>$(id).add(new Option(f,String(j))));});
  $('coverage-y').value=String(Math.min(1,state.result.features.length-1));
  show('coverage-y',state.result.features.length>1);
  const diagnostics=state.result.bound_diagnostics || [];
  fillTable($('bounds-table'),['Paramètre','Valeur','Minimum','Maximum','Position'],diagnostics.map(d=>[d.name,d.value,d.lower,d.upper,d.hit==='lower'?'Borne minimale':d.hit==='upper'?'Borne maximale':'Dans les bornes']));
  const notes=[...new Set(diagnostics.filter(d=>d.hit).map(d=>d.explanation))];
  $('bounds-note').textContent=notes.length?notes.join(' '):'Aucun paramètre du modèle final n’est proche de ses bornes.';
}

document.getElementById('excel-choose').addEventListener('click',()=>{if(!state.busy)$('excel-file').click();});
document.getElementById('excel-file').addEventListener('change',async event=>{
  const file=event.target.files[0];if(!file||state.busy)return;
  if(!/\.xlsx$/i.test(file.name)||file.size>20*1024*1024){error('Choisissez un fichier .xlsx de moins de 20 Mo.');return;}
  research.workbook=null;research.info=null;show('excel-settings',false);show('excel-mapping-panel',false);$('excel-filename').textContent='';
  const form=new FormData();form.append('file',file);busy(true);error(null);$('excel-message').textContent='Lecture du classeur…';
  try {
    const data=await api('/api/workbooks',{method:'POST',body:form});research.workbook=data;
    $('excel-filename').textContent=data.filename;$('excel-sheet').replaceChildren();data.sheets.forEach(s=>$('excel-sheet').add(new Option(s.name,s.name)));
    $('excel-header').value='1';show('excel-settings',true);await readExcelColumns();
  } catch(e){error(e.message);$('excel-message').textContent='';}
  finally {busy(false);$('excel-file').value='';}
});
async function readExcelColumns() {
  if(!research.workbook)return;
  show('excel-mapping-panel',false);research.info=null;
  const info=await api(`/api/workbooks/${research.workbook.id}/inspect`,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({sheet:$('excel-sheet').value,header_row:Number($('excel-header').value)})});
  research.info=info;$('excel-mapping').replaceChildren();
  const addColumns=select=>info.columns.forEach(c=>select.add(new Option(`${c.letter} — ${c.name || 'Sans titre'}${c.samples.filter(Boolean).length ? ' (ex. '+c.samples.filter(Boolean)[0]+')' : ''}`,String(c.index))));
  state.result.features.forEach(name=>{
    const field=document.createElement('label'),select=document.createElement('select');field.textContent='Entrée du modèle : '+name;
    select.dataset.feature=name;select.add(new Option('Choisir une colonne',''));addColumns(select);
    const match=info.columns.find(c=>c.name.trim().toLowerCase()===name.trim().toLowerCase());if(match)select.value=String(match.index);
    field.append(select);$('excel-mapping').append(field);
  });
  $('excel-output').replaceChildren();addColumns($('excel-output'));
  $('excel-output').add(new Option('Ajouter une nouvelle colonne',String(info.columns.length+1)));
  $('excel-output').value=String(info.columns.length+1);updateOutputName();
  $('excel-first').value=info.first_row;$('excel-last').value=info.last_row;
  $('excel-message').textContent=info.total_rows>info.last_row?'La feuille est longue : le traitement est limité à 10 000 lignes à la fois. Ajustez la plage si nécessaire.':'Vérifiez les associations de colonnes et la plage de lignes.';
  show('excel-mapping-panel',true);
}
document.getElementById('excel-inspect').addEventListener('click',async()=>{
  if(state.busy)return;busy(true);error(null);
  try{await readExcelColumns();}catch(e){error(e.message);}finally{busy(false);}
});
['excel-sheet','excel-header'].forEach(id=>document.getElementById(id).addEventListener('change',()=>{research.info=null;show('excel-mapping-panel',false);$('excel-message').textContent='Cliquez sur « Lire les colonnes » pour cette feuille et cette ligne de titres.';}));
function updateOutputName(){show('excel-output-name-label',Number($('excel-output').value)===research.info?.columns.length+1);}
document.getElementById('excel-output').addEventListener('change',updateOutputName);
document.getElementById('excel-predict').addEventListener('click',async()=>{
  if(state.busy||!research.info)return;
  try {
    const mapping=Object.fromEntries([...document.querySelectorAll('#excel-mapping select')].map(s=>{if(!s.value)throw new Error('Choisissez la colonne de '+s.dataset.feature+'.');return[s.dataset.feature,Number(s.value)];}));
    const payload={workbook_id:research.workbook.id,sheet:research.info.sheet,header_row:research.info.header_row,
      first_row:Number($('excel-first').value),last_row:Number($('excel-last').value),mapping,
      output_column:Number($('excel-output').value),output_name:$('excel-output-name').value,overwrite:$('excel-overwrite').checked};
    busy(true);error(null);$('excel-message').textContent='Calcul des prédictions et préparation du classeur…';
    const filename=research.workbook.filename.replace(/\.xlsx$/i,'')+'-predictions.xlsx';
    const headers=await downloadResponse(`/api/jobs/${state.job}/predict-excel`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)},filename);
    const summary=JSON.parse(headers.get('X-GP-Summary'));
    $('excel-message').textContent=`Classeur téléchargé : ${summary.predicted} prédictions écrites en colonne ${summary.output}. ${summary.skipped_existing} cellules déjà remplies conservées. ${summary.skipped_invalid} lignes sans entrées valides.${summary.invalid_rows.length?' Lignes concernées : '+summary.invalid_rows.join(', ')+(summary.skipped_invalid>20?'…':'')+'.':''}`;
  }catch(e){error(e.message);$('excel-message').textContent='';}finally{busy(false);}
});

function renderCoverage() {
  if(research.distributionKey===state.job){Plotly.Plots.resize($('target-distribution'));if(research.proposals)renderSuggestions();return;}
  research.distributionKey=state.job;
  const y=state.result.data.y, minimum=Math.min(...y), maximum=Math.max(...y);
  const bins=12,width=(maximum-minimum)/bins||1,counts=Array(bins).fill(0);
  y.forEach(v=>counts[Math.min(bins-1,Math.max(0,Math.floor((v-minimum)/width)))]++);
  const centers=counts.map((_,i)=>minimum+(i+.5)*width);
  state.coverageReady=Plotly.react('target-distribution',[{type:'bar',x:centers,y:counts,width:width*.9,marker:{color:'#8eb0e6'}}],
    {...layout(state.result.target+' observé','Nombre d’observations'),height:280},localPlotConfig).catch(e=>error(e.message));
  const rare=counts.map((n,i)=>({n,i})).sort((a,b)=>a.n-b.n).slice(0,3);
  $('rare-ranges').textContent='Plages de sortie les moins représentées (12 classes de largeur égale) : '+rare.map(({n,i})=>`[${number(minimum+i*width,4)} ; ${number(minimum+(i+1)*width,4)}] : ${n} observations`).join(' · ')+'.';
  if(research.proposals)renderSuggestions();
}
document.getElementById('coverage-focus').addEventListener('change',()=>show('coverage-focus-fields',$('coverage-focus').checked));
document.getElementById('coverage-form').addEventListener('input',()=>{research.proposals=null;show('coverage-results',false);$('coverage-message').textContent='';});
document.getElementById('coverage-form').addEventListener('submit',async event=>{
  event.preventDefault();if(state.busy)return;
  try {
    const domain=[...document.querySelectorAll('#coverage-domain fieldset')].map(field=>({name:field.dataset.feature,
      minimum:decimal(field.querySelector('[data-bound=minimum]').value,field.dataset.feature+' minimum'),
      maximum:decimal(field.querySelector('[data-bound=maximum]').value,field.dataset.feature+' maximum'),integer:field.querySelector('[data-bound=integer]').checked}));
    const target_range=$('coverage-focus').checked?[decimal($('coverage-ymin').value,'la sortie minimale'),decimal($('coverage-ymax').value,'la sortie maximale')]:null;
    busy(true);error(null);$('coverage-message').textContent='Exploration du domaine et sélection de configurations complémentaires…';
    const data=await api(`/api/jobs/${state.job}/suggestions`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({domain,count:Number($('coverage-count').value),target_range})});
    research.proposals=data;renderSuggestions();
    $('coverage-message').textContent=`${data.rows.length} configurations proposées parmi ${data.candidates} candidats, en évitant les ${data.known_count} configurations connues. ${target_range?'Le filtre utilise la sortie estimée, pas une mesure réelle.':''}`;
  }catch(e){error(e.message);$('coverage-message').textContent='';}finally{busy(false);}
});
function renderSuggestions() {
  const data=research.proposals;if(!data)return;show('coverage-results',true);
  fillTable($('suggestions-table'),['Priorité',...data.features,data.target+' estimé','Incertitude de la fonction (σ)','Distance aux données','Hors plages d’apprentissage'],
    data.rows.map((r,i)=>[i+1,...r.values,r.mean,r.latent_std,r.distance,r.extrapolation?'Oui':'Non']));
  const i=Number($('coverage-x').value),j=Number($('coverage-y').value),one=data.features.length===1;
  const known=state.result.data;
  const traces=[{x:known.x.map(r=>r[i]),y:one?known.y:known.x.map(r=>r[j]),mode:'markers',name:'Configurations connues',marker:{color:'#9eafc6',size:5,opacity:.45}},
    {x:data.rows.map(r=>r.values[i]),y:data.rows.map(r=>one?r.mean:r.values[j]),mode:'markers',name:'Simulations proposées',
     marker:{color:'#cd654c',size:12,symbol:'diamond'},text:data.rows.map((r,k)=>`Priorité ${k+1}<br>${data.target} estimé : ${number(r.mean,6)}<br>σ fonction : ${number(r.latent_std,4)}`),hovertemplate:'%{text}<extra></extra>'}];
  state.suggestionsReady=Plotly.react('coverage-chart',traces,{...layout(data.features[i],one?data.target+' (propositions : estimation)':data.features[j]),height:360},localPlotConfig).catch(e=>error(e.message));
}
['coverage-x','coverage-y'].forEach(id=>document.getElementById(id).addEventListener('change',renderSuggestions));
document.getElementById('coverage-download').addEventListener('click',()=>{
  const data=research.proposals;if(!data)return;
  const quote=value=>'"'+String(value).replaceAll('"','""')+'"';
  const text='\uFEFF'+[data.features.concat(data.target).map(quote).join(';'),...data.rows.map(r=>r.values.map(v=>String(v).replace('.',',')).concat('').join(';'))].join('\r\n');
  const url=URL.createObjectURL(new Blob([text],{type:'text/csv;charset=utf-8'}));
  const link=document.createElement('a');link.href=url;link.download=downloadName('simulations-a-demander.csv');document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),60000);
});
