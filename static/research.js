'use strict';
const research={workbook:null,info:null,proposals:null,plannerKey:null,schema:[]};
function decimal(text,label){const raw=String(text).trim().replace(',','.');if(!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(raw)||!Number.isFinite(Number(raw)))throw new Error('Saisissez un nombre fini pour '+label+'.');return Number(raw);}
function categoryText(value){return typeof value==='boolean'?(value?'True':'False'):String(value).trim();}
function columnValues(name,type){const j=state.dataset.columns.findIndex(c=>c.name===name);return state.dataset.records.map(r=>r[j]).filter(v=>v!==null&&String(v).trim()!=='').map(v=>{if(type==='categorical')return categoryText(v);try{return decimal(v,name);}catch{return null;}}).filter(v=>v!==null);}
function setupResearch(){
  research.workbook=null;research.info=null;invalidatePlanner();
  ['excel-settings','excel-mapping-panel'].forEach(id=>show(id,false));$('excel-file').value='';$('excel-filename').textContent='';$('excel-message').textContent='';$('excel-overwrite').checked=false;
}
function renderBounds(){
  const result=currentResult(),variants=result.kernel_variants||[];const select=$('kernel-class');select.replaceChildren();
  variants.forEach((variant,index)=>select.add(new Option('Classe : '+variant.label,String(index))));show('kernel-class',variants.length>0);show('kernel-class-label',variants.length>0);renderKernelParameters();
}
$('kernel-class').addEventListener('change',renderKernelParameters);
function invalidatePlanner(){research.plannerKey=null;research.proposals=null;if(document.getElementById('design-results'))show('design-results',false);}
function plannerSchema(){
  if(state.result)return state.result.input_ranges.map(r=>({...r,type:r.type||'numeric'}));
  return selectedFeatures().map(name=>{const type=state.configuration[name].type,values=columnValues(name,type);return type==='categorical'?{name,type,categories:[...new Set(values)],mode:values[0]}:{name,type,minimum:values.length?Math.min(...values):0,maximum:values.length?Math.max(...values):1,median:values.length?values.reduce((a,b)=>a+b,0)/values.length:0};});
}
function field(parent,label,key,value,type='text'){const holder=document.createElement('label'),input=document.createElement('input');holder.textContent=label;input.type=type;input.value=String(value);input.dataset.bound=key;if(type==='text')input.inputMode='decimal';holder.append(input);parent.append(holder);return input;}
function setupPlanner(){
  const schema=plannerSchema(),key=JSON.stringify([state.dataset.id,state.job,selectedFeatures(),selectedTypes()]);if(research.plannerKey===key)return;
  research.plannerKey=key;research.schema=schema;research.proposals=null;state.designId=null;show('design-results',false);$('design-message').textContent='';$('design-domain').replaceChildren();$('design-constraints').replaceChildren();show('no-constraints',true);$('domain-search').value='';
  schema.forEach((r,j)=>{
    const box=document.createElement('details');box.className='design-variable';box.open=schema.length<=8;box.dataset.index=j;
    const summary=document.createElement('summary');summary.textContent=r.name+' · '+(r.type==='categorical'?'catégories':'numérique');box.append(summary);
    const grid=document.createElement('div');grid.className='control-grid';box.append(grid);
    if(r.type==='categorical'){
      const list=document.createElement('div');list.className='category-weights';
      r.categories.forEach(value=>{const label=document.createElement('label'),check=document.createElement('input'),weight=document.createElement('input'),name=document.createElement('span');check.type='checkbox';check.checked=true;check.value=value;check.dataset.category='true';name.textContent=value;weight.type='number';weight.min='0';weight.step='any';weight.value='1';weight.dataset.weight='true';weight.setAttribute('aria-label','Poids de '+value+' pour '+r.name);label.append(check,name,weight);list.append(label);});box.append(list);
    }else{
      const values=columnValues(r.name,'numeric');field(grid,'Minimum','minimum',r.minimum);field(grid,'Maximum','maximum',r.maximum);
      const label=document.createElement('label'),select=document.createElement('select');label.textContent='Loi marginale';select.dataset.bound='distribution';[['uniform','Uniforme'],['normal','Normale tronquée'],['loguniform','Log-uniforme']].forEach(([v,t])=>select.add(new Option(t,v)));label.append(select);grid.append(label);
      const normal=document.createElement('div');normal.className='control-grid normal-parameters';normal.hidden=true;field(normal,'Moyenne','mean',r.median);field(normal,'Écart-type','std',Math.max((r.maximum-r.minimum)/6,1e-6));box.append(normal);
      select.addEventListener('change',()=>normal.hidden=select.value!=='normal');
      const checkLabel=document.createElement('label'),check=document.createElement('input');checkLabel.className='check-label';check.type='checkbox';check.dataset.bound='integer';check.checked=values.length>0&&values.every(Number.isInteger);checkLabel.append(check,document.createTextNode('Valeurs entières'));box.append(checkLabel);
    }
    $('design-domain').append(box);
  });
  const outputs=state.result?.outputs|| (state.result?[{...state.result,type:'numeric'}]:[]);
  $('design-target').replaceChildren();$('design-target').add(new Option('Toutes les sorties, poids égaux',''));outputs.forEach(o=>$('design-target').add(new Option(o.target,o.target)));
  $('design-mode').querySelector('[value=active]').disabled=!state.result;$('design-mode').value=state.result?'active':'lhs';updateDesignMode();
  ['design-x','design-y'].forEach(id=>{$(id).replaceChildren();if(id==='design-y')$(id).add(new Option('Projection 1D',''));schema.forEach((r,j)=>$(id).add(new Option(r.name,'feature:'+j)));outputs.forEach(o=>{if(id==='design-y')$(id).add(new Option(o.target+' prédit','output:'+o.target));});});
  $('design-y').value=schema.length>1?'feature:1':outputs.length?'output:'+outputs[0].target:'';updateSteps();renderDesign();
}
function updateDesignMode(){const active=$('design-mode').value==='active';show('design-pool-field',active);show('design-model-settings',active);show('design-exclude-field',!active);$('design-count').max=active?'200':'3000';$('design-count').value=active?'10':'30';$('design-mode-note').textContent=active?'Le LHS forme un réservoir. Le modèle choisit des points éloignés et incertains, puis diversifie le lot. Les distributions décrivent le réservoir, pas nécessairement le lot retenu.':'Créez un plan initial sans entraînement. Le LHS répartit les quantiles selon la loi choisie pour chaque entrée.';updateDesignFocus();}
$('design-mode').addEventListener('change',updateDesignMode);
$('domain-search').addEventListener('input',()=>{const term=$('domain-search').value.toLocaleLowerCase('fr');document.querySelectorAll('.design-variable').forEach(box=>box.hidden=!research.schema[Number(box.dataset.index)].name.toLocaleLowerCase('fr').includes(term));});
function addConstraint(){
  const row=document.createElement('section');row.className='constraint-card';
  const header=document.createElement('div');header.className='constraint-header';
  const type=document.createElement('select');type.dataset.rule='type';type.setAttribute('aria-label','Type de contrainte');
  [['comparison','Comparer deux valeurs'],['interval','Inclure / exclure un intervalle'],['membership','Autoriser / exclure des catégories'],['linear','Somme pondérée / budget'],['distance','Écart entre deux valeurs']].forEach(([value,label])=>{const option=new Option(label,value);option.disabled=value==='membership'?!research.schema.some(s=>s.type==='categorical'):value!=='comparison'&&!research.schema.some(s=>s.type==='numeric');type.add(option);});
  const remove=document.createElement('button');remove.type='button';remove.className='icon-button';remove.textContent='×';remove.setAttribute('aria-label','Supprimer cette contrainte');
  remove.addEventListener('click',()=>{row.remove();show('no-constraints',!$('design-constraints').children.length);clearDesignResult();});header.append(type,remove);row.append(header);
  const body=document.createElement('div');body.className='constraint-body';row.append(body);
  const select=(label,key,options)=>{const holder=document.createElement('label'),input=document.createElement('select');holder.textContent=label;input.dataset.rule=key;input.setAttribute('aria-label',label);options.forEach(([v,t])=>input.add(new Option(t,v)));holder.append(input);body.append(holder);return input;};
  const input=(label,key,value)=>{const holder=document.createElement('label'),entry=document.createElement('input');holder.textContent=label;entry.dataset.rule=key;entry.setAttribute('aria-label',label);entry.type='text';entry.inputMode='decimal';entry.value=value;holder.append(entry);body.append(holder);return entry;};
  const build=()=>{
    body.replaceChildren();const kind=type.value,numeric=research.schema.filter(s=>s.type==='numeric'),choices=items=>items.map(s=>[s.name,s.name]);
    if(kind==='linear'){
      const terms=document.createElement('div');terms.className='linear-terms';body.append(terms);
      numeric.forEach((s,index)=>{const holder=document.createElement('label'),coefficient=document.createElement('input');holder.textContent=s.name+' ×';coefficient.type='text';coefficient.inputMode='decimal';coefficient.value=index<2?'1':'0';coefficient.dataset.coefficient=s.name;coefficient.setAttribute('aria-label','Coefficient de '+s.name);holder.append(coefficient);terms.append(holder);});
      select('Comparateur','operator',[['<=','≤'],['>=','≥'],['==','= avec tolérance']]);input('Limite de la somme','value','1');const tolerance=input('Tolérance pour l’égalité','tolerance','0.01');tolerance.parentElement.hidden=true;body.querySelector('[data-rule=operator]').addEventListener('change',event=>tolerance.parentElement.hidden=event.target.value!=='==');return;
    }
    const available=kind==='membership'?research.schema.filter(s=>s.type==='categorical'):kind==='interval'||kind==='distance'?numeric:research.schema;
    const left=select('Entrée à contraindre','left',choices(available));
    if(kind==='interval'){
      select('Condition','operator',[['inside','Dans l’intervalle (bornes incluses)'],['outside','Hors de l’intervalle']]);const lo=input('Minimum inclus','lower','0'),hi=input('Maximum inclus','upper','1');const defaults=()=>{const s=research.schema.find(s=>s.name===left.value);lo.value=s.minimum;hi.value=s.maximum;};left.addEventListener('change',defaults);defaults();return;
    }
    if(kind==='membership'){
      select('Condition','operator',[['in','Parmi les catégories'],['not_in','Hors des catégories']]);const list=document.createElement('div');list.className='constraint-categories';body.append(list);
      const values=()=>{list.replaceChildren();research.schema.find(s=>s.name===left.value).categories.forEach(value=>{const label=document.createElement('label'),check=document.createElement('input');check.type='checkbox';check.dataset.allowed='true';check.value=value;check.checked=true;label.append(check,document.createTextNode(value));list.append(label);});};left.addEventListener('change',values);values();return;
    }
    const op=select('Comparateur','operator',[]),side=select('Comparer avec','kind',[['feature','Une entrée'],['constant','Une constante']]),right=select('Entrée comparée','feature',[]),constant=input('Valeur constante','constant','0');
    const refresh=()=>{const cat=research.schema.find(s=>s.name===left.value).type==='categorical';op.replaceChildren();(kind==='distance'?[['>=','Écart au moins égal à'],['<=','Écart au plus égal à']]:cat?[['==','='],['!=','≠']]:[['<','<'],['<=','≤'],['>','>'],['>=','≥'],['==','='],['!=','≠']]).forEach(([v,t])=>op.add(new Option(t,v)));right.replaceChildren();research.schema.filter(s=>(s.type==='categorical')===cat).forEach(s=>right.add(new Option(s.name,s.name)));if(right.options.length>1)right.selectedIndex=1;constant.inputMode=cat?'text':'decimal';};
    const updateSide=()=>{right.parentElement.hidden=side.value!=='feature';constant.parentElement.hidden=side.value!=='constant';};left.addEventListener('change',refresh);side.addEventListener('change',updateSide);refresh();updateSide();if(kind==='distance')input('Écart absolu','value','0.1');
  };
  type.addEventListener('change',build);build();$('design-constraints').append(row);show('no-constraints',false);clearDesignResult();
}
function designConstraints(){
  return [...document.querySelectorAll('.constraint-card')].map(row=>{
    const v=key=>row.querySelector('[data-rule='+key+']').value,type=v('type');
    if(type==='linear')return{type,terms:[...row.querySelectorAll('[data-coefficient]')].map(input=>({feature:input.dataset.coefficient,coefficient:decimal(input.value,'le coefficient')})).filter(term=>term.coefficient!==0),operator:v('operator'),value:decimal(v('value'),'la limite'),tolerance:decimal(v('tolerance'),'la tolérance')};
    const rule={type,left:v('left'),operator:v('operator')};
    if(type==='interval')return{...rule,lower:decimal(v('lower'),'le minimum'),upper:decimal(v('upper'),'le maximum')};
    if(type==='membership')return{...rule,values:[...row.querySelectorAll('[data-allowed]:checked')].map(input=>input.value)};
    const categorical=research.schema.find(s=>s.name===rule.left).type==='categorical';rule.right={kind:v('kind'),value:v('kind')==='feature'?v('feature'):categorical?v('constant'):decimal(v('constant'),'la constante')};if(type==='distance')rule.value=decimal(v('value'),'l’écart');return rule;
  });
}
$('add-constraint').addEventListener('click',addConstraint);
function updateDesignFocus(){
  const output=state.result?.outputs?.find(o=>o.target===$('design-target').value)||(state.result&&state.result.target===$('design-target').value?{...state.result,type:'numeric'}:null);
  $('design-focus').disabled=state.busy||!output;if(!output)$('design-focus').checked=false;
  const visible=$('design-focus').checked&&Boolean(output);show('design-focus-numeric',visible&&output.type!=='categorical');show('design-focus-category',visible&&output.type==='categorical');
  const previous=$('design-class').value;$('design-class').replaceChildren();(output?.classes||[]).forEach(c=>$('design-class').add(new Option(c,c)));if(output?.classes?.includes(previous))$('design-class').value=previous;
}
['design-target','design-focus'].forEach(id=>$(id).addEventListener('change',updateDesignFocus));
function clearDesignResult(){research.proposals=null;state.designId=null;show('design-results',false);$('design-message').textContent='';updateSteps();renderDesign();}
$('design-form').addEventListener('input',event=>{if(!event.target.closest('.design-preview'))clearDesignResult();});
$('design-form').addEventListener('change',event=>{if(!event.target.closest('.design-preview'))clearDesignResult();});
$('design-form').addEventListener('submit',async event=>{
  event.preventDefault();if(state.busy)return;
  try{
    const domain=[...document.querySelectorAll('.design-variable')].map(box=>{const schema=research.schema[Number(box.dataset.index)];const value=key=>box.querySelector('[data-bound='+key+']').value;if(schema.type==='categorical'){const labels=[...box.querySelectorAll('.category-weights label')].filter(l=>l.querySelector('[data-category]').checked);return{name:schema.name,categories:labels.map(l=>l.querySelector('[data-category]').value),probabilities:labels.map(l=>decimal(l.querySelector('[data-weight]').value,schema.name))};}return{name:schema.name,minimum:decimal(value('minimum'),schema.name),maximum:decimal(value('maximum'),schema.name),integer:box.querySelector('[data-bound=integer]').checked,distribution:value('distribution'),...(value('distribution')==='normal'?{mean:decimal(value('mean'),schema.name),std:decimal(value('std'),schema.name)}:{})};});
    const constraints=designConstraints();
    const active=$('design-mode').value==='active';const payload={domain,constraints,count:Number($('design-count').value),seed:Number($('design-seed').value)};
    if(active){payload.pool_size=Number($('design-pool').value);payload.acquisition_target=$('design-target').value||null;if($('design-focus').checked){const o=state.result.outputs?.find(p=>p.target===payload.acquisition_target)||state.result;if(o.type==='categorical')payload.target_class=$('design-class').value;else payload.target_range=[decimal($('design-ymin').value,'la sortie minimale'),decimal($('design-ymax').value,'la sortie maximale')];}}
    else Object.assign(payload,{dataset_id:state.dataset.id,features:selectedFeatures(),targets:selectedTargets(),column_types:selectedTypes(),exclude_known:$('design-exclude').checked,job_id:state.result?state.job:null});
    busy(true);error(null);$('design-message').textContent='Génération et vérification des contraintes…';
    research.proposals=await api(active?`/api/jobs/${state.job}/suggestions`:'/api/design',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});state.designId=research.proposals.id||null;renderDesign();
    $('design-message').textContent=research.proposals.warning||'';
  }catch(e){error(e.message);$('design-message').textContent='';}finally{busy(false);updateDesignFocus();}
});
function predictionText(p){if(!p)return '—';return p.type==='categorical'?p.label:number(p.mean,6);}
function renderDesign(){
  const data=research.proposals;show('design-results',Boolean(data));const outputs=data?.targets||[];
  if(data){
    const headers=['N°',...data.features,...(data.rows[0]?.predictions?outputs.map(n=>n+' prédit'):[])];if(data.rows[0]?.predictions)headers.push('Priorité','Distance');
    fillTable($('design-table'),headers,data.rows.map((r,j)=>[j+1,...r.values,...(r.predictions?outputs.map(n=>predictionText(r.predictions[n])).concat([r.score,r.distance]):[])]));
    $('design-diagnostics').textContent=data.method||'';
  }
  const x=$('design-x').value,y=$('design-y').value;
  const coordinate=(row,key)=>key.startsWith('feature:')?row.values[Number(key.slice(8))]:key.startsWith('output:')?(row.predictions?.[key.slice(7)]?.mean??row.predictions?.[key.slice(7)]?.label??null):0;
  const label=key=>key.startsWith('feature:')?research.schema[Number(key.slice(8))]?.name||'Entrée':key.startsWith('output:')?key.slice(7):'Projection 1D';
  const traces=[],result=state.result;
  if(result){
    const observed=(i,key)=>key.startsWith('feature:')?result.data.x[i][Number(key.slice(8))]:key.startsWith('output:')?(result.data.y_by_target?.[key.slice(7)]||result.data.y)[i]:0;
    [['train','Apprentissage','#93a4bb','circle'],['test','Test réservé','#54769d','diamond-open']].forEach(([part,name,color,symbol])=>{const ids=result.data[part];traces.push({x:ids.map(i=>observed(i,x)),y:ids.map(i=>observed(i,y)),mode:'markers',name,marker:{color,size:7,symbol,opacity:.65},hovertemplate:name+'<br>%{x}<br>%{y}<extra></extra>'});});
  }else if(state.dataset){
    const numericOrCategory=(i,key)=>{if(!key.startsWith('feature:'))return 0;const s=research.schema[Number(key.slice(8))],j=state.dataset.columns.findIndex(c=>c.name===s.name),value=state.dataset.records[i][j];if(value===null)return null;if(s.type==='categorical')return categoryText(value);try{return decimal(value,s.name);}catch{return null;}};
    const ids=state.dataset.records.map((_,i)=>i);traces.push({x:ids.map(i=>numericOrCategory(i,x)),y:ids.map(i=>numericOrCategory(i,y)),mode:'markers',name:'Données importées',marker:{color:'#93a4bb',size:7,opacity:.65}});
  }
  if(data)traces.push({x:data.rows.map(r=>coordinate(r,x)),y:data.rows.map(r=>coordinate(r,y)),mode:'markers',name:'Points proposés',marker:{color:'#cd654c',size:10,symbol:'diamond',line:{color:'#fff',width:1}},text:data.rows.map((_,j)=>'Point '+(j+1)),hovertemplate:'%{text}<br>%{x}<br>%{y}<extra></extra>'});
  const plotLayout={...layout(label(x),label(y)),height:380};
  const categorical=key=>key.startsWith('feature:')?research.schema[Number(key.slice(8))]?.type==='categorical':key.startsWith('output:')?result?.outputs?.find(o=>o.target===key.slice(7))?.type==='categorical':false;
  if(categorical(x))plotLayout.xaxis.type='category';if(categorical(y))plotLayout.yaxis.type='category';
  state.suggestionsReady=Plotly.react('design-chart',traces,plotLayout,localPlotConfig).catch(e=>error(e.message));updateSteps();
}
['design-x','design-y'].forEach(id=>$(id).addEventListener('change',renderDesign));
$('design-download').addEventListener('click',()=>{const data=research.proposals;if(!data)return;const quote=v=>'"'+String(v).replaceAll('"','""')+'"';const rows=[data.features.concat(data.targets||[]),...data.rows.map(r=>r.values.map(v=>typeof v==='number'?String(v).replace('.',','):v).concat((data.targets||[]).map(()=>'')))];const text='\uFEFF'+rows.map(r=>r.map(quote).join(';')).join('\r\n');const url=URL.createObjectURL(new Blob([text],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download=downloadName('plan-experiences.csv');a.click();setTimeout(()=>URL.revokeObjectURL(url),60000);});

// Batch prediction shares the same typed model and writes one column per output.
$('excel-choose').addEventListener('click',()=>{if(!state.busy)$('excel-file').click();});
$('excel-file').addEventListener('change',async event=>{
  const file=event.target.files[0];if(!file||state.busy)return;if(!/\.xlsx$/i.test(file.name)||file.size>20*1024*1024){error('Choisissez un fichier .xlsx de moins de 20 Mo.');return;}
  research.workbook=null;research.info=null;show('excel-settings',false);show('excel-mapping-panel',false);const form=new FormData();form.append('file',file);busy(true);error(null);$('excel-message').textContent='Lecture du classeur…';
  try{research.workbook=await api('/api/workbooks',{method:'POST',body:form});$('excel-filename').textContent=research.workbook.filename;$('excel-sheet').replaceChildren();research.workbook.sheets.forEach(s=>$('excel-sheet').add(new Option(s.name,s.name)));$('excel-header').value='1';show('excel-settings',true);await readExcelColumns();}catch(e){error(e.message);$('excel-message').textContent='';}finally{busy(false);$('excel-file').value='';}
});
async function readExcelColumns(){
  if(!research.workbook)return;show('excel-mapping-panel',false);
  const info=await api(`/api/workbooks/${research.workbook.id}/inspect`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sheet:$('excel-sheet').value,header_row:Number($('excel-header').value)})});research.info=info;$('excel-mapping').replaceChildren();$('excel-outputs').replaceChildren();
  const add=select=>info.columns.forEach(c=>select.add(new Option(`${c.letter} — ${c.name||'Sans titre'}`,String(c.index))));
  state.result.features.forEach(name=>{const label=document.createElement('label'),select=document.createElement('select');label.textContent='Entrée : '+name;select.dataset.feature=name;select.add(new Option('Choisir une colonne',''));add(select);const match=info.columns.find(c=>c.name.trim().toLowerCase()===name.trim().toLowerCase());if(match)select.value=String(match.index);label.append(select);$('excel-mapping').append(label);});
  (state.result.targets||[state.result.target]).forEach(name=>{const label=document.createElement('label'),select=document.createElement('select');label.textContent='Sortie : '+name;select.dataset.target=name;select.add(new Option('Ajouter « '+name+' prédit »','new'));add(select);label.append(select);$('excel-outputs').append(label);});
  $('excel-first').value=info.first_row;$('excel-last').value=info.last_row;show('excel-mapping-panel',true);$('excel-message').textContent='';
}
$('excel-inspect').addEventListener('click',async()=>{if(state.busy)return;busy(true);try{await readExcelColumns();}catch(e){error(e.message);}finally{busy(false);}});
['excel-sheet','excel-header'].forEach(id=>$(id).addEventListener('change',()=>{research.info=null;show('excel-mapping-panel',false);$('excel-message').textContent='Cliquez sur « Lire les colonnes » pour cette feuille.';}));
$('excel-predict').addEventListener('click',async()=>{
  if(state.busy||!research.info)return;
  try{
    const mapping=Object.fromEntries([...document.querySelectorAll('#excel-mapping select')].map(select=>{if(!select.value)throw new Error('Choisissez la colonne de '+select.dataset.feature+'.');return[select.dataset.feature,Number(select.value)];}));let next=research.info.columns.length+1;
    const outputs=Object.fromEntries([...document.querySelectorAll('#excel-outputs select')].map(select=>[select.dataset.target,{column:select.value==='new'?next++:Number(select.value),name:select.dataset.target+' prédit'}]));
    const first=Object.values(outputs)[0],payload={workbook_id:research.workbook.id,sheet:research.info.sheet,header_row:research.info.header_row,first_row:Number($('excel-first').value),last_row:Number($('excel-last').value),mapping,outputs,output_column:first.column,output_name:first.name,overwrite:$('excel-overwrite').checked};
    busy(true);error(null);$('excel-message').textContent='Calcul de toutes les sorties…';const headers=await downloadResponse(`/api/jobs/${state.job}/predict-excel`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)},research.workbook.filename.replace(/\.xlsx$/i,'')+'-predictions.xlsx');const s=JSON.parse(headers.get('X-GP-Summary'));$('excel-message').textContent=`${s.predicted} cellule(s) de prédiction écrite(s). ${s.skipped_existing} cellules existantes conservées. ${s.skipped_invalid} entrée(s) de ligne invalides ou catégories inconnues. Colonnes : ${s.output}.`;
  }catch(e){error(e.message);$('excel-message').textContent='';}finally{busy(false);}
});
