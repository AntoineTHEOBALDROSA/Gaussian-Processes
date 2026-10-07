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

async function plotImage(traces, sourceLayout, width=1300, height=720, large=true) {
  const host = document.createElement('div');
  host.style.cssText = `position:fixed;left:-20000px;top:0;width:${width}px;height:${height}px;pointer-events:none`;
  host.setAttribute('aria-hidden', 'true'); document.body.append(host);
  const plotLayout = structuredClone(sourceLayout);
  delete plotLayout.template;
  Object.assign(plotLayout, {width, height, autosize:false, paper_bgcolor:'#fff', plot_bgcolor:'#fff',
    font:{...plotLayout.font, family:'Arial, sans-serif', size:large ? 19 : 15},
    margin:large ? {l:Math.max(120,sourceLayout.margin?.l || 0),r:110,t:85,b:100} : {l:72,r:30,t:20,b:65}});
  if (plotLayout.legend) plotLayout.legend = {...plotLayout.legend, font:{size:large ? 17 : 13}, y:1.12};
  if (plotLayout.scene) {
    plotLayout.scene.domain = {x:[0,.9],y:[.1,.94]};
    for (const axis of ['xaxis','yaxis','zaxis']) {
      plotLayout.scene[axis] = {...plotLayout.scene[axis],tickfont:{size:15},
        title:{...plotLayout.scene[axis]?.title,font:{size:17}}};
    }
  }
  try {
    await Plotly.newPlot(host, structuredClone(traces), plotLayout, {displayModeBar:false, staticPlot:true});
    return await Plotly.toImage(host, {format:'png', width, height, scale:2});
  } finally { Plotly.purge(host); host.remove(); }
}
async function loadedImage(uri) {
  const image = new Image(); image.src = uri;
  await image.decode(); return image;
}
function canvasText(context, text, x, y, maxWidth, font) {
  context.font = font;
  while (context.measureText(text).width > maxWidth && text.length > 3) text = text.slice(0,-2);
  context.fillText(text, x, y);
}
async function matrixPages(id) {
  const matrix = matrices.get(id);
  if (!matrix?.plots.length) throw new Error('Sélectionnez des entrées avant d’exporter la matrice.');
  const plots = matrix.plots;
  const pages = [];
  // Four cells per landscape page, including cells not yet rendered on screen.
  for (let start=0; start<plots.length; start+=4) {
    transferStatus(`Préparation de la matrice : ${Math.min(start+4,plots.length)} / ${plots.length} graphiques…`);
    const sheet = document.createElement('canvas'); sheet.width=2800; sheet.height=1920;
    const context = sheet.getContext('2d'); context.fillStyle='#fff'; context.fillRect(0,0,sheet.width,sheet.height);
    for (let offset=0; offset<4 && start+offset<plots.length; offset++) {
      const plot = plots[start+offset], cell = plot.element.parentElement;
      const x = offset%2*1400, y = Math.floor(offset/2)*960;
      context.fillStyle='#172c49';
      canvasText(context, cell.querySelector('h4').textContent, x+36,y+44,1320,'bold 40px Arial');
      context.fillStyle='#63748c';
      canvasText(context, cell.querySelector('.matrix-stats').textContent,x+36,y+88,1320,'32px Arial');
      const uri = await plotImage(plot.traces, plot.layout, 700, 420, false);
      context.drawImage(await loadedImage(uri), x,y+100,1400,840);
    }
    pages.push({title:'Matrice des entrées', description:'Axe horizontal : variable de la colonne ; axe vertical : variable de la ligne. Diagonale : distribution, moyenne et variance (division par N).', image:sheet.toDataURL('image/png')});
  }
  return pages;
}
async function exportPdf(matrixId=null) {
  if (state.busy || (!matrixId && !state.result)) return;
  busy(true); error(null); transferStatus('Préparation du PDF…');
  try {
    let pages;
    if (matrixId || state.tab === 'matrix') {
      pages = await matrixPages(matrixId || 'result-matrix');
    } else {
      await state.chartReady;
      let traces, plotLayout;
      if (state.tab === 'kernels') {
        const ranking=state.result.ranking;
        traces=[{type:'bar',orientation:'h',x:ranking.map(k=>k.rmse),y:ranking.map(k=>k.name),
          error_x:{type:'data',array:ranking.map(k=>k.std),color:'#6788b9'},marker:{color:'#316bd1'}}];
        plotLayout=layout('RMSE de validation croisée','Noyau');
        plotLayout.yaxis.autorange='reversed';
        plotLayout.margin={l:290,r:60,t:30,b:70};
      } else {
        traces=$('chart').data; plotLayout=structuredClone($('chart').layout);
        if (state.tab === 'data' && plotLayout.scene) {
          plotLayout.scene.camera=structuredClone($('chart')._fullLayout.scene.camera);
        }
      }
      const image=await plotImage(traces,plotLayout);
      pages=[{title:$('chart-title').textContent, description:$('chart-description').textContent, image}];
    }
    transferStatus('Création du fichier PDF…');
    await downloadResponse('/api/export/pdf', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pages})}, downloadName(`${matrixId?'matrice':state.tab}.pdf`));
    transferStatus('PDF téléchargé.');
  } catch (e) { error(e.message); transferStatus(null); }
  finally { busy(false); }
}
document.getElementById('export-pdf').addEventListener('click', () => exportPdf());
document.getElementById('input-matrix-pdf').addEventListener('click', () => exportPdf('input-matrix'));
