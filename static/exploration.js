'use strict';
// Scrollable data window: all rows are available without thousands of DOM rows.
let tableWindow = null;
function renderDataTable(dataset) {
  const scroller = document.getElementById('data-scroll');
  const table = document.getElementById('preview-table');
  table.replaceChildren();
  table.style.minWidth = `${dataset.columns.length * 130}px`;
  table.setAttribute('aria-rowcount', String(dataset.rows + 1));
  const head = table.createTHead(), header = head.insertRow();
  dataset.columns.forEach(column => {
    const th = document.createElement('th'); th.scope = 'col'; th.textContent = column.name; header.append(th);
  });
  const body = table.createTBody();
  tableWindow = {dataset, body, start:-1};
  scroller.scrollTop = 0;
  updateTableWindow();
}
function updateTableWindow() {
  if (!tableWindow) return;
  const {dataset, body} = tableWindow;
  const start = Math.min(Math.max(0, dataset.rows - 15), Math.max(0, Math.floor((document.getElementById('data-scroll').scrollTop - 40) / 40) - 5));
  if (start === tableWindow.start) return;
  tableWindow.start = start;
  const end = Math.min(dataset.rows, start + 35);
  const fragment = document.createDocumentFragment();
  function spacer(height) {
    if (!height) return;
    const tr = document.createElement('tr'), td = document.createElement('td');
    tr.setAttribute('aria-hidden', 'true'); td.colSpan = dataset.columns.length;
    td.className = 'table-spacer'; td.style.height = `${height}px`; tr.append(td); fragment.append(tr);
  }
  spacer(start * 40);
  dataset.records.slice(start, end).forEach((row, offset) => {
    const tr = document.createElement('tr'); tr.setAttribute('aria-rowindex', String(start + offset + 2));
    row.forEach(value => {
      const td = document.createElement('td');
      td.textContent = value === null ? '—' : typeof value === 'number' ? number(value, 8) : String(value);
      td.title = value === null ? '' : String(value); tr.append(td);
    }); fragment.append(tr);
  });
  spacer((dataset.rows - end) * 40);
  body.replaceChildren(fragment);
}
document.getElementById('data-scroll').addEventListener('scroll', updateTableWindow, {passive:true});

let inputTab = 'table';
function selectInputTab(tab) {
  inputTab = tab;
  document.querySelectorAll('[data-input-tab]').forEach(button => {
    const selected = button.dataset.inputTab === tab;
    button.setAttribute('aria-selected', String(selected)); button.tabIndex = selected ? 0 : -1;
  });
  document.getElementById('input-table-panel').hidden = tab !== 'table';
  document.getElementById('input-matrix-panel').hidden = tab !== 'matrix';
  if (tab === 'matrix') refreshInputMatrix();
}
function refreshInputMatrix() {
  if (inputTab === 'matrix' && state.dataset) renderInputMatrix('input-matrix', state.dataset, selectedFeatures());
}
document.querySelectorAll('[data-input-tab]').forEach((button, i, all) => {
  button.addEventListener('click', () => selectInputTab(button.dataset.inputTab));
  button.addEventListener('keydown', event => {
    if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? all.length - 1 : (i + 1) % all.length;
    all[next].focus({preventScroll:true}); selectInputTab(all[next].dataset.inputTab);
  });
});

const matrices = new Map();
const localPlotConfig = {responsive:true, displayModeBar:false, displaylogo:false, scrollZoom:false};
function renderInputMatrix(id, dataset, features) {
  const host = document.getElementById(id);
  const key = JSON.stringify([dataset.id, features]);
  const previous = matrices.get(id);
  if (previous?.key === key) return;
  if (previous) {
    previous.observer.disconnect(); previous.resize.disconnect();
    previous.plots.forEach(plot => { if (plot.rendered) Plotly.purge(plot.element); });
  }
  host.replaceChildren(); host.scrollTop = 0; host.scrollLeft = 0;
  const grid = document.createElement('div'); grid.className = 'matrix-grid';
  grid.style.gridTemplateColumns = `repeat(${Math.max(1, features.length)}, minmax(300px, 1fr))`;
  host.append(grid);
  if (!features.length) { grid.textContent = 'Sélectionnez au moins une variable d’entrée.'; matrices.delete(id); return; }
  const plots = [];
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      const plot = entry.target.matrixPlot;
      if (entry.isIntersecting && !plot.rendered) {
        plot.rendered = true;
        Plotly.newPlot(plot.element, plot.traces, plot.layout, localPlotConfig).catch(() => {
          plot.element.textContent = 'Ce graphique n’a pas pu être affiché.';
        });
      }
    });
  }, {root:host, rootMargin:'60px'});
  features.forEach((vertical, i) => features.forEach((horizontal, j) => {
    const cell = document.createElement('section'); cell.className = 'matrix-cell';
    const title = document.createElement('h4'); title.textContent = i === j ? vertical : `${vertical} en fonction de ${horizontal}`;
    const note = document.createElement('p'); note.className = 'matrix-stats';
    const element = document.createElement('div'); element.className = 'matrix-plot';
    element.setAttribute('role', 'img'); element.setAttribute('aria-label', title.textContent);
    cell.append(title, note, element); grid.append(cell);
    const valuesX = dataset.numeric_data[horizontal], valuesY = dataset.numeric_data[vertical];
    let traces;
    const config = {height:290, autosize:true, margin:{l:54,r:18,t:14,b:50}, showlegend:false,
      font:{family:'Inter, Segoe UI, sans-serif',size:13,color:'#677a96'}, paper_bgcolor:'#fff',plot_bgcolor:'#fff',
      xaxis:{title:{text:horizontal,standoff:5},gridcolor:'#edf1f6',zeroline:false},
      yaxis:{title:{text:i===j?'Effectif':vertical,standoff:5},gridcolor:'#edf1f6',zeroline:false}};
    if (i === j) {
      const values = valuesX.filter(Number.isFinite);
      const mean = values.length ? values.reduce((sum,x)=>sum+x/values.length,0) : NaN;
      const variance = values.length ? values.reduce((sum,x)=>sum+(x-mean)**2/values.length,0) : NaN;
      note.textContent = `Moyenne : ${Number.isFinite(mean)?number(mean,6):'—'} · Variance : ${Number.isFinite(variance)?number(variance,6):'—'} · N = ${values.length}`;
      traces = [{type:'histogram',x:values,marker:{color:'#9dbbe9'},nbinsx:16,hovertemplate:'%{x}<br>Effectif : %{y}<extra></extra>'}];
      if (Number.isFinite(mean)) config.shapes = [{type:'line',xref:'x',yref:'paper',x0:mean,x1:mean,y0:0,y1:1,line:{color:'#285fcd',width:2,dash:'dash'}}];
    } else {
      const x = [], y = [];
      valuesX.forEach((value,k) => { if(Number.isFinite(value) && Number.isFinite(valuesY[k])) {x.push(value);y.push(valuesY[k]);} });
      note.textContent = `${x.length} observations`;
      traces = [{type:'scatter',mode:'markers',x,y,marker:{size:4,color:'#316bd1',opacity:.55},hovertemplate:'x : %{x:.6g}<br>y : %{y:.6g}<extra></extra>'}];
    }
    const plot = {element,traces,layout:config,rendered:false}; cell.matrixPlot = plot; plots.push(plot); observer.observe(cell);
  }));
  const resize = new ResizeObserver(() => {
    if(host.clientWidth) plots.filter(p=>p.rendered && p.element.getBoundingClientRect().width).forEach(p=>Plotly.Plots.resize(p.element));
  });
  resize.observe(host); matrices.set(id,{key,observer,resize,plots});
}

let predictionVersion = 0;
function setupPrediction() {
  ++predictionVersion;
  const container = document.getElementById('prediction-inputs'); container.replaceChildren();
  document.getElementById('prediction-error').hidden = true;
  document.getElementById('prediction-output').hidden = true;
  document.getElementById('predict-button').disabled = false;
  state.result.input_ranges.forEach((range, index) => {
    const field = document.createElement('div'), label = document.createElement('label'), input = document.createElement('input'), hint = document.createElement('small');
    label.htmlFor = `new-point-${index}`; label.textContent = range.name;
    input.id = label.htmlFor; input.type = 'text'; input.inputMode = 'decimal'; input.required = true;
    input.value = String(range.median); input.dataset.feature = range.name;
    hint.id = `new-point-hint-${index}`; hint.textContent = `Apprentissage : ${number(range.minimum,6)} à ${number(range.maximum,6)}`;
    input.setAttribute('aria-describedby',hint.id);
    field.append(label,input,hint); container.append(field);
  });
}
document.getElementById('prediction-form').addEventListener('input', () => {
  ++predictionVersion;
  document.getElementById('prediction-output').hidden = true;
  document.getElementById('prediction-error').hidden = true;
});
document.getElementById('prediction-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (!state.result || state.busy) return;
  const result = state.result, job = state.job, version = ++predictionVersion;
  const button = document.getElementById('predict-button'), output = document.getElementById('prediction-output'), errorBox = document.getElementById('prediction-error');
  const values = Object.create(null);
  output.hidden = true; errorBox.hidden = true;
  try {
    document.querySelectorAll('#prediction-inputs input').forEach(input => {
      const raw = input.value.trim().replace(',', '.');
      if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(raw) || !Number.isFinite(Number(raw))) throw new Error(`Saisissez un nombre fini pour ${input.dataset.feature}.`);
      values[input.dataset.feature] = Number(raw);
    });
    button.disabled = true; button.textContent = 'Calcul en cours…';
    const prediction = await api(`/api/jobs/${job}/predict`, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({values})});
    if (state.result !== result || state.job !== job || predictionVersion !== version) return;
    output.replaceChildren();
    const mean = document.createElement('strong'); mean.textContent = `${prediction.target} prédit = ${number(prediction.mean,8)}`;
    const interval = document.createElement('p'); interval.textContent = `Intervalle prédictif nominal à 95 % : [${number(prediction.lower,8)} ; ${number(prediction.upper,8)}]`;
    output.append(mean,interval);
    if (prediction.outside.length) {
      const note = document.createElement('p'); note.className = 'extrapolation';
      note.textContent = `Extrapolation : ${prediction.outside.join(', ')} hors de la plage d’apprentissage. La prédiction peut être moins fiable.`; output.append(note);
    }
    output.hidden = false;
  } catch(e) {
    if(state.result === result && predictionVersion === version) { errorBox.textContent=e.message; errorBox.hidden=false; }
  } finally {
    button.disabled = false; button.textContent = 'Calculer la prédiction';
  }
});
