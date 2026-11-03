'use strict';
// Fixed LaTeX templates; imported column names never become executable TeX.
const kernelPalette={signal:'#7c3eac',noise:'#bd4b32',length0:'#167b78',length1:'#2374ad',alpha:'#96630e',linear:'#4e62a6',mix:'#a14c83'};
function renderLatex(tex,element,displayMode=false){
  katex.render(tex,element,{displayMode,throwOnError:true,strict:'ignore',trust:context=>context.command==='\\htmlClass'});
}
function kernelMath(result){
  const variant=result.kernel_variants?.[Number($('kernel-class').value)||0];const diagnostics=variant?.bound_diagnostics||result.bound_diagnostics||[],name=result.kernel;
  const lengthGroups=[...new Set(diagnostics.filter(d=>d.parameter.endsWith('length_scale')).map(d=>d.parameter))];if(!lengthGroups.length&&!['Linéaire','Quadratique'].includes(name)){lengthGroups.push('length-default');if(name==='RBF à deux échelles')lengthGroups.push('length-second');}
  const constants=[...new Set(diagnostics.filter(d=>d.parameter.endsWith('constant_value')).map(d=>d.parameter))];
  const term=(id,tex)=>String.raw`\htmlClass{kernel-term term-${id}}{\color{${kernelPalette[id]}}{${tex}}}`;
  const lengthSymbol=(g,index=null)=>String.raw`\ell${lengthGroups.length>1?`^{(${g+1})}`:''}${index===null?'':`_{${index}}`}`;
  const distance=g=>{
    const records=diagnostics.filter(d=>d.parameter===lengthGroups[g]);
    const ell=term('length'+g,lengthSymbol(g,records.length>1?'d':null));
    return records.length===1?String.raw`\frac{\lVert\mathbf z_i-\mathbf z_j\rVert^2}{${ell}^2}`:String.raw`\sum_{d=1}^{D}\left(\frac{z_{id}-z_{jd}}{${ell}}\right)^2`;
  };
  const r=g=>String.raw`r${name==='RBF à deux échelles'?`_{${g+1},ij}`:'_{ij}'}`;
  const matern=(g,nu)=>nu===3?String.raw`\left(1+\sqrt{3}${r(g)}\right)e^{-\sqrt{3}${r(g)}}`:String.raw`\left(1+\sqrt{5}${r(g)}+\frac{5${r(g)}^2}{3}\right)e^{-\sqrt{5}${r(g)}}`;
  const dot=String.raw`\left(${term('linear',String.raw`\sigma_0`)}^2+\mathbf z_i^{\mathsf T}\mathbf z_j\right)`;
  let signal;
  if(name==='Linéaire')signal=dot;
  else if(name==='Quadratique')signal=dot+'^2';
  else if(name==='Tendance linéaire + Matérn 3/2')signal=String.raw`\left[${dot}+${matern(0,3)}\right]`;
  else if(name==='RBF à deux échelles')signal=String.raw`\left[e^{-${r(0)}^2/2}+${term('mix',String.raw`\rho^2`)}e^{-${r(1)}^2/2}\right]`;
  else if(name.startsWith('Matérn 3/2'))signal=matern(0,3);
  else if(name.startsWith('Matérn 5/2'))signal=matern(0,5);
  else if(name.startsWith('RationalQuadratic')){const alpha=term('alpha',String.raw`\alpha`);signal=String.raw`\left(1+\frac{r_{ij}^2}{2${alpha}}\right)^{-${alpha}}`;}
  else signal=String.raw`e^{-r_{ij}^2/2}`;
  const noise=result.type!=='categorical'?String.raw`+${term('noise',String.raw`\sigma_n^2\delta_{ij}`)}`:'';
  const formula=String.raw`k_{ij}=${term('signal',String.raw`\sigma_f^2`)}\,${signal}${noise}`;
  const definitions=lengthGroups.map((_,g)=>String.raw`${r(g)}^2=${distance(g)}`);
  // Old classification archives contain no scalar diagnostics; still display the selected family.
  if(!lengthGroups.length&&!['Linéaire','Quadratique'].includes(name))definitions.push(String.raw`r_{ij}^2=\sum_{d=1}^{D}\left(\frac{z_{id}-z_{jd}}{${term('length0',String.raw`\ell_d`)}}\right)^2`);
  const occurrence={};
  const parameters=diagnostics.map(d=>{
    let group,symbol;
    if(d.parameter.endsWith('length_scale')){const g=lengthGroups.indexOf(d.parameter),index=(occurrence[d.parameter]||0)+1;occurrence[d.parameter]=index;group='length'+g;symbol=lengthSymbol(g,diagnostics.filter(v=>v.parameter===d.parameter).length>1?index:null);}
    else if(d.parameter.endsWith('noise_level')){group='noise';symbol=String.raw`\sigma_n^2`;}
    else if(d.parameter.endsWith('sigma_0')){group='linear';symbol=String.raw`\sigma_0`;}
    else if(d.parameter.endsWith('alpha')){group='alpha';symbol=String.raw`\alpha`;}
    else{group=constants.indexOf(d.parameter)>0?'mix':'signal';symbol=group==='mix'?String.raw`\rho^2`:String.raw`\sigma_f^2`;}
    const label=group==='mix'?'Variance relative de la seconde échelle':group.startsWith('length')&&lengthGroups.length>1?d.name+' · échelle '+(Number(group.slice(6))+1):d.name;return{...d,name:label,group,symbol};
  });
  return{formula,definitions,parameters};
}
function renderKernelParameters(){
  const result=currentResult(),math=kernelMath(result),host=$('selected-kernel-formula');host.replaceChildren();
  const heading=document.createElement('strong');heading.textContent='Noyau retenu · '+result.kernel;host.append(heading);
  [math.formula,...math.definitions].forEach(tex=>{const block=document.createElement('div');block.className='latex-equation';renderLatex(tex,block,true);host.append(block);});
  const caption=document.createElement('p');caption.className='note';caption.textContent='Les couleurs relient chaque paramètre à son terme dans la formule. z désigne les entrées encodées et standardisées.';host.append(caption);
  fillTable($('bounds-table'),['Paramètre','Symbole','Valeur','Minimum','Maximum','Position'],math.parameters.map(d=>[d.name,'',d.value,d.lower,d.upper,d.hit==='lower'?'Borne minimale':d.hit==='upper'?'Borne maximale':'Dans les bornes']));
  document.querySelectorAll('#bounds-table tbody tr').forEach((row,index)=>{
    const d=math.parameters[index];row.dataset.term=d.group;row.style.setProperty('--parameter-color',kernelPalette[d.group]);row.tabIndex=0;renderLatex(d.symbol,row.cells[1]);
    const emphasize=()=>{const active=row.matches(':hover, :focus');row.classList.toggle('parameter-active',active);host.querySelectorAll('.term-'+d.group).forEach(el=>el.classList.toggle('kernel-term-active',active));};
    row.addEventListener('mouseenter',()=>emphasize(true));row.addEventListener('mouseleave',()=>emphasize(false));row.addEventListener('focus',()=>emphasize(true));row.addEventListener('blur',()=>emphasize(false));
  });
  show('bounds-table',math.parameters.length>0);
  $('bounds-note').textContent=[...new Set(math.parameters.filter(d=>d.hit).map(d=>d.explanation))].join(' ');
}
