"""Rapport autonome de toute l'étude, avec sommaire et graphiques vectoriels."""
from design import constraint_description
import io
import math
from itertools import combinations
from xml.sax.saxutils import escape
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, PageBreak, LongTable, TableStyle, Flowable, KeepTogether
from reportlab.platypus.tableofcontents import TableOfContents
from pdf_export import fonts  # Registers the bundled Vera fonts.

BLUE='#316BD1'; INK='#243C5C'; MUTED='#657B97'
PALETTE=['#316BD1','#39897C','#C88738','#975C9B','#C65E54','#568AAB']


def text(value):
    if value is None or (isinstance(value,float) and not np.isfinite(value)): return '-'
    if isinstance(value,(float,np.floating)): return format(float(value),'.7g')
    return str(value)


class Plot(Flowable):
    def __init__(self, traces, xlabel, ylabel, width=730, height=265, heatmap=None):
        super().__init__();self.width=width;self.height=height
        self.traces=traces;self.xlabel=xlabel;self.ylabel=ylabel;self.heatmap=heatmap

    def draw(self):
        c=self.canv;left,bottom=65,48;w,h=self.width-left-25,self.height-bottom-35
        c.setFont('GPVera',8);c.setFillColor(colors.HexColor(MUTED))
        if self.heatmap is not None:
            labels,z=self.heatmap;z=np.asarray(z);size=max(1,len(labels));maximum=max(1,float(z.max()))
            for i in range(size):
                for j in range(size):
                    shade=float(z[i,j])/maximum
                    c.setFillColor(colors.Color(.96-.77*shade,.97-.56*shade,.99-.20*shade))
                    c.rect(left+j*w/size,bottom+(size-1-i)*h/size,w/size,h/size,stroke=0,fill=1)
                    if size<=12:
                        c.setFillColor(colors.white if shade>.55 else colors.HexColor(INK));c.drawCentredString(left+(j+.5)*w/size,bottom+(size-i-.5)*h/size-3,str(z[i,j]))
            xticks=[(i+.5,text(v)) for i,v in enumerate(labels)];yticks=xticks
            xf=lambda v:left+v*w/size;yf=lambda v:bottom+h-v*h/size
        else:
            xv=[v for t in self.traces for v in t['x'] if v is not None]
            yv=[v for t in self.traces for v in t['y'] if v is not None]
            for t in self.traces:
                for key in ('lower','upper'):
                    if key in t: yv.extend(t[key])
            def axis(values):
                numeric=bool(values) and all(isinstance(v,(int,float,np.number)) and not isinstance(v,bool) for v in values)
                if not numeric:
                    labels=list(dict.fromkeys(map(str,values))) or [''];mapping={v:i for i,v in enumerate(labels)}
                    return -.5,len(labels)-.5,lambda v:mapping[str(v)],[(v,v) for i,v in enumerate(labels) if i%max(1,math.ceil(len(labels)/7))==0]
                lo,hi=min(values),max(values);pad=(hi-lo)*.07 or .5;lo-=pad;hi+=pad
                return lo,hi,float,[(v,format(v,'.4g')) for v in np.linspace(lo,hi,5)]
            xmin,xmax,xvalue,xticks=axis(xv);ymin,ymax,yvalue,yticks=axis(yv)
            if any(t.get('mode')=='bar' for t in self.traces) and yv and min(yv)>=0:
                ymin=0;ymax=max(1,math.ceil(max(yv)*1.1))
                yticks=[(v,str(int(v))) for v in np.unique(np.rint(np.linspace(0,ymax,5)))]
            xf=lambda v:left+(xvalue(v)-xmin)/(xmax-xmin)*w
            yf=lambda v:bottom+(yvalue(v)-ymin)/(ymax-ymin)*h
            for value,label in yticks:
                y=yf(value);c.setStrokeColor(colors.HexColor('#E7EDF5'));c.setLineWidth(.4);c.line(left,y,left+w,y)
            for i,t in enumerate(self.traces):
                color=colors.HexColor(t.get('color',PALETTE[i%len(PALETTE)]));c.setStrokeColor(color);c.setFillColor(color);c.setLineWidth(1.5)
                pairs=[(x,y) for x,y in zip(t['x'],t['y']) if x is not None and y is not None]
                if not pairs:continue
                if 'lower' in t:
                    c.setFillColor(colors.HexColor('#E5EDFC'));path=c.beginPath();path.moveTo(xf(t['x'][0]),yf(t['lower'][0]))
                    for x,y in zip(t['x'][1:],t['lower'][1:]):path.lineTo(xf(x),yf(y))
                    for x,y in zip(reversed(t['x']),reversed(t['upper'])):path.lineTo(xf(x),yf(y))
                    path.close();c.drawPath(path,stroke=0,fill=1);c.setFillColor(color)
                if t.get('mode')=='bar':
                    bw=max(1,w/max(1,len(pairs))*.65)
                    for x,y in pairs:
                        base=yf(max(ymin,0));yy=yf(y);c.rect(xf(x)-bw/2,min(base,yy),bw,abs(yy-base),stroke=0,fill=1)
                elif t.get('mode','line')=='scatter':
                    for x,y in pairs:c.circle(xf(x),yf(y),1.6,stroke=0,fill=1)
                else:
                    path=c.beginPath();path.moveTo(xf(pairs[0][0]),yf(pairs[0][1]))
                    for x,y in pairs[1:]:path.lineTo(xf(x),yf(y))
                    c.drawPath(path,stroke=1,fill=0)
        c.setStrokeColor(colors.HexColor('#C8D4E4'));c.setLineWidth(.6);c.line(left,bottom,left+w,bottom);c.line(left,bottom,left,bottom+h)
        c.setFont('GPVera',7);c.setFillColor(colors.HexColor(MUTED))
        for value,label in xticks:c.drawCentredString(xf(value),bottom-14,str(label)[:24])
        for value,label in yticks:c.drawRightString(left-8,yf(value)-2,str(label)[:24])
        c.setFont('GPVera',8);c.drawCentredString(left+w/2,8,self.xlabel[:100])
        c.saveState();c.translate(11,bottom+h/2);c.rotate(90);c.drawCentredString(0,0,self.ylabel[:90]);c.restoreState()


class ReportDocument(BaseDocTemplate):
    def afterFlowable(self, flowable):
        if hasattr(flowable,'toc_key'):
            self.canv.bookmarkPage(flowable.toc_key)
            self.canv.addOutlineEntry(flowable.getPlainText(),flowable.toc_key,flowable.toc_level,closed=False)
            self.notify('TOCEntry',(flowable.toc_level,flowable.getPlainText(),self.page,flowable.toc_key))


def make_report(job):
    output=io.BytesIO();width,height=landscape(A4);body_width=width-84
    doc=ReportDocument(output,pagesize=(width,height),leftMargin=42,rightMargin=42,topMargin=37,bottomMargin=38,
                       title='Processus-Gaussiens - Rapport complet',author='Processus-Gaussiens')
    def footer(c,document):
        c.setFont('GPVera',8);c.setFillColor(colors.HexColor(MUTED));c.drawString(42,20,'Processus-Gaussiens - Rapport complet');c.drawRightString(width-42,20,str(document.page))
    doc.addPageTemplates(PageTemplate(id='main',frames=[Frame(42,38,body_width,height-75,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=footer))
    styles={
        'body':ParagraphStyle('body',fontName='GPVera',fontSize=9,leading=14,textColor=colors.HexColor(MUTED),spaceAfter=9),
        'h1':ParagraphStyle('h1',fontName='GPVeraBold',fontSize=19,leading=25,textColor=colors.HexColor(INK),spaceAfter=16,keepWithNext=True),
        'h2':ParagraphStyle('h2',fontName='GPVeraBold',fontSize=12,leading=17,textColor=colors.HexColor(INK),spaceBefore=14,spaceAfter=10,keepWithNext=True),
        'cell':ParagraphStyle('cell',fontName='GPVera',fontSize=8,leading=11,textColor=colors.HexColor(INK)),
        'headcell':ParagraphStyle('headcell',fontName='GPVeraBold',fontSize=8,leading=11,textColor=colors.white),
    }
    story=[];serial=0
    def paragraph(value,style='body'):return Paragraph(escape(text(value)),styles[style])
    def heading(value,level=0):
        nonlocal serial
        serial+=1;p=paragraph(value,'h1' if level==0 else 'h2');p.toc_key='section-'+str(serial);p.toc_level=level;story.append(p)
    def table(headers,rows,colwidths=None):
        data=[[paragraph(v,'headcell') for v in headers]]+[[paragraph(v,'cell') for v in row] for row in rows]
        t=LongTable(data,colWidths=colwidths or [body_width/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#304F78')),('VALIGN',(0,0),(-1,-1),'TOP'),
                              ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),
                              ('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.4,colors.white),
                              ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F6F8FC')])]))
        story.extend([t,Spacer(1,12)])
    def wide_table(headers,rows,chunk=6):
        for first in range(0,len(headers),chunk):
            columns=headers[first:first+chunk];part=[row[first:first+chunk] for row in rows]
            table(['Ligne']+columns,[[i+1]+r for i,r in enumerate(part)],[42]+[(body_width-42)/len(columns)]*len(columns))
    dataset=job['dataset'];result=job.get('result');plan=job.get('design')
    toc_heading=ParagraphStyle('toc-heading',fontName='GPVeraBold',fontSize=16,leading=20,textColor=colors.HexColor(INK),spaceAfter=6,keepWithNext=True)
    story.append(Paragraph('Sommaire du rapport complet',toc_heading))
    toc=TableOfContents();toc.levelStyles=[ParagraphStyle('toc0',fontName='GPVeraBold',fontSize=9.5,leading=13,textColor=colors.HexColor(INK),spaceBefore=5),
                                         ParagraphStyle('toc1',fontName='GPVera',fontSize=9,leading=12,leftIndent=16,textColor=colors.HexColor(MUTED))]
    story.extend([toc,PageBreak()]);heading('1. Données et configuration')
    options=job['options'];features=options['features'];targets=options.get('targets',[options.get('target')])
    story.append(paragraph('Fichier : '+dataset['filename']))
    story.append(paragraph(f"{dataset['rows']} lignes importées. {len(features)} entrées : {', '.join(features)}. Sorties : {', '.join(n for n in targets if n)}."))
    if result:
        story.append(paragraph(f"{result['train_count']} observations d'apprentissage, {result['test_count']} de test, {result['dropped']} lignes exclues. {result['folds']} plis. Graine de partage : {result.get('split_seed',result['seed'])}."))
        story.append(paragraph('Un modèle indépendant est appris par sortie. Les configurations identiques restent ensemble. Le test final ne participe ni au choix des noyaux ni à l’apprentissage. Les intervalles de régression sont nominaux ; les probabilités de classification sont des estimations du modèle.'))
    schema=result.get('input_ranges') if result else plan['domain']
    table(['Entrée','Type','Domaine / modalités'],[[r['name'],'Catégorie' if r.get('type')=='categorical' else 'Numérique',', '.join(r.get('categories',[])) if r.get('type')=='categorical' else str(r['minimum'])+' à '+str(r['maximum'])] for r in schema])
    heading('Distribution des entrées',1)
    raw=np.array(result['data']['x'],dtype=object) if result else np.array([[record[[c['name'] for c in dataset['columns']].index(f)] for f in features] for record in dataset['records']],dtype=object)
    for j,name in enumerate(features):
        values=[v for v in raw[:,j] if v is not None]
        numeric=schema[j].get('type','numeric')=='numeric'
        if numeric:
            values=pd.to_numeric(pd.Series(values),errors='coerce').dropna().to_numpy()
            counts,edges=np.histogram(values,bins=12);x=((edges[:-1]+edges[1:])/2).tolist();y=counts.tolist()
        else:
            series=pd.Series(values,dtype=str).value_counts();x=series.index.tolist();y=series.tolist()
        story.append(paragraph(name,'h2'))
        if numeric and len(values):
            story.append(paragraph(f'Moyenne : {np.mean(values):.6g}. Variance descriptive (division par N) : {np.var(values):.6g}. N = {len(values)}.'))
        story.append(Plot([dict(x=x,y=y,mode='bar')],name,'Effectif',width=body_width,height=220))
    if len(features)>1:
        heading('Paires d’entrées',1)
        for i,j in combinations(range(len(features)),2):
            xs=[];ys=[]
            for row in raw:
                if row[i] is None or row[j] is None:continue
                try:
                    xv=float(row[i]) if schema[i].get('type','numeric')=='numeric' else str(row[i])
                    yv=float(row[j]) if schema[j].get('type','numeric')=='numeric' else str(row[j])
                    if isinstance(xv,float) and not np.isfinite(xv) or isinstance(yv,float) and not np.isfinite(yv):continue
                    xs.append(xv);ys.append(yv)
                except (ValueError,TypeError):continue
            story.extend([paragraph(features[j]+' en fonction de '+features[i],'h2'),Plot([dict(x=xs,y=ys,mode='scatter')],features[i],features[j],width=body_width,height=220)])
    if result:
        for number,o in enumerate(result.get('outputs',[dict(result,type='numeric')]),2):
            story.append(PageBreak());heading(f"{number}. Sortie : {o['target']}")
            story.append(paragraph(f"{'Classification' if o.get('type')=='categorical' else 'Régression'} - noyau {o['kernel']} - calcul d’entraînement : {o['device']}."))
            labels={'rmse':'RMSE','mae':'MAE','coverage':'Couverture à 95 % (%)','accuracy':'Exactitude (%)','f1':'F1 macro','log_loss':'Log-loss'}
            table(['Indicateur de test','Valeur'],[[labels.get(k,k),v] for k,v in o['metrics'].items()])
            heading('Prédictions sur le test',1)
            parity=o['parity']
            if o.get('type')=='categorical':
                story.append(Plot([], 'Classe prédite','Classe réelle',width=body_width,heatmap=(o['classes'],o['confusion'])))
                rows=[[a,p]+prob for a,p,prob in zip(parity['actual'],parity['predicted'],parity['probabilities'])]
                wide_table(['Classe réelle','Classe prédite']+['P('+c+')' for c in o['classes']],rows)
            else:
                lo=min(parity['actual']+parity['predicted']);hi=max(parity['actual']+parity['predicted'])
                story.append(Plot([dict(x=[lo,hi],y=[lo,hi],color='#A3B1C4'),dict(x=parity['actual'],y=parity['predicted'],mode='scatter')],'Valeur réelle','Valeur prédite',width=body_width))
                table(['Réel','Prédit','Sigma','IC95 bas','IC95 haut'],[[a,m,s,m-1.96*s,m+1.96*s] for a,m,s in zip(parity['actual'],parity['predicted'],parity['std'])])
            heading('Distribution des sorties observées',1)
            values=result['data'].get('y_by_target',{}).get(o['target'],result['data']['y'])
            if o.get('type')=='categorical':
                observed=pd.Series(values,dtype=str).value_counts();xx=observed.index.tolist();yy=observed.tolist()
            else:
                counts,edges=np.histogram(values,bins=12);xx=((edges[:-1]+edges[1:])/2).tolist();yy=counts.tolist()
            story.append(Plot([dict(x=xx,y=yy,mode='bar')],o['target'],'Effectif',width=body_width,height=220))
            heading('Coupes 1D',1)
            story.append(paragraph('Les autres entrées sont fixées à leur médiane ou leur modalité la plus fréquente dans l’apprentissage. Les observations projetées ne correspondent pas nécessairement à ces valeurs fixes.'))
            for slice in o['slices']:
                story.append(paragraph(slice['name'],'h2'))
                if o.get('type')=='categorical':
                    traces=[dict(x=slice['x'],y=[p[j] for p in slice['probabilities']],color=PALETTE[j%len(PALETTE)]) for j in range(len(o['classes']))]
                    story.append(paragraph('Courbes (dans cet ordre) : '+', '.join(o['classes'])))
                else:
                    traces=[dict(x=slice['train_x'],y=slice['train_y'],mode='scatter',color='#A3B1C4'),dict(x=slice['x'],y=slice['mean'],lower=slice['lower'],upper=slice['upper'],color=BLUE)]
                story.append(Plot(traces,slice['name'],'Probabilité' if o.get('type')=='categorical' else o['target'],width=body_width))
            heading('Comparaison des noyaux et paramètres',1)
            table(['Noyau',o.get('score_name','RMSE')+' CV','Écart-type','Durée (s)','Alertes'],[[r['name'],r.get('score',r.get('rmse')),r['std'],r['seconds'],r['warnings']] for r in o['ranking']])
            story.append(paragraph(o['optimized_kernel']))
            if o.get('bound_diagnostics'):
                table(['Paramètre','Valeur','Minimum','Maximum','Position'],[[r['name'],r['value'],r['lower'],r['upper'],r['hit'] or 'Dans les bornes'] for r in o['bound_diagnostics']])
    if plan:
        story.append(PageBreak());heading('Plan d’expériences')
        story.append(paragraph(plan.get('method','Plan LHS avec lois marginales et contraintes.')))
        story.append(paragraph(f"{len(plan['rows'])} points. Graine : {plan.get('seed',2025)}. {plan.get('attempted','-')} tirages, {plan.get('rejected_constraints',0)} rejets par les contraintes."))
        story.append(paragraph('Après rejet, discrétisation ou sélection selon le modèle, les strates LHS et les proportions marginales peuvent être modifiées. Les sorties estimées ne remplacent pas de nouvelles mesures.'))
        for rule in plan.get('constraints',[]):
            story.append(paragraph(constraint_description(rule)))
        for domain in plan.get('domain',[]):
            story.append(paragraph(domain['name']+' : '+(', '.join(domain['categories'])+' ; poids '+str(domain['probabilities']) if domain['type']=='categorical' else domain['distribution']+' ; bornes '+str(domain['minimum'])+' à '+str(domain['maximum'])+(' ; moyenne '+str(domain['mean'])+' ; écart-type '+str(domain['std']) if domain['distribution']=='normal' else '')+(' ; valeurs entières' if domain['integer'] else ''))))
        wide_table(plan['features'],[row['values'] for row in plan['rows']])
        first=plan['rows'][0] if plan['rows'] else None
        if first:
            xx=[row['values'][0] for row in plan['rows']]
            yy=[row['values'][1] if len(plan['features'])>1 else 0 for row in plan['rows']]
            story.append(Plot([dict(x=xx,y=yy,mode='scatter',color='#C65E54')],plan['features'][0],plan['features'][1] if len(plan['features'])>1 else 'Projection 1D',width=body_width,height=230))
        predicted=plan['rows'][0].get('predictions') if plan['rows'] else None
        if predicted:
            heading('Sorties estimées des points proposés',1)
            for target in predicted:
                rows=[]
                for row in plan['rows']:
                    p=row['predictions'][target]
                    rows.append([p.get('mean',p.get('label')),p.get('std'),p.get('latent_std'),', '.join(f'{c}: {v:.5g}' for c,v in p.get('probabilities',{}).items())])
                story.append(paragraph(target,'h2'));table(['Prédiction','Sigma prédictif','Sigma de fonction','Probabilités'],rows)
    if result and result.get('warnings'):
        story.append(PageBreak());heading('Diagnostics de l’entraînement')
        for message in result['warnings']:story.append(paragraph(message))
    story.append(PageBreak());heading('Annexe : toutes les données importées')
    story.append(paragraph('Données originales, y compris les colonnes non utilisées et les lignes éventuellement exclues. Les tableaux larges sont répartis en groupes de colonnes ; le numéro de ligne permet de les rapprocher.'))
    wide_table([c['name'] for c in dataset['columns']],dataset['records'])
    doc.multiBuild(story)
    return output.getvalue()
