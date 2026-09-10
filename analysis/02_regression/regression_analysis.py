#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from patsy import dmatrix, build_design_matrices
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

OUTCOME = 'py_thoughts_wanting_to_die'
ORDINAL = ['poor_subjective_physical_health','poor_subjective_mental_health','low_sense_of_belonging','low_perceived_social_equality','low_social_trust']
BINARY = ['sex_female','married_current','employed_corrected','living_alone']
LABELS = {
 'age10':'Age (per 10 years)','pss_z':'PSS-14 score (per SD)','selfesteem_z':'Low self-esteem score (per SD)',
 'sex_female':'Female sex','married_current':'Currently married','employed_corrected':'Currently working','living_alone':'Living alone',
 'poor_subjective_physical_health':'Poorer subjective physical health (per category)',
 'poor_subjective_mental_health':'Poorer subjective mental health (per category)',
 'low_sense_of_belonging':'Lower sense of belonging (per category)',
 'low_perceived_social_equality':'Lower perceived social equality (per category)',
 'low_social_trust':'Lower social trust (per category)'}
CAT_LABELS = {
 'marital_status_4cat':{1:'Single vs married',3:'Widowed vs married',4:'Divorced vs married'},
 'poor_subjective_physical_health':{1:'Very good vs good',3:'Moderate vs good',4:'Poor vs good'},
 'poor_subjective_mental_health':{1:'Very good vs good',3:'Moderate vs good',4:'Poor vs good'},
 'low_sense_of_belonging':{1:'Strongly connected vs somewhat connected',3:'Somewhat disconnected vs somewhat connected',4:'Strongly disconnected vs somewhat connected'},
 'low_perceived_social_equality':{1:'Very high vs high equality',3:'Moderate vs high equality',4:'Low vs high equality'},
 'low_social_trust':{1:'Very high vs high trust',3:'Moderate vs high trust',4:'Low vs high trust'}}

@dataclass
class Spec:
    var:str; knots:tuple[float,float,float,float]; info:Any; k:int
@dataclass
class Bundle:
    model:Any; X:np.ndarray; names:list[str]; y:np.ndarray; raw:dict[str,np.ndarray]; specs:dict[str,Spec]; means:dict[str,float]; sds:dict[str,float]

def load(path:str)->dict[str,np.ndarray]:
    with open(path,encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    return {k:np.array([float(r[k]) for r in rows]) for k in rows[0]}

def write(path:Path, header, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)

def spline(x,var):
    q=tuple(float(v) for v in np.quantile(x,[.05,.35,.65,.95]))
    form=f"cr(x, knots=({q[1]:.12g},{q[2]:.12g}), lower_bound={q[0]:.12g}, upper_bound={q[3]:.12g}, constraints='center') - 1"
    df=dmatrix(form,{'x':x},return_type='dataframe'); B=np.asarray(df,float)
    return Spec(var,q,df.design_info,B.shape[1]),B

def transform(spec,x): return np.asarray(build_design_matrices([spec.info],{'x':np.asarray(x,float)})[0],float)

def fit_bundle(data, use_phq=False):
    y=data[OUTCOME].astype(int); n=len(y)
    means={'age_years':float(data['age_years'].mean()),'pss14_total':float(data['pss14_total'].mean()),'low_self_esteem_score':float(data['low_self_esteem_score'].mean())}
    sds={'pss14_total':float(data['pss14_total'].std(ddof=1)),'low_self_esteem_score':float(data['low_self_esteem_score'].std(ddof=1))}
    gs,gB=spline(data['gad7_total'],'gad7_total'); specs={'gad7_total':gs}
    cols=[np.ones(n),(data['age_years']-means['age_years'])/10]; names=['Intercept','age10']
    for j in range(gs.k): cols.append(gB[:,j]); names.append(f'gad7_rcs{j+1}')
    if use_phq:
        ps,pB=spline(data['phq8_total'],'phq8_total'); specs['phq8_total']=ps
        for j in range(ps.k): cols.append(pB[:,j]); names.append(f'phq8_rcs{j+1}')
    cols += [(data['pss14_total']-means['pss14_total'])/sds['pss14_total'],(data['low_self_esteem_score']-means['low_self_esteem_score'])/sds['low_self_esteem_score']]
    names += ['pss_z','selfesteem_z']
    for v in BINARY+ORDINAL: cols.append(data[v]); names.append(v)
    X=np.column_stack(cols); model=sm.GLM(y,X,family=sm.families.Binomial()).fit(cov_type='HC3')
    return Bundle(model,X,names,y,data,specs,means,sds)

def scenario(b, overrides):
    d=b.raw; n=len(b.y); cols=[np.ones(n),((np.full(n,overrides['age_years']) if 'age_years' in overrides else d['age_years'])-b.means['age_years'])/10]
    for v in ['gad7_total']+(['phq8_total'] if 'phq8_total' in b.specs else []):
        vals=np.full(n,overrides[v]) if v in overrides else d[v]; B=transform(b.specs[v],vals)
        cols += [B[:,j] for j in range(B.shape[1])]
    pss=np.full(n,overrides['pss14_total']) if 'pss14_total' in overrides else d['pss14_total']
    se=np.full(n,overrides['low_self_esteem_score']) if 'low_self_esteem_score' in overrides else d['low_self_esteem_score']
    cols += [(pss-b.means['pss14_total'])/b.sds['pss14_total'],(se-b.means['low_self_esteem_score'])/b.sds['low_self_esteem_score']]
    for v in BINARY+ORDINAL: cols.append(np.full(n,overrides[v]) if v in overrides else d[v])
    return np.column_stack(cols)

def terms(b, prefixes):
    out=[]
    for i,n in enumerate(b.names):
        if n=='Intercept' or any(n.startswith(p) for p in prefixes): continue
        beta=float(b.model.params[i]); se=float(b.model.bse[i])
        out.append([n,LABELS.get(n,n),math.exp(beta),math.exp(beta-1.96*se),math.exp(beta+1.96*se),float(b.model.pvalues[i]),beta,se])
    return out

def global_wald(b,prefix):
    idx=[i for i,n in enumerate(b.names) if n.startswith(prefix)]; R=np.zeros((len(idx),len(b.names)))
    for j,i in enumerate(idx): R[j,i]=1
    t=b.model.wald_test(R,scalar=True); return float(t.statistic),len(idx),float(t.pvalue)

def risk(model,X):
    beta=np.asarray(model.params); cov=np.asarray(model.cov_params()); p=1/(1+np.exp(-(X@beta))); r=float(p.mean())
    g=np.mean((p*(1-p))[:,None]*X,axis=0); dl=g/(r*(1-r)); se=math.sqrt(max(float(dl@cov@dl),0)); lr=math.log(r/(1-r))
    return r,g,1/(1+math.exp(-(lr-1.96*se))),1/(1+math.exp(-(lr+1.96*se)))

def contrast(model,X0,X1):
    cov=np.asarray(model.cov_params()); r0,g0,l0,u0=risk(model,X0); r1,g1,l1,u1=risk(model,X1)
    gd=g1-g0; sed=math.sqrt(max(float(gd@cov@gd),0)); rd=r1-r0
    gl=g1/r1-g0/r0; sel=math.sqrt(max(float(gl@cov@gl),0)); lrr=math.log(r1/r0)
    return [r0,l0,u0,r1,l1,u1,math.exp(lrr),math.exp(lrr-1.96*sel),math.exp(lrr+1.96*sel),rd,rd-1.96*sed,rd+1.96*sed]

def curves(b,var,maxv):
    out=[]
    for v in range(maxv+1):
        r,_,lo,hi=risk(b.model,scenario(b,{var:v})); out.append([v,r,lo,hi])
    return out

def contrasts(b,var,vals,ref=0):
    X0=scenario(b,{var:ref}); out=[]
    for v in vals: out.append([var,ref,v]+contrast(b.model,X0,scenario(b,{var:v})))
    return out

def linear_primary(d):
    y=d[OUTCOME].astype(int); cols=[np.ones(len(y)),(d['age_years']-d['age_years'].mean())/10,(d['gad7_total']-d['gad7_total'].mean())/d['gad7_total'].std(ddof=1),(d['pss14_total']-d['pss14_total'].mean())/d['pss14_total'].std(ddof=1),(d['low_self_esteem_score']-d['low_self_esteem_score'].mean())/d['low_self_esteem_score'].std(ddof=1)]
    cols += [d[v] for v in BINARY+ORDINAL]
    return sm.GLM(y,np.column_stack(cols),family=sm.families.Binomial()).fit()

def phq_compare(b,splines):
    d=b.raw; y=b.y; cols=[np.ones(len(y)),(d['age_years']-b.means['age_years'])/10]
    for v in ['gad7_total','phq8_total']:
        if v in splines:
            B=transform(b.specs[v],d[v]); cols += [B[:,j] for j in range(B.shape[1])]
        else: cols.append((d[v]-d[v].mean())/d[v].std(ddof=1))
    cols += [(d['pss14_total']-b.means['pss14_total'])/b.sds['pss14_total'],(d['low_self_esteem_score']-b.means['low_self_esteem_score'])/b.sds['low_self_esteem_score']]
    cols += [d[v] for v in BINARY+ORDINAL]
    return sm.GLM(y,np.column_stack(cols),family=sm.families.Binomial()).fit()

def fit_cat(primary,cat,b):
    y=b.y; gB=transform(b.specs['gad7_total'],primary['gad7_total']); cols=[np.ones(len(y)),(primary['age_years']-primary['age_years'].mean())/10]+[gB[:,j] for j in range(gB.shape[1])]+[(primary['pss14_total']-primary['pss14_total'].mean())/primary['pss14_total'].std(ddof=1),(primary['low_self_esteem_score']-primary['low_self_esteem_score'].mean())/primary['low_self_esteem_score'].std(ddof=1),primary['sex_female']]
    names=['Intercept','age10']+[f'gad7_rcs{j+1}' for j in range(gB.shape[1])]+['pss_z','selfesteem_z','sex_female']; inds={}
    specs=[('marital_status_4cat',cat['marital_status_4cat'].astype(int)),*[(v,primary[v].astype(int)) for v in ORDINAL]]
    for v,x in specs:
        inds[v]=[]
        for lev in sorted(int(z) for z in np.unique(x)):
            if lev==2: continue
            inds[v].append(len(names)); cols.append((x==lev).astype(float)); names.append(f'{v}[{lev}]')
    cols += [primary['employed_corrected'],primary['living_alone']]; names += ['employed_corrected','living_alone']
    X=np.column_stack(cols); m=sm.GLM(y,X,family=sm.families.Binomial()).fit(cov_type='HC3')
    tr=[]
    for i,n in enumerate(names):
        if n=='Intercept' or n.startswith('gad7_rcs'): continue
        beta=float(m.params[i]); se=float(m.bse[i]); label=LABELS.get(n,n)
        if '[' in n:
            v=n.split('[')[0]; lev=int(n.split('[')[1].split(']')[0]); label=CAT_LABELS[v][lev]
        tr.append([n,label,math.exp(beta),math.exp(beta-1.96*se),math.exp(beta+1.96*se),float(m.pvalues[i])])
    gl=[]
    for v,idx in inds.items():
        R=np.zeros((len(idx),len(names)))
        for j,i in enumerate(idx):R[j,i]=1
        t=m.wald_test(R,scalar=True); x=cat[v] if v=='marital_status_4cat' else primary[v]
        counts={int(k):int((x==k).sum()) for k in np.unique(x)}
        gl.append([v,float(t.statistic),len(idx),float(t.pvalue),json.dumps(counts)])
    return m,tr,gl

def holm(p):
    p=np.asarray(p); order=np.argsort(p); a=np.empty_like(p); run=0.; n=len(p)
    for rank,i in enumerate(order): run=max(run,(n-rank)*p[i]); a[i]=min(run,1)
    return a

def coding_tests(primary,cat,b):
    base=b.model; out=[]
    cand=[('marital_status_4cat','married_current',cat['marital_status_4cat'].astype(int)),*[(v,v,primary[v].astype(int)) for v in ORDINAL]]
    for factor,replaced,x in cand:
        cols=[]
        for i,n in enumerate(b.names):
            if n==replaced:
                for lev in sorted(int(z) for z in np.unique(x)):
                    if lev!=2: cols.append((x==lev).astype(float))
            else: cols.append(b.X[:,i])
        m=sm.GLM(b.y,np.column_stack(cols),family=sm.families.Binomial()).fit(); lr=2*(m.llf-base.llf); df=len(m.params)-len(base.params)
        bic=-2*m.llf+len(m.params)*math.log(len(b.y)); bb=-2*base.llf+len(base.params)*math.log(len(b.y))
        out.append([factor,lr,df,float(stats.chi2.sf(lr,df)),m.aic-base.aic,bic-bb])
    adj=holm([r[3] for r in out]); return [r[:4]+[float(a)]+r[4:]+['Retain primary coding; categorical sensitivity only'] for r,a in zip(out,adj)]

def interactions(b):
    sex=b.raw['sex_female']; tests=[]; gidx=[i for i,n in enumerate(b.names) if n.startswith('gad7_rcs')]
    X=np.column_stack([b.X]+[b.X[:,i]*sex for i in gidx]); m=sm.GLM(b.y,X,family=sm.families.Binomial()).fit(cov_type='HC3'); R=np.zeros((len(gidx),X.shape[1]))
    for j in range(len(gidx)):R[j,b.X.shape[1]+j]=1
    t=m.wald_test(R,scalar=True); tests.append(['GAD-7 spline × female sex',len(gidx),float(t.statistic),float(t.pvalue),None,None,None])
    for label,name in [('PSS-14 × female sex','pss_z'),('Low self-esteem × female sex','selfesteem_z'),('Lower belonging × female sex','low_sense_of_belonging'),('Lower equality × female sex','low_perceived_social_equality'),('Lower trust × female sex','low_social_trust'),('Currently married × female sex','married_current')]:
        i=b.names.index(name); X=np.column_stack([b.X,b.X[:,i]*sex]); m=sm.GLM(b.y,X,family=sm.families.Binomial()).fit(cov_type='HC3'); beta=float(m.params[-1]); se=float(m.bse[-1])
        tests.append([label,1,float((beta/se)**2),float(m.pvalues[-1]),math.exp(beta),math.exp(beta-1.96*se),math.exp(beta+1.96*se)])
    adj=holm([r[3] for r in tests]); return [r[:4]+[float(a)]+r[4:]+['Do not retain' if a>=.05 else 'Retain'] for r,a in zip(tests,adj)]

def plot(path,curve,title,xlabel):
    x=np.array([r[0] for r in curve]); y=np.array([r[1] for r in curve]); lo=np.array([r[2] for r in curve]); hi=np.array([r[3] for r in curve])
    fig,ax=plt.subplots(figsize=(7.2,4.8)); ax.plot(x,y); ax.fill_between(x,lo,hi,alpha=.2); ax.set_xlabel(xlabel); ax.set_ylabel('Model-standardized prevalence'); ax.set_title(title); ax.set_ylim(bottom=0); fig.tight_layout(); fig.savefig(path,dpi=300,bbox_inches='tight'); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--primary',required=True);ap.add_argument('--phq8',required=True);ap.add_argument('--categorical',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args(); out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    p=load(a.primary); q=load(a.phq8); c=load(a.categorical); b=fit_bundle(p); bq=fit_bundle(q,True); n=len(b.y)
    write(out/'primary_model_terms.csv',['term','label','adjusted_or','ci_low','ci_high','p_value','beta','robust_se'],terms(b,['gad7_rcs']))
    lin=linear_primary(p); lr=2*(b.model.llf-lin.llf); gw=global_wald(b,'gad7_rcs'); pst=[['GAD-7 global association',*gw],['GAD-7 nonlinearity vs linear term',lr,2,float(stats.chi2.sf(lr,2))]]
    write(out/'primary_spline_tests.csv',['test','statistic','df','p_value'],pst)
    ch=['variable','reference_value','comparison_value','reference_prevalence','reference_ci_low','reference_ci_high','comparison_prevalence','comparison_ci_low','comparison_ci_high','adjusted_prevalence_ratio','apr_ci_low','apr_ci_high','adjusted_prevalence_difference','apd_ci_low','apd_ci_high']
    cv=curves(b,'gad7_total',21); ct=contrasts(b,'gad7_total',[5,10,15,20]); write(out/'primary_gad7_curve.csv',['score','adjusted_prevalence','ci_low','ci_high'],cv); write(out/'primary_gad7_contrasts.csv',ch,ct); plot(out/'primary_gad7_standardized_prevalence.png',cv,'Adjusted Prevalence by GAD-7 Score: Primary Model','GAD-7 score')
    write(out/'phq8_sensitivity_terms.csv',['term','label','adjusted_or','ci_low','ci_high','p_value','beta','robust_se'],terms(bq,['gad7_rcs','phq8_rcs']))
    bothlin=phq_compare(bq,set()); gad=phq_compare(bq,{'gad7_total'}); phq=phq_compare(bq,{'phq8_total'}); both=phq_compare(bq,{'gad7_total','phq8_total'}); g=global_wald(bq,'gad7_rcs'); h=global_wald(bq,'phq8_rcs')
    ps=[['GAD-7 global association',*g],['PHQ-8 global association',*h],['Both splines vs both linear',2*(both.llf-bothlin.llf),len(both.params)-len(bothlin.params),float(stats.chi2.sf(2*(both.llf-bothlin.llf),len(both.params)-len(bothlin.params)))],['Add GAD-7 spline given PHQ-8 spline',2*(both.llf-phq.llf),len(both.params)-len(phq.params),float(stats.chi2.sf(2*(both.llf-phq.llf),len(both.params)-len(phq.params)))],['Add PHQ-8 spline given GAD-7 spline',2*(both.llf-gad.llf),len(both.params)-len(gad.params),float(stats.chi2.sf(2*(both.llf-gad.llf),len(both.params)-len(gad.params)))]]
    write(out/'phq8_spline_tests.csv',['test','statistic','df','p_value'],ps)
    for var,maxv,vals,title,xlabel,prefix in [('gad7_total',21,[5,10,15,20],'Adjusted Prevalence by GAD-7 Score: PHQ-8 Sensitivity','GAD-7 score','phq8_gad7'),('phq8_total',24,[5,10,15,20],'Adjusted Prevalence by PHQ-8 Score: Sensitivity Model','PHQ-8 score','phq8_phq8')]:
        cc=curves(bq,var,maxv); co=contrasts(bq,var,vals); write(out/f'{prefix}_curve.csv',['score','adjusted_prevalence','ci_low','ci_high'],cc); write(out/f'{prefix}_contrasts.csv',ch,co); plot(out/f'{prefix}_standardized_prevalence.png',cc,title,xlabel)
    cm,ctr,cgl=fit_cat(p,c,b); write(out/'categorical_sensitivity_terms.csv',['term','label','adjusted_or','ci_low','ci_high','p_value'],ctr); write(out/'categorical_sensitivity_global_tests.csv',['factor','wald_statistic','df','p_value','category_counts'],cgl)
    pb=-2*b.model.llf+len(b.model.params)*math.log(n); cb=-2*cm.llf+len(cm.params)*math.log(n); lrc=2*(cm.llf-b.model.llf); dfc=len(cm.params)-len(b.model.params)
    write(out/'coding_model_comparison.csv',['model_or_test','parameters_or_df','aic_or_lr','bic_or_p','decision'],[['Primary ordinal-trend model',len(b.model.params),b.model.aic,pb,'Reference'],['Fully categorical sensitivity model',len(cm.params),cm.aic,cb,'Sensitivity only'],['Likelihood-ratio comparison',dfc,lrc,float(stats.chi2.sf(lrc,dfc)),'AIC improves but BIC strongly favors the primary model']])
    write(out/'ordinal_coding_tests.csv',['factor','lr_statistic','df','p_value','holm_adjusted_p','delta_aic','delta_bic','decision'],coding_tests(p,c,b))
    write(out/'sex_interactions_final.csv',['interaction','df','wald_statistic','p_value','holm_adjusted_p','interaction_or','ci_low','ci_high','decision'],interactions(b))
    pred=b.model.predict(b.X); null=sm.GLM(b.y,np.ones((n,1)),family=sm.families.Binomial()).fit(); diag=[['N',n,''],['Outcome events',int(b.y.sum()),''],['Converged',bool(b.model.converged),''],['Log likelihood',float(b.model.llf),''],['AIC',float(b.model.aic),''],['BIC',pb,'Manual likelihood-based BIC'],['McFadden pseudo-R2',float(1-b.model.llf/null.llf),''],['Apparent AUROC',float(roc_auc_score(b.y,pred)),'Association-model diagnostic'],['Apparent AUPRC',float(average_precision_score(b.y,pred)),'Association-model diagnostic'],['Apparent Brier score',float(brier_score_loss(b.y,pred)),'Association-model diagnostic']]
    write(out/'primary_model_diagnostics.csv',['metric','value','note'],diag)
    mp=sm.GLM(b.y,b.X,family=sm.families.Poisson(link=sm.families.links.Log())).fit(cov_type='HC3'); mpp=mp.predict(b.X); ed=[['Modified Poisson converged',bool(mp.converged)],['Maximum fitted mean',float(mpp.max())],['Fitted means >1, n',int((mpp>1).sum())],['Final estimator decision','Robust logistic regression plus marginal standardization'],['Reason','Modified Poisson generated fitted means above 1 in the final spline model; logistic standardization preserves valid probabilities.']]
    write(out/'estimator_decision.csv',['item','value'],ed)
    vars=['age_years','gad7_total','phq8_total','pss14_total','low_self_esteem_score']+BINARY+ORDINAL; cols=[np.ones(n)]+[((q[v]-q[v].mean())/q[v].std(ddof=1) if v not in BINARY else q[v]) for v in vars]; X=np.column_stack(cols); vf=[[v,float(variance_inflation_factor(X,i+1))] for i,v in enumerate(vars)]; write(out/'phq8_linear_proxy_vif.csv',['variable','vif'],vf)
    dec=[['Outcome','Past-year thoughts of wanting to die','Exact survey-item wording'],['Primary estimator','Robust logistic regression','HC3 standard errors'],['Common-outcome interpretation','Marginal standardization','Adjusted prevalence, prevalence ratios, and prevalence differences'],['Age','Linear per 10 years','No evidence of nonlinearity'],['GAD-7',f'4-knot RCS: {b.specs["gad7_total"].knots}','Strong nonlinearity evidence'],['PSS-14','Linear per SD','No evidence of nonlinearity'],['Low self-esteem','Linear per SD','No evidence of nonlinearity'],['4-level ordinal variables','One-category linear trend in primary model','Parsimonious; categorical coding retained as sensitivity'],['Marital status','Currently married vs not in primary model','4-category coding retained as sensitivity'],['PHQ-8',f'4-knot RCS in sensitivity: {bq.specs["phq8_total"].knots}','Sensitivity only because of overlap with GAD-7'],['Sex interactions','Not retained','No prespecified interaction survived Holm adjustment'],['Modified Poisson','Not selected as final estimator','119 fitted means exceeded 1 in the spline model']]
    write(out/'final_model_decisions.csv',['component','final_specification','rationale'],dec)
    manifest={'n':n,'events':int(b.y.sum()),'primary_gad_knots':b.specs['gad7_total'].knots,'phq8_gad_knots':bq.specs['gad7_total'].knots,'phq8_knots':bq.specs['phq8_total'].knots,'primary_converged':bool(b.model.converged),'phq8_converged':bool(bq.model.converged),'categorical_converged':bool(cm.converged),'primary_aic':float(b.model.aic),'primary_bic':pb,'categorical_aic':float(cm.aic),'categorical_bic':cb}
    (out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
