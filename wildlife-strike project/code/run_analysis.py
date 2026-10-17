"""Reproducible FAA airport-month analysis. Run with --input workbook.xlsx --output project.
The source extract is retrospective: incident dates do not establish report availability.
"""
import argparse,json,hashlib,math,platform
from pathlib import Path
import numpy as np,pandas as pd,openpyxl,joblib,sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load(path,out):
 cache=out/'results'/'selected_source_fields.csv'
 stamp=out/'results'/'source_cache_sha256.txt'
 source_hash=hashlib.sha256(Path(path).read_bytes()).hexdigest()
 if cache.exists() and stamp.exists() and stamp.read_text().strip()==source_hash:return pd.read_csv(cache,keep_default_na=False)
 w=openpyxl.load_workbook(path,read_only=True,data_only=True);s=w.worksheets[0];it=s.iter_rows(values_only=True);head=next(it)
 cols=['INDEX_NR','INCIDENT_DATE','INCIDENT_YEAR','INCIDENT_MONTH','AIRPORT_ID','AIRPORT','INDICATED_DAMAGE','SIZE','COST_REPAIRS_INFL_ADJ','COST_OTHER_INFL_ADJ']
 ix=[head.index(c) for c in cols];rows=[]
 for row in it:rows.append([row[i] for i in ix])
 d=pd.DataFrame(rows,columns=cols);d.to_csv(cache,index=False);stamp.write_text(source_hash);return d

def assess(d,p,name,year):
 y=d.y.to_numpy();selected=[]
 for _,g in d.assign(score=p).groupby('date'):
  selected.extend(g.sort_values(['score','AIRPORT_ID'],ascending=[False,True]).head(math.ceil(.2*len(g))).index.tolist())
 mask=d.index.isin(selected);caught=int(y[mask].sum());total=int(y.sum());n=int(mask.sum())
 return dict(year=year,model=name,n=len(y),events=total,reviewed=n,captured=caught,missed=total-caught,precision=caught/n,recall=caught/total,ap=average_precision_score(y,p),auc=roc_auc_score(y,p),brier=brier_score_loss(y,p))

def main():
 a=argparse.ArgumentParser();a.add_argument('--input',required=True);a.add_argument('--output',default='project');args=a.parse_args();out=Path(args.output)
 for x in ['results','figures','models']: (out/x).mkdir(parents=True,exist_ok=True)
 raw=load(args.input,out);d=raw.drop_duplicates('INDEX_NR').copy();d['date']=pd.to_datetime(d.INCIDENT_DATE,errors='coerce').dt.to_period('M').dt.to_timestamp();d['damage']=pd.to_numeric(d.INDICATED_DAMAGE,errors='coerce');d['AIRPORT_ID']=d.AIRPORT_ID.fillna('').astype(str).str.strip().str.upper();d['SIZE']=d.SIZE.fillna('').astype(str).str.strip().str.title()
 audit=dict(raw_rows=len(raw),unique_rows=len(d),duplicates=len(raw)-len(d),date_min=str(d.date.min()),date_max=str(d.date.max()),invalid_dates=int(d.date.isna().sum()),invalid_damage=int((~d.damage.isin([0,1])).sum()))
 good=d.SIZE.isin(['Small','Medium','Large'])&d.damage.isin([0,1]);eda=d[good].groupby('SIZE').damage.agg(['size','sum','mean']).reindex(['Small','Medium','Large']);n=eda['size'];p=eda['mean'];z=1.96;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;eda['ci_low']=center-half;eda['ci_high']=center+half;eda.to_csv(out/'results'/'size_damage.csv');audit['eda_included']=int(good.sum());audit['eda_excluded']=int((~good).sum())
 d.assign(excluded=~good).groupby(d.date.dt.year).agg(reports=('INDEX_NR','size'),excluded=('excluded','sum')).to_csv(out/'results'/'exclusions_by_year.csv')
 valid=d[d.date.between('2012-01-01','2024-12-01')&d.AIRPORT_ID.str.match(r'^[A-Z0-9]{4}$')&d.AIRPORT_ID.ne('ZZZZ')].copy();valid['damage']=valid.damage.eq(1).astype(int)
 grouped=valid.groupby(['AIRPORT_ID','date']).agg(strikes=('INDEX_NR','size'),damages=('damage','sum'))
 hist=grouped.reset_index();hist=hist[hist.date.le('2022-12-01')];coverage=hist.groupby('AIRPORT_ID').agg(reporting_months=('date','size'),damaging_months=('damages',lambda x:int((x>0).sum())));coverage['eligible']=(coverage.reporting_months>=119)&(coverage.damaging_months>=12);coverage.to_csv(out/'results'/'airport_eligibility.csv');airports=coverage.index[coverage.eligible].tolist()
 index=pd.MultiIndex.from_product([airports,pd.date_range('2012-01-01','2024-12-01',freq='MS')],names=['AIRPORT_ID','date']);panel=grouped.reindex(index,fill_value=0).reset_index();panel['y']=panel.damages.gt(0).astype(int)
 features=[]
 for c in ['strikes','damages']:
  for w in [1,3,12]:
   name=f'{c}_prior_{w}m';panel[name]=panel.groupby('AIRPORT_ID')[c].transform(lambda s:s.shift(1).rolling(w,min_periods=w).sum());features.append(name)
 panel['sin_month']=np.sin(2*np.pi*panel.date.dt.month/12);panel['cos_month']=np.cos(2*np.pi*panel.date.dt.month/12);features+=['sin_month','cos_month'];panel=panel.dropna(subset=features).reset_index(drop=True);panel.to_csv(out/'results'/'airport_month_panel.csv',index=False)
 assert len(airports)>0 and panel.groupby('date').size().nunique()==1
 # Training-only rolling folds. Each holdout year follows its training observations.
 def make(kind,value):
  pre=ColumnTransformer([('num',StandardScaler(),features),('airport',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['AIRPORT_ID'])])
  model=LogisticRegression(C=value,max_iter=2000,random_state=692) if kind=='logistic' else RandomForestClassifier(n_estimators=250,min_samples_leaf=value,max_features='sqrt',random_state=692,n_jobs=2)
  return Pipeline([('pre',pre),('model',model)])
 Xcols=features+['AIRPORT_ID'];cv=[];best={}
 for kind,values in [('logistic',[.01,.1,1,10]),('forest',[5,15,30])]:
  scores=[]
  for v in values:
   fold=[]
   for yr in [2020,2021,2022]:
    tr=panel[panel.date.dt.year<yr];va=panel[panel.date.dt.year==yr];m=make(kind,v);m.fit(tr[Xcols],tr.y);pr=m.predict_proba(va[Xcols])[:,1];r=assess(va,pr,kind,yr);r['setting']=v;cv.append(r);fold.append(r['ap'])
   scores.append((np.mean(fold),v))
  best[kind]=max(scores)[1]
 pd.DataFrame(cv).to_csv(out/'results'/'rolling_cv.csv',index=False)
 spec={'airports':airports,'features':features,'settings':best,'training_end':'2022-12','validation':'2023','test':'2024','capacity':'ceil(0.2 * eligible airports) EACH month','switch_rule':'forest captures at least 5 additional test months and Brier <= logistic Brier + .01','report_availability_caveat':'Incident month is a proxy for availability; ingestion timestamps unavailable.'};(out/'results'/'frozen_specification.json').write_text(json.dumps(spec,indent=2))
 metrics=[];preds=[];coeffs=[]
 for kind in best:
  m=make(kind,best[kind]);tr=panel[panel.date.dt.year<=2022];m.fit(tr[Xcols],tr.y)
  for yr in [2023,2024]:
   te=panel[panel.date.dt.year==yr];pr=m.predict_proba(te[Xcols])[:,1];metrics.append(assess(te,pr,kind,yr));q=te[['AIRPORT_ID','date','y']].copy();q['model']=kind;q['probability']=pr;q['selected']=False
   for _,g in q.groupby('date'):q.loc[g.sort_values(['probability','AIRPORT_ID'],ascending=[False,True]).head(math.ceil(.2*len(g))).index,'selected']=True
   preds.append(q)
  joblib.dump(m,out/'models'/f'{kind}.joblib')
  if kind=='logistic':
   names=m.named_steps['pre'].get_feature_names_out();pd.DataFrame({'feature':names,'coefficient':m.named_steps['model'].coef_[0]}).to_csv(out/'results'/'coefficients.csv',index=False)
   for yr in [2020,2021,2022]:
    mm=make(kind,best[kind]);tt=panel[panel.date.dt.year<yr];mm.fit(tt[Xcols],tt.y)
    for name,c in zip(features,mm.named_steps['model'].coef_[0][:len(features)]):coeffs.append({'fold':yr,'feature':name,'coefficient':c})
 # Historical airport prevalence baseline using only 2013-2022 target observations.
 rates=tr.groupby('AIRPORT_ID').y.mean()
 for yr in [2023,2024]:
  te=panel[panel.date.dt.year==yr];metrics.append(assess(te,te.AIRPORT_ID.map(rates).to_numpy(),'airport_history',yr));metrics.append(assess(te,np.repeat(tr.y.mean(),len(te)),'constant',yr))
 pd.DataFrame(coeffs).to_csv(out/'results'/'coefficient_stability.csv',index=False);mt=pd.DataFrame(metrics);mt.to_csv(out/'results'/'model_metrics.csv',index=False);pd.concat(preds).to_csv(out/'results'/'predictions.csv',index=False)
 test=mt[mt.year==2024].set_index('model');switch=test.loc['forest','captured']>=test.loc['logistic','captured']+5 and test.loc['forest','brier']<=test.loc['logistic','brier']+.01;audit.update(eligible_airports=len(airports),panel_rows=len(panel),switch_to_forest=bool(switch),chosen_model='forest' if switch else 'logistic',source_sha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),python=platform.python_version(),sklearn=sklearn.__version__);(out/'results'/'audit.json').write_text(json.dumps(audit,indent=2))
 plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(1,2,figsize=(12,4));ax[0].barh(eda.index,eda['size'],color='#315975');ax[0].set_xlabel('Reported strikes');ax[1].barh(eda.index,100*eda['mean'],color='#315975',xerr=100*(eda.ci_high-eda.ci_low)/2);ax[1].set_xlim(0,40);ax[1].set_xlabel('Reports indicating damage (%)');fig.tight_layout();fig.savefig(out/'figures'/'size_damage.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,4));tt=test.loc[['logistic','forest','airport_history']];ax.bar(tt.index,tt.captured,color='#315975');ax.set_ylabel('Damaging airport-months captured');ax.set_ylim(0,test.events.max());fig.tight_layout();fig.savefig(out/'figures'/'holdout_capture.png',dpi=180);plt.close(fig)
 from sklearn.calibration import calibration_curve
 fig,ax=plt.subplots(figsize=(6,5));ax.plot([0,1],[0,1],color='gray',linestyle='--')
 for q in preds:
  if q.date.dt.year.iloc[0]==2024:
   obs,prob=calibration_curve(q.y,q.probability,n_bins=6,strategy='quantile');ax.plot(prob,obs,marker='o',label=q.model.iloc[0])
 ax.set(xlabel='Mean predicted probability',ylabel='Observed damaging-month fraction',xlim=(0,1),ylim=(0,1));ax.legend();fig.tight_layout();fig.savefig(out/'figures'/'calibration.png',dpi=180);plt.close(fig)
 print(json.dumps(audit,indent=2));print(mt.to_string(index=False))
if __name__=='__main__':main()
