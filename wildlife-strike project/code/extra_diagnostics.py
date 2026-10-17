"""Generate precision-recall plot and capacity sensitivity from frozen predictions."""
from pathlib import Path
import math
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
root=Path(__file__).resolve().parents[1]
d=pd.read_csv(root/'results/predictions.csv',parse_dates=['date']);d=d[d.date.dt.year==2024];rows=[]
fig,ax=plt.subplots(figsize=(7,5))
for name,g in d.groupby('model'):
 p,r,_=precision_recall_curve(g.y,g.probability);ax.plot(r,p,label=name)
 for capacity in [.1,.2,.3,.5]:
  selected=pd.concat([q.sort_values(['probability','AIRPORT_ID'],ascending=[False,True]).head(math.ceil(capacity*len(q))) for _,q in g.groupby('date')]);caught=int(selected.y.sum());rows.append(dict(model=name,capacity=capacity,reviewed=len(selected),captured=caught,missed=int(g.y.sum())-caught,precision=caught/len(selected),recall=caught/g.y.sum()))
ax.axhline(d.y.mean(),linestyle='--',color='gray',label='Test prevalence');ax.set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1));ax.legend();fig.tight_layout();fig.savefig(root/'figures/precision_recall.png',dpi=180);pd.DataFrame(rows).to_csv(root/'results/capacity_sensitivity.csv',index=False)
