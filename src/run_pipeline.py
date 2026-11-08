"""Run the complete experiment, exports and seed stability study."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import json
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from load import ROOT, FEATURES, HASHES, load_data
from models import MODELS, fit_score
from evaluate import RATIOS, metrics, optimal, roc

def experiment(train,test,seed):
    start=time.perf_counter()
    oof={m['id']:np.zeros(len(train)) for m in MODELS}
    audits=[]
    for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(train,train.y)):
        t=time.perf_counter()
        scores,audit=fit_score(train.iloc[a],train.iloc[b],seed)
        for k in oof: oof[k][b]=scores[k]
        audit.update(fold=fold,heldout_ids=b.tolist(),seed=seed)
        audits.append(audit)
        print(f'Seed {seed}, fold {fold+1}: {time.perf_counter()-t:.1f}s',flush=True)
    scores,audit=fit_score(train,test,seed)
    audit.update(fold='full',seed=seed)
    audits.append(audit)
    thresholds={k:{str(r):optimal(train.y,oof[k],r) for r in RATIOS} for k in oof}
    cv={k:metrics(train.y,oof[k],thresholds[k]['10']) for k in oof}
    tm={k:metrics(test.y,scores[k],thresholds[k]['10']) for k in oof}
    for k in tm:
        for flag,label in [(1,'seen'),(0,'unseen')]:
            mask=test.y.eq(1)&test.seen.eq(flag)
            tm[k][label+'_recall']=float((scores[k][mask]>=thresholds[k]['10']).mean())
    print(f'Seed {seed} complete: {time.perf_counter()-start:.1f}s',flush=True)
    return dict(oof=oof,scores=scores,audits=audits,thresholds=thresholds,cv=cv,test=tm,seconds=time.perf_counter()-start)

def main():
    start=time.perf_counter()
    out=ROOT/'results'; out.mkdir(exist_ok=True)
    train,test=load_data(); test.attrs['external_test']=True
    print(f'Loaded {len(train):,} / {len(test):,} rows; unseen attacks: 3,750',flush=True)
    runs={}
    with threadpool_limits(limits=1):
        for seed in [42,41,43]:
            runs[seed]=experiment(train,test,seed)
            # Preserve expensive computation if a later export check fails.
            np.savez_compressed(out/f'scores_{seed}.npz',**{f'oof_{k}':v for k,v in runs[seed]['oof'].items()},**{f'test_{k}':v for k,v in runs[seed]['scores'].items()})
    run=runs[42]
    names=sorted(set(train.label)|set(test.label)); name_index={n:i for i,n in enumerate(names)}
    attack_names=[dict(name=n,family=(train.loc[train.label.eq(n),'family'].iloc[0] if n in set(train.label) else test.loc[test.label.eq(n),'family'].iloc[0]),in_train=n in set(train.label),train_count=int(train.label.eq(n).sum()),test_count=int(test.label.eq(n).sum())) for n in names]
    family_rows=[]; name_rows=[]
    for k,scores in run['scores'].items():
        pred=scores>=run['thresholds'][k]['10']
        for fam in ['all','DoS','Probe','R2L','U2R']:
            row=dict(model=k,family=fam)
            for flag,label in [(1,'seen'),(0,'unseen')]:
                mask=test.y.eq(1)&test.seen.eq(flag)&(True if fam=='all' else test.family.eq(fam))
                row[label+'_count']=int(mask.sum()); row[label+'_recall']=float(pred[mask].mean()) if mask.any() else None
            family_rows.append(row)
        for n in names:
            mask=test.label.eq(n)
            if n!='normal' and mask.any(): name_rows.append(dict(model=k,attack_name=n,recall=float(pred[mask].mean()),test_count=int(mask.sum()),in_train=n in set(train.label)))
    rng=np.random.default_rng(42)
    selected=[]
    for _, group in test.groupby(['family','seen']):
        selected.extend(rng.choice(group.index,min(20,len(group)),replace=False).tolist())
    remaining=np.setdiff1d(test.index,selected)
    selected.extend(rng.choice(remaining,300-len(selected),replace=False).tolist())
    samples=[dict(id=int(i),features=test.loc[i,FEATURES].to_dict()) for i in sorted(selected)]
    stability=[]
    for k in run['scores']:
        for setting in ['cv','test']:
            for metric in ['accuracy','precision','recall','f1','fpr','auc','cost']+(['seen_recall','unseen_recall'] if setting=='test' else []):
                base=run[setting][k][metric]
                stability.append(dict(model=k,setting=setting,metric=metric,seed42=base,seed41=runs[41][setting][k][metric],seed43=runs[43][setting][k][metric],max_change=max(abs(runs[s][setting][k][metric]-base) for s in [41,43])))
    data=dict(meta=dict(train_rows=len(train),test_rows=len(test),seed=42,generated_at=datetime.now(timezone.utc).isoformat(),feature_count_after_encoding=run['audits'][-1]['features'],dropped_constant_columns=run['audits'][-1]['dropped'],input_sha256=HASHES,threshold_rule='score >= threshold',score_decimals=4,majority_baseline=float(test.y.mean()),runtime_seconds=time.perf_counter()-start),class_counts=dict(train=train.family.value_counts().to_dict(),test=test.family.value_counts().to_dict(),seen=9083,unseen=3750),attack_names=attack_names,models=MODELS,cv_metrics=run['cv'],test_metrics=run['test'],per_family_recall=family_rows,per_name_recall=name_rows,thresholds=run['thresholds'],roc={k:dict(cv=roc(train.y,run['oof'][k]),test=roc(test.y,run['scores'][k])) for k in run['scores']},oof_hist={k:{label:np.histogram(s[train.y.to_numpy()==v],bins=200,range=(0,1))[0].tolist() for label,v in [('normal',0),('attack',1)]} for k,s in run['oof'].items()},test_records=dict(id=test.index.tolist(),y_true=test.y.tolist(),family=test.family.tolist(),attack_name_index=test.label.map(name_index).tolist(),seen=test.seen.tolist(),difficulty=test.difficulty.tolist(),**{k:s.tolist() for k,s in run['scores'].items()}),samples=samples,seed_stability=stability)
    text=json.dumps(data,separators=(',',':'),allow_nan=False)
    (out/'results.json').write_text(text,encoding='utf-8')
    (ROOT/'dashboard'/'data.js').write_text('window.NSLKDD = '+text+';',encoding='utf-8')
    assert len(text.encode())<8_000_000
    pd.DataFrame([dict(model=k,r=r,threshold=v,source='OOF training') for k,vals in run['thresholds'].items() for r,v in vals.items()]).to_csv(out/'thresholds.csv',index=False)
    for name,rows in [('per_family_recall',family_rows),('per_name_recall',name_rows),('seed_stability',stability),('attack_names',attack_names)]: pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    pd.DataFrame([dict(model=k,setting=setting,**v) for setting in ['cv','test'] for k,v in run[setting].items()]).to_csv(out/'metrics.csv',index=False)
    (out/'audit.json').write_text(json.dumps({str(s):r['audits'] for s,r in runs.items()}),encoding='utf-8')
    import subprocess, sys
    verified=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,capture_output=True,text=True)
    (out/'validation.txt').write_text(verified.stdout+verified.stderr,encoding='utf-8')
    verified.check_returncode()
    from report import write_report
    write_report(data)
    print(f'Features: {data["meta"]["feature_count_after_encoding"]}; dashboard data: {len(text)/1e6:.2f} MB; total {time.perf_counter()-start:.1f}s',flush=True)
    assert time.perf_counter()-start < 600, 'Runtime exceeded the 10-minute requirement on this machine'

if __name__=='__main__': main()
