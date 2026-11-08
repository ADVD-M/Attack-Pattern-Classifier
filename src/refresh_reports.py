"""Regenerate derived reports from saved scores without refitting models."""
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from load import ROOT, load_data
from evaluate import optimal
from report import write_report

def refresh():
    out=ROOT/'results'
    d=json.loads((out/'results.json').read_text())
    train,test=load_data()
    d['seed_stability']=[r for r in d['seed_stability'] if r['metric'] not in ['seen_recall','unseen_recall']]
    score_files={s:np.load(out/f'scores_{s}.npz') for s in [42,41,43]}
    for m in d['models']:
        k=m['id']
        for flag,label in [(1,'seen'),(0,'unseen')]:
            mask=test.y.eq(1)&test.seen.eq(flag)
            vals={s:float((score_files[s]['test_'+k][mask]>=optimal(train.y,score_files[s]['oof_'+k],10)).mean()) for s in score_files}
            metric=label+'_recall';d['test_metrics'][k][metric]=vals[42]
            d['seed_stability'].append(dict(model=k,setting='test',metric=metric,seed42=vals[42],seed41=vals[41],seed43=vals[43],max_change=max(abs(vals[s]-vals[42]) for s in [41,43])))
    text=json.dumps(d,separators=(',',':'),allow_nan=False)
    (out/'results.json').write_text(text,encoding='utf-8')
    (ROOT/'dashboard/data.js').write_text('window.NSLKDD = '+text+';',encoding='utf-8')
    pd.DataFrame(d['seed_stability']).to_csv(out/'seed_stability.csv',index=False)
    pd.DataFrame([dict(model=k,setting=setting,**v) for setting in ['cv','test'] for k,v in d[setting+'_metrics'].items()]).to_csv(out/'metrics.csv',index=False)
    verified=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,capture_output=True,text=True)
    (out/'validation.txt').write_text(verified.stdout+verified.stderr,encoding='utf-8')
    print(verified.stdout+verified.stderr,flush=True)
    verified.check_returncode()
    write_report(d)

if __name__=='__main__':refresh()
