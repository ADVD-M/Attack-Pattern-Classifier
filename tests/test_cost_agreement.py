import json
import subprocess
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from load import ROOT
from evaluate import confusion, optimal

class CostTests(unittest.TestCase):
    def test_python_node_parity(self):
        d=json.loads((ROOT/'results/results.json').read_text());r=d['test_records']
        script="const fs=require('fs'),f=require('./dashboard/app.js'),d=JSON.parse(fs.readFileSync('results/results.json')); let out=[];for(const m of d.models)for(const t of [0,.2,.5,.995,1,1.0001,d.thresholds[m.id]['10']])out.push(f.confusion(d.test_records.y_true,d.test_records[m.id],t,10));process.stdout.write(JSON.stringify(out));"
        js=json.loads(subprocess.check_output(['node','-e',script],cwd=ROOT,text=True))
        py=[confusion(r['y_true'],r[m['id']],t,10) for m in d['models'] for t in [0,.2,.5,.995,1,1.0001,d['thresholds'][m['id']]['10']]]
        self.assertEqual(js,py)

    def test_ties_and_endpoints(self):
        self.assertEqual(confusion([0,1],[.5,.5],.5)['tp'],1)
        self.assertEqual(confusion([0,1],[1,1],1.0001)['fp'],0)
        self.assertEqual(optimal([0,0],[1,1],10),1.0001)

if __name__=='__main__':unittest.main()
