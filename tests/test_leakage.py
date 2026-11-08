import json
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from load import ROOT, FEATURES, load_data
from features import Features
from evaluate import optimal, RATIOS

class LeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train,cls.test=load_data()
        cls.audit=json.loads((ROOT/'results/audit.json').read_text())
        cls.data=json.loads((ROOT/'results/results.json').read_text())

    def test_inputs_and_split(self):
        self.assertEqual((len(self.train),len(self.test)),(125973,22544))
        self.assertEqual(len(FEATURES),41)
        self.assertNotIn('difficulty',FEATURES)
        self.assertEqual(int(self.test.seen.eq(0).sum()),3750)

    def test_actual_fit_provenance(self):
        normal=set(self.train.index[self.train.y.eq(0)])
        for seed,audits in self.audit.items():
            held=[]
            for a in audits:
                fitted=set(a['preprocess_fit_ids']); anomaly=set(a['anomaly_fit_ids'])
                self.assertEqual(anomaly,fitted & normal)
                self.assertEqual(a['cdf_rows'],len(anomaly))
                self.assertEqual(a['normal_fit_rows'],len(anomaly))
                if a['fold']!='full':
                    h=set(a['heldout_ids']); self.assertFalse(fitted & h)
                    self.assertEqual(fitted|h,set(self.train.index));held.extend(h)
            self.assertEqual(sorted(held),list(range(len(self.train))))

    def test_refit_changes_parameters(self):
        a=Features().fit(self.train)
        b=Features().fit(pd.concat([self.train,self.test],ignore_index=True))
        self.assertFalse(np.allclose(a.scaler.mean_,b.scaler.mean_))
        before=a.scaler.mean_.copy();a.transform(self.test)
        np.testing.assert_array_equal(before,a.scaler.mean_)

    def test_thresholds_recomputed_from_oof(self):
        scores=np.load(ROOT/'results/scores_42.npz')
        for model in self.data['models']:
            k=model['id']
            for r in RATIOS:self.assertEqual(optimal(self.train.y,scores['oof_'+k],r),self.data['thresholds'][k][str(r)])

    def test_exports(self):
        text=(ROOT/'dashboard/data.js').read_text(encoding='utf-8')
        self.assertEqual(json.loads(text.removeprefix('window.NSLKDD = ').removesuffix(';')),self.data)
        self.assertLess(len(text.encode()),8_000_000)
        self.assertEqual(len(self.data['samples']),300)
        for m in self.data['models']:
            k=m['id'];h=self.data['oof_hist'][k]
            self.assertEqual(sum(h['normal']),67343);self.assertEqual(sum(h['attack']),58630)
            self.assertLessEqual(len(self.data['roc'][k]['test']['fpr']),200)

if __name__=='__main__':unittest.main()
