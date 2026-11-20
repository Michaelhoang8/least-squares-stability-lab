from pathlib import Path
import csv
import hashlib
import json
import os
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from stability_lab.core import make_problem,rounded,solve,reference_solution,metrics,METHODS


class NumericalChecks(unittest.TestCase):
    def test_well_conditioned_methods_recover_known_coefficients(self):
        p=make_problem(42,10)
        for dtype,tolerance in [('float32',2e-5),('float64',2e-13)]:
            a,b=rounded(p,dtype)
            for method in METHODS:
                solution=solve(a,b,method)
                self.assertEqual(solution.status,'ok')
                np.testing.assert_allclose(solution.x,p.truth,rtol=tolerance,atol=tolerance)

    def test_seed_and_requested_spectrum(self):
        first=make_problem(42,1e6);second=make_problem(42,1e6)
        self.assertTrue(np.array_equal(first.a,second.a))
        self.assertFalse(np.array_equal(first.a,make_problem(43,1e6).a))
        np.testing.assert_allclose(np.linalg.svd(first.a,compute_uv=False),first.target_singular_values,rtol=2e-10)

    def test_duplicate_columns_have_no_unique_coefficient_recovery(self):
        p=make_problem(42,100,rank_deficient=True)
        for dtype in ['float32','float64']:
            a,b=rounded(p,dtype)
            for method in ['normal_equations','qr']:
                self.assertEqual(solve(a,b,method).status,'rank_deficient_rejected')
                self.assertIsNone(solve(a,b,method).x)
            svd=solve(a,b,'svd')
            self.assertEqual(svd.effective_rank,9)
            self.assertEqual(svd.status,'truncated')
            ref=reference_solution(a,b,duplicate_column=True)
            self.assertAlmostEqual(ref[0],ref[-1],places=14)
            shifted=ref.copy();shifted[0]+=2;shifted[-1]-=2
            np.testing.assert_allclose(a.astype(float)@shifted,a.astype(float)@ref,atol=2e-15)
            self.assertGreater(np.linalg.norm(shifted),np.linalg.norm(ref))

    def test_reference_converges_and_solves_same_rounded_input(self):
        p=make_problem(11,1e10,1e-8)
        for dtype in ['float32','float64']:
            a,b=rounded(p,dtype)
            ref80=reference_solution(a,b,digits=80)
            ref110=reference_solution(a,b,digits=110)
            np.testing.assert_array_equal(ref80,ref110)
            # Rounding/noise changes the inverse problem; reference is not truth.
            self.assertGreater(np.linalg.norm(ref80-p.truth),1e-3)
            self.assertLess(np.linalg.norm(a.astype(float).T@(a.astype(float)@ref80-b.astype(float))),1e-11)

    def test_explicit_svd_cutoff_reports_truncation(self):
        p=make_problem(42,1e10)
        a,b=rounded(p,'float32')
        out=solve(a,b,'svd')
        self.assertEqual(out.status,'truncated')
        self.assertLess(out.effective_rank,10)
        self.assertEqual(out.cutoff_ratio,max(a.shape)*np.finfo(np.float32).eps)

    def test_published_coefficients_reconcile_to_metrics_and_inputs(self):
        folder=Path(os.environ.get('STABILITY_RESULTS',ROOT/'results/published'))
        manifest=json.loads((folder/'input_manifest.json').read_text())
        for record in manifest:
            self.assertEqual(hashlib.sha256((folder/record['path']).read_bytes()).hexdigest(),record['sha256'])
        inputs={record['case']:record['path'] for record in manifest}
        with (folder/'coefficients.csv').open() as f:
            coeff={}
            for row in csv.DictReader(f):
                coeff.setdefault((row['case'],row['method']),[]).append((int(row['index']),float(row['coefficient'])))
        with (folder/'measurements.csv').open() as f:
            for row in csv.DictReader(f):
                if row['status'] not in ('ok','truncated'):
                    self.assertEqual(row['reference_gap'],'');continue
                d=np.load(folder/inputs[row['case']])
                values=np.array([v for _,v in sorted(coeff[(row['case'],row['method'])])])
                expected=metrics(d['a'],d['b'],values,d['reference'],d['truth'])
                for key,value in expected.items():
                    self.assertAlmostEqual(value,float(row[key]),delta=max(1e-16,abs(value)*1e-12))

if __name__=='__main__':unittest.main()
