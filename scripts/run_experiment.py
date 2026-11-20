"""Run the controlled study; published results are never overwritten implicitly."""
from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys
import time
import numpy as np
import mpmath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from stability_lab.core import METHODS, make_problem, rounded, solve, reference_solution, metrics


def run(output, quick=False):
    output = Path(output)
    if (output/'measurements.csv').exists():
        raise FileExistsError('Choose a new output directory; existing measurements are preserved')
    (output/'inputs').mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    seeds = [811, 812, 813] if not quick else [811]
    conditions = [1., 1e2, 1e4, 1e6, 1e8, 1e10, 1e12] if not quick else [1., 1e4, 1e10]
    settings = [(s,c,n,False) for s in seeds for c in conditions for n in [0.,1e-8]]
    settings += [(s,100.,0.,True) for s in seeds]
    rows, coefficients, manifests = [], [], []
    for seed, condition, noise, deficient in settings:
        problem = make_problem(seed,condition,noise,rank_deficient=deficient)
        for dtype in ['float32','float64']:
            case = f'{problem.scenario}-s{seed}-k{condition:.0e}-n{noise:.0e}-{dtype}'
            a,b = rounded(problem,dtype)
            reference = reference_solution(a,b,duplicate_column=deficient)
            input_path = output/'inputs'/f'{case}.npz'
            np.savez_compressed(input_path,a=a,b=b,truth=problem.truth,
                                reference=reference,target_singular_values=problem.target_singular_values)
            manifests.append({'case':case,'path':input_path.relative_to(output).as_posix(),
                              'sha256':hashlib.sha256(input_path.read_bytes()).hexdigest()})
            for method in METHODS:
                start = time.perf_counter_ns()
                answer = solve(a,b,method)
                elapsed = time.perf_counter_ns()-start
                row = {'case':case,'scenario':problem.scenario,'seed':seed,'dtype':dtype,
                       'rows':a.shape[0],'columns':a.shape[1],'target_condition':condition,
                       'observed_condition64':float(np.linalg.cond(a.astype(np.float64))),
                       'noise_relative':noise,'a_rounding_relative':float(np.linalg.norm(a.astype(np.float64)-problem.a)/np.linalg.norm(problem.a)),
                       'method':method,'status':answer.status,'effective_rank':answer.effective_rank,
                       'cutoff_ratio':answer.cutoff_ratio,'single_solve_ns':elapsed,
                       'relative_residual':'','reference_gap':'','recovery_gap':'',
                       'reference_recovery_gap':'','coefficient_norm':''}
                if answer.x is not None:
                    row.update(metrics(a,b,answer.x,reference,problem.truth))
                    for index,value in enumerate(answer.x):
                        coefficients.append({'case':case,'method':method,'index':index,'coefficient':float(value)})
                rows.append(row)
    for name,data in [('measurements',rows),('coefficients',coefficients)]:
        with (output/f'{name}.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    config=np.__config__.CONFIG.get('Build Dependencies',{}).get('blas',{})
    metadata={'started_utc':started,'completed_utc':datetime.now(timezone.utc).isoformat(),
              'python':platform.python_version(),'numpy':np.__version__,'mpmath':mpmath.__version__,
              'os':platform.system(),'architecture':platform.machine(),
              'blas':{k:config.get(k) for k in ['name','version']},'seeds':seeds,
              'conditions':conditions,'matrix_shape':[48,10],'noise_relative':[0.,1e-8],
              'reference_decimal_digits':80,'input_cases':len(manifests),'measurement_rows':len(rows),
              'quick':quick,'timing_scope':'One call per case, including float64 rank diagnostic; descriptive trace, not a performance benchmark',
              'data_provenance':'Seeded synthetic matrices; no external dataset',
              'precision_scope':'Float32/float64 input-output workflows; NumPy linalg may promote internals to double and cast outputs back; matrix products use array dtype.',
              'rank_policy':'Normal equations/QR reject float64-diagnostic rank deficiency. SVD cutoff=max(shape)*eps(input dtype).',
              'reference_policy':'80-digit QR on exact binary rounded inputs; duplicate-column case uses reduced QR and equal split for minimum norm.'}
    (output/'environment.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    (output/'input_manifest.json').write_text(json.dumps(manifests,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(output),'cases':len(manifests),'measurements':len(rows)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/latest')
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args();run(args.out,args.quick)
