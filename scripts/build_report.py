"""Render charts and an offline report from the preserved measured run."""
from pathlib import Path
import csv
import html
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results/published'
CHARTS=ROOT/'docs/charts'
CHARTS.mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader((RESULTS/'measurements.csv').open()))
meta=json.loads((RESULTS/'environment.json').read_text())
COLORS={'normal_equations':'#b85e3f','qr':'#087f8c','svd':'#353c68'}
LABELS={'normal_equations':'Normal equations','qr':'QR','svd':'SVD with cutoff'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
                    'axes.titleweight':'bold','axes.titlelocation':'left',
                    'axes.spines.top':False,'axes.spines.right':False,
                    'figure.facecolor':'#fcfcf8','axes.facecolor':'#fcfcf8',
                    'axes.edgecolor':'#a7bcb9','grid.color':'#dce7e4','savefig.dpi':180})

def export(fig,name):
    fig.text(.01,.01,'Synthetic 48 × 10 matrices · 3 seeds · measured run '+meta['started_utc'][:10],fontsize=8,color='#5a7470')
    fig.tight_layout(rect=(0,.04,1,1))
    for suffix in ['png','svg']:fig.savefig(CHARTS/f'{name}.{suffix}',bbox_inches='tight')
    plt.close(fig)

def select(**criteria):
    return [r for r in rows if all(str(r[k])==str(v) for k,v in criteria.items())]

def med(values,key):return float(np.median([float(r[key]) for r in values if r[key]!='']))

conditions=meta['conditions']
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,dtype in zip(axes,['float32','float64']):
    for method,color in COLORS.items():
        chosen=select(dtype=dtype,method=method,scenario='full_rank',noise_relative='0.0')
        ys=[med([r for r in chosen if float(r['target_condition'])==c],'reference_gap') for c in conditions]
        ax.loglog(conditions,ys,'o-',color=color,lw=2,label=LABELS[method],ms=5)
        if dtype=='float32' and method=='svd':
            idx=[i for i,c in enumerate(conditions) if any(r['status']=='truncated' for r in chosen if float(r['target_condition'])==c)]
            ax.scatter([conditions[i] for i in idx],[ys[i] for i in idx],s=85,facecolor='#fcfcf8',edgecolor=color,zorder=5,label='SVD rank truncated')
    ax.set(title=dtype,xlabel='Target construction condition number',ylabel='Relative coefficient gap to 80-digit reference')
    ax.grid(which='major',alpha=.65);ax.legend(frameon=False,fontsize=8)
export(fig,'01_reference_gap')

fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,dtype in zip(axes,['float32','float64']):
    for method,color in COLORS.items():
        chosen=[r for r in select(dtype=dtype,method=method,scenario='full_rank') if r['status'] in ('ok','truncated')]
        ax.scatter([max(float(r['relative_residual']),1e-18) for r in chosen],
                   [max(float(r['reference_gap']),1e-18) for r in chosen],color=color,s=26,alpha=.7,label=LABELS[method])
    ax.set(xscale='log',yscale='log',xlabel='Relative residual on the rounded input',ylabel='Coefficient gap to reference',title=dtype)
    ax.grid(alpha=.65);ax.legend(frameon=False,fontsize=8)
export(fig,'02_residual_vs_gap')

fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,dtype in zip(axes,['float32','float64']):
    for noise,color,label in [('0.0','#087f8c','No added noise'),('1e-08','#b85e3f','Relative added noise 10⁻⁸')]:
        chosen=select(dtype=dtype,method='qr',scenario='full_rank',noise_relative=noise)
        values=[med([r for r in chosen if float(r['target_condition'])==c],'reference_recovery_gap') for c in conditions]
        ax.loglog(conditions,values,'o-',label=label,color=color,lw=2,ms=5)
    ax.set(title=dtype,xlabel='Target construction condition number',ylabel='High-precision reference gap to generating coefficients')
    ax.grid(alpha=.65);ax.legend(frameon=False,fontsize=8)
export(fig,'03_noise_and_rounding')

fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for ax,dtype in zip(axes,['float32','float64']):
    chosen=select(dtype=dtype,scenario='full_rank',method='qr',noise_relative='0.0')
    actual=[med([r for r in chosen if float(r['target_condition'])==c],'observed_condition64') for c in conditions]
    ax.loglog(conditions,conditions,'--',color='#a7bcb9',label='Target equals observed')
    ax.loglog(conditions,actual,'o-',color='#087f8c',label='Rounded matrix, float64 diagnostic')
    ax.set(title=dtype,xlabel='Target before dtype conversion',ylabel='Observed condition number of stored A')
    ax.grid(alpha=.65);ax.legend(frameon=False,fontsize=8)
export(fig,'04_stored_condition')

case=select(scenario='duplicate_column',dtype='float64',method='svd')[0]
data=np.load(RESULTS/'inputs'/f"{case['case']}.npz")
fig,ax=plt.subplots(figsize=(11,4.8));index=np.arange(10)
ax.bar(index-.18,data['truth'],width=.36,color='#b85e3f',label='Generating coefficients (not identifiable)')
ax.bar(index+.18,data['reference'],width=.36,color='#087f8c',label='Minimum-norm reference')
ax.set_xticks(index,[f'x{i}' for i in index]);ax.set_ylabel('Coefficient value');ax.set_title('Identical first and last columns leave one unconstrained direction')
ax.axhline(0,color='#a7bcb9',lw=1);ax.legend(frameon=False,fontsize=9)
export(fig,'05_rank_deficiency')

highlight32={m:med(select(dtype='float32',method=m,scenario='full_rank',noise_relative='0.0',target_condition='10000.0'),'reference_gap') for m in COLORS}
highlight64={m:med(select(dtype='float64',method=m,scenario='full_rank',noise_relative='0.0',target_condition='100000000.0'),'reference_gap') for m in COLORS}
summary={'float32_condition_1e4_median_reference_gap':highlight32,
         'float64_condition_1e8_median_reference_gap':highlight64,
         'rows':len(rows),'cases':meta['input_cases'],
         'rejected_rows':sum(r['status']=='rank_deficient_rejected' for r in rows),
         'truncated_rows':sum(r['status']=='truncated' for r in rows)}
(RESULTS/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

table=''.join(f'<tr><td>{LABELS[m]}</td><td>{highlight32[m]:.3g}</td><td>{highlight64[m]:.3g}</td></tr>' for m in COLORS)
report=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Least-squares stability | Measured numerical study</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;color:#173c37;background:#fcfcf8;font:17px/1.6 Arial,sans-serif}}header{{padding:22px 6vw;border-bottom:1px solid #c9d9d4;display:flex;justify-content:space-between;font-size:13px;letter-spacing:.06em}}a{{color:#087f8c}}main{{max-width:1280px;margin:auto;padding:0 48px}}section{{padding:64px 0 44px;border-bottom:1px solid #c9d9d4;scroll-margin-top:0}}h1,h2{{font-family:Georgia,serif;font-weight:400;line-height:1.05;letter-spacing:-.04em}}h1{{font-size:76px;max-width:950px;margin:16px 0 28px}}h2{{font-size:42px;margin:12px 0 20px}}h3{{font-size:17px}}p{{max-width:900px}}.label{{font-size:12px;font-weight:700;letter-spacing:.17em;color:#087f8c;text-transform:uppercase}}.lead{{font-size:21px;max-width:860px}}.equation{{font:38px Georgia,serif;padding:26px 0;color:#087f8c}}.facts{{display:flex;gap:70px;padding:26px 0}}.facts strong{{font:43px Georgia,serif;display:block}}.facts span{{font-size:13px;color:#5a7470}}.split{{display:grid;grid-template-columns:1.1fr 1fr;gap:52px}}.note{{border-left:3px solid #087f8c;padding:12px 20px;background:#eef5f2;font-size:15px}}img{{display:block;width:100%;height:auto;margin:16px 0 4px}}figcaption{{font-size:13px;color:#5a7470}}table{{border-collapse:collapse;width:100%;font-size:15px;margin:18px 0}}td,th{{padding:14px 10px;text-align:left;border-bottom:1px solid #dce7e4}}th{{font-size:12px;color:#5a7470}}code,pre{{font-family:Consolas,monospace;font-size:14px}}pre{{padding:24px;background:#eef5f2;overflow:auto}}footer{{padding:30px 0;font-size:13px;color:#5a7470}}@media(max-width:800px){{main{{padding:0 20px}}h1{{font-size:48px}}.split{{display:block}}.facts{{gap:24px}}}}
</style>
<header><span>NUMERICAL COMPUTING / EXPERIMENT 01</span><span>Measured {meta['started_utc'][:10]} · NumPy {meta['numpy']}</span></header><main>
<section id="overview"><div class="label">Least-squares stability lab</div><h1>A small residual is not the whole answer.</h1>
<p class="lead">Normal equations, QR and SVD are applied to the same rounded inputs. An 80-digit reference separates solver error from the information already lost to noise and finite precision.</p>
<div class="equation">min<sub>x</sub> ‖Ax − b‖₂ &nbsp; · &nbsp; A ∈ ℝ<sup>48×10</sup></div>
<div class="facts"><div><strong>{meta['input_cases']}</strong><span>stored input cases</span></div><div><strong>{len(rows)}</strong><span>method measurements</span></div><div><strong>80</strong><span>reference decimal digits</span></div><div><strong>2</strong><span>floating-point formats</span></div></div>
<table><thead><tr><th>MEDIAN RELATIVE REFERENCE GAP</th><th>FLOAT32 / TARGET κ = 10⁴</th><th>FLOAT64 / TARGET κ = 10⁸</th></tr></thead><tbody>{table}</tbody></table>
<p class="note">Three seeds per configuration, no added noise in this table. These are controlled synthetic cases, not an accuracy guarantee for every matrix.</p></section>
<section id="solver-error"><div class="label">01 / Arithmetic</div><h2>Compare against the problem actually stored.</h2><p>Each solution is compared with high-precision QR on the exact binary values of its own A and b. SVD’s hollow markers indicate deliberate rank truncation; those gaps include a changed solution policy, not just arithmetic error.</p><img src="charts/01_reference_gap.png" alt="Median coefficient gap to high-precision reference across condition numbers"><figcaption>Median across three seeds. No added noise. The horizontal axis is the construction target; observed conditioning is checked separately.</figcaption></section>
<section id="residual"><div class="label">02 / Diagnostics</div><h2>Inspect coefficients alongside the residual.</h2><p>A small residual can coexist with a large coefficient gap when the inverse problem is ill-conditioned. These points include both noise levels and all full-rank construction cases.</p><img src="charts/02_residual_vs_gap.png" alt="Residual versus coefficient error for three methods"><figcaption>Diagnostics use float64 arithmetic on the rounded input. SVD includes explicitly flagged truncated cases.</figcaption></section>
<section id="conditioning"><div class="label">03 / Information loss</div><h2>Even the reference cannot undo rounding.</h2><p>The reference-to-generating-coefficients gap measures a different issue: noise and input representation have changed the problem before the solver starts.</p><img src="charts/03_noise_and_rounding.png" alt="High precision reference gap to generating coefficients with and without added noise"><div class="note">The target spectrum is generated in float64. Casting to float32 can change its smallest singular values. <a href="charts/04_stored_condition.png">Inspect the observed-condition plot.</a></div></section>
<section id="rank"><div class="label">04 / Identifiability</div><h2>Rank deficiency is a modeling constraint.</h2><p>With the first and last columns exactly identical, changing their coefficients by +c and −c leaves the prediction unchanged. Recovering the original pair is not a meaningful accuracy target.</p><img src="charts/05_rank_deficiency.png" alt="Generating coefficients compared with minimum norm reference for duplicate columns"><p class="note">Normal equations and QR explicitly reject these fixtures. SVD returns a flagged rank-9, minimum-norm solution. Equal fit does not imply unique coefficients.</p></section>
<section id="method"><div class="label">05 / Reproduce and audit</div><h2>The experiment is inspectable.</h2><div class="split"><div><h3>Inputs and policies</h3><p>Three fixed seeds. Seven target condition numbers from 1 to 10¹². Two relative noise levels, 0 and 10⁻⁸. Both float32 and float64 input-output workflows. Six additional duplicate-column cases.</p><p>NumPy can promote QR/SVD internals to double before casting factors back. These are input-output workflows, not pure FP32/FP64 kernels. Reference QR runs at 80 decimal digits. SVD keeps singular values above max(m,n)·ε(dtype)·σ₁. No regularization parameter is tuned to the known answer.</p></div><div><h3>Evidence</h3><p><a href="../results/published/measurements.csv">270 measured rows</a> · <a href="../results/published/coefficients.csv">Returned coefficients</a> · <a href="../results/published/input_manifest.json">Input hashes</a></p><p><a href="../results/published/environment.json">Environment and method settings</a> · <a href="methods.md">Definitions and limitations</a> · <a href="../tests/test_numerics.py">Reconciliation and semantic tests</a></p></div></div><pre>python -m pip install -r requirements.txt
python scripts/run_experiment.py --out outputs/my-run
python -m unittest discover -s tests -v</pre><p class="note">Existing measurements are protected from accidental overwrite. CI creates a separate small run and checks the committed full run independently. Single-call timings are trace data, not a performance ranking.</p></section>
<footer>AI-assisted reproducible study · Synthetic matrices · No production accuracy or business-impact claims.</footer></main></html>'''
(ROOT/'docs/report.html').write_text(report,encoding='utf-8')
memo=f'''# Findings from the published run

The experiment stores {meta['input_cases']} rounded input cases and {len(rows)} method measurements. It compares normal equations, QR and SVD on 48 × 10 synthetic systems across three seeds.

At target condition 10⁴ in float32, the median relative coefficient gap to the same-input high-precision reference is **{highlight32['normal_equations']:.3g}** for normal equations, **{highlight32['qr']:.3g}** for QR and **{highlight32['svd']:.3g}** for untruncated SVD. At target condition 10⁸ in float64, the corresponding gaps are **{highlight64['normal_equations']:.3g}**, **{highlight64['qr']:.3g}**, and **{highlight64['svd']:.3g}**. These summaries use no added noise and three seeds.

## Interpretation

Forming AᵀA loses accuracy earlier in this study. Its singular values are squared in exact arithmetic, making its condition number κ₂(A)² for a full-column-rank A. QR avoids this formation. SVD also exposes a rank cutoff, but truncating directions intentionally changes the answer: its reference gap must not be described entirely as numerical error.

Reference-to-truth gaps rise with input rounding and noise. A more accurate solver cannot reconstruct information absent from the input. The exact duplicate-column fixtures demonstrate a separate issue: coefficients are nonunique. Normal equations and QR reject those fixtures; the SVD result is compared with the minimum-norm reference, not treated as recovery of the generating vector.

## Practical decision

For an unregularized dense least-squares workflow, compare QR or an SVD-based solve before accepting normal equations. Record the rank threshold and inspect coefficients as well as residuals. For ill-conditioned data, investigate scaling, measurement noise, identifiability, and an explicitly chosen regularization model; changing the solver alone is insufficient.

## Limits

These small synthetic matrices do not cover sparse, streaming, constrained, or distributed solvers. Three seeds describe the chosen constructions, not a population confidence interval. NumPy may promote linalg internals to double and cast outputs back; this is an input-output workflow comparison, not a pure-precision kernel benchmark. Float64 diagnostics and the stored float64 reference introduce a display floor near machine precision. Eighty-digit reference precision was checked against 110 digits on representative difficult cases, not proved sufficient for arbitrary inputs. Single-call timings include rank diagnostics and are not a speed benchmark.
'''
(ROOT/'docs/findings.md').write_text(memo,encoding='utf-8')
print(ROOT/'docs/report.html')
