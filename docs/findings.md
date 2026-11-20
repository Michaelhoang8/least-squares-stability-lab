# Findings from the published run

The experiment stores 90 rounded input cases and 270 method measurements. It compares normal equations, QR and SVD on 48 × 10 synthetic systems across three seeds.

At target condition 10⁴ in float32, the median relative coefficient gap to the same-input high-precision reference is **1.04** for normal equations, **2.75e-05** for QR and **3.84e-05** for untruncated SVD. At target condition 10⁸ in float64, the corresponding gaps are **0.0435**, **1.52e-09**, and **1.91e-09**. These summaries use no added noise and three seeds.

## Interpretation

Forming AᵀA loses accuracy earlier in this study. Its singular values are squared in exact arithmetic, making its condition number κ₂(A)² for a full-column-rank A. QR avoids this formation. SVD also exposes a rank cutoff, but truncating directions intentionally changes the answer: its reference gap must not be described entirely as numerical error.

Reference-to-truth gaps rise with input rounding and noise. A more accurate solver cannot reconstruct information absent from the input. The exact duplicate-column fixtures demonstrate a separate issue: coefficients are nonunique. Normal equations and QR reject those fixtures; the SVD result is compared with the minimum-norm reference, not treated as recovery of the generating vector.

## Practical decision

For an unregularized dense least-squares workflow, compare QR or an SVD-based solve before accepting normal equations. Record the rank threshold and inspect coefficients as well as residuals. For ill-conditioned data, investigate scaling, measurement noise, identifiability, and an explicitly chosen regularization model; changing the solver alone is insufficient.

## Limits

These small synthetic matrices do not cover sparse, streaming, constrained, or distributed solvers. Three seeds describe the chosen constructions, not a population confidence interval. NumPy may promote linalg internals to double and cast outputs back; this is an input-output workflow comparison, not a pure-precision kernel benchmark. Float64 diagnostics and the stored float64 reference introduce a display floor near machine precision. Eighty-digit reference precision was checked against 110 digits on representative difficult cases, not proved sufficient for arbitrary inputs. Single-call timings include rank diagnostics and are not a speed benchmark.
