# Methods and metric definitions

## Controlled construction

For each seed, generate two orthogonal factors by reduced QR of standard-normal matrices. Form `A = U @ diag(s) @ V.T`, where `s` is geometrically spaced from 1 to `1 / target_condition`. All construction is in float64. Generate a unit-norm coefficient vector `x_true`, then `b = A @ x_true + e`, with `||e|| / ||A @ x_true||` equal to the configured relative noise level. Thus even the zero-added-noise float64 case includes finite-precision construction arithmetic.

The full run uses seeds 811, 812, 813; shape 48 × 10; target conditions 1, 10², 10⁴, 10⁶, 10⁸, 10¹⁰, 10¹²; added noise 0 or 10⁻⁸; and float32/float64 input-output arrays. This gives 84 rounded input cases. Six exact duplicate-column cases provide a rank-deficiency check. The **target condition** describes the initial construction. `observed_condition64` is recomputed from the stored matrix in float64; the two can diverge substantially after float32 conversion.

## Three concrete NumPy workflows

| Method | Computation | Rank policy |
|---|---|---|
| Normal equations | `solve(A.T @ A, A.T @ b)` | Reject when a float64 SVD diagnostic says rank < n |
| QR | Reduced QR, then `solve(R, Q.T @ b)` | Same rejection diagnostic |
| SVD | `V @ diag(1/s_kept) @ U.T @ b` | Retain s > max(m,n) × eps(input dtype) × s_max |

**Precision scope matters.** These are float32/float64 *input-output workflows*, not pure FP32/FP64 hardware kernels. In the installed NumPy 2.3.5 implementation, `_commonType` promotes QR/SVD/solve work to double and casts the output back to the input result dtype. Matrix products and the explicit SVD reconstruction operate on their array dtypes. Inspection of [NumPy 2.3.5 source](https://github.com/numpy/numpy/blob/v2.3.5/numpy/linalg/_linalg.py) substantiates this distinction. No conclusion about a native single-precision LAPACK driver or GPU kernel follows from these results.

An `ok` status means a finite result was returned; it does **not** certify accuracy. `truncated` indicates a deliberate reduction of SVD rank. `rank_deficient_rejected`, `solver_failure`, and `nonfinite_result` produce no coefficients and no misleading zero-error metric. The float64 rank diagnostic is itself a numerical threshold, not a symbolic proof of exact rank.

In exact arithmetic, the nonzero singular values of AᵀA are the squares of A's singular values. Therefore κ₂(AᵀA) = κ₂(A)² for full column rank. This explains why forming normal equations can magnify numerical difficulty; it is not an empirical fit to the plots.

## Same-input high-precision reference

Every stored binary float is converted through `as_integer_ratio` into an exact mpmath value. `mpmath.qr_solve` then solves that specific rounded problem at 80 decimal digits. The reference is stored as float64 for diagnostic norms. Representative difficult cases are recomputed at 110 digits in the test suite and must round to identical float64 coefficients.

The duplicate-column case replaces the last column with an exact copy of the first. The reduced matrix excludes the last column. Its QR solution is expanded by splitting the first coefficient equally between the duplicated pair, yielding the minimum-norm representative. This construction avoids pretending that the generating coefficients are identifiable in a singular problem. The reference routine asserts that the columns really are identical.

## Distinguish the errors

Let x̂ be the method result, x_ref the reference for the **same stored A,b**, and x_true the original generating vector.

| CSV field | Formula | Interpretation |
|---|---|---|
| relative_residual | ‖A x̂ − b‖₂ / ‖b‖₂ | Fit to the rounded input, evaluated in float64 |
| reference_gap | ‖x̂ − x_ref‖₂ / ‖x_ref‖₂ | Solver agreement; includes intentional model change if SVD truncated |
| recovery_gap | ‖x̂ − x_true‖₂ / ‖x_true‖₂ | End-to-end recovery including noise, rounding and solver behavior |
| reference_recovery_gap | ‖x_ref − x_true‖₂ / ‖x_true‖₂ | Effect of the changed input problem even with a high-precision solve |
| a_rounding_relative | ‖A_stored − A_constructed‖F / ‖A_constructed‖F | Relative matrix representation change |

Do not describe `reference_gap` for a truncated SVD as pure arithmetic error. Do not interpret `recovery_gap` as a uniquely meaningful accuracy measure for duplicate-column cases. These caveats apply to individual raw rows as well as charts.

## Reproducibility and limitations

The repository stores every A, b, x_true and x_ref in compressed NPZ files, a SHA-256 manifest, returned coefficients, raw metrics, environment versions, and generated summaries. Tests independently recompute all published metrics from those stored arrays. The full run has only three seeds per configuration; median curves are descriptive, not confidence intervals or a universal error bound.

NumPy/BLAS implementation, platform and version may alter last-bit values. The checked-in run is preserved. New runs use a separate output folder and refuse to overwrite existing measurements. Single-call nanosecond timings include the rank diagnostic and are not randomized/repeated performance trials; they are retained only as execution trace information.

This study does not evaluate sparse matrices, regularization selection, constrained least squares, covariance estimation, pivoted QR, native precision-specific LAPACK drivers, or distributed/GPU implementations. Those are possible extensions, not claimed capabilities.

## Primary references

- [NumPy QR documentation](https://numpy.org/doc/2.3/reference/generated/numpy.linalg.qr.html) — reduced factorization and returned factors.
- [NumPy SVD documentation](https://numpy.org/doc/2.3/reference/generated/numpy.linalg.svd.html) — singular vectors and values.
- [NumPy least-squares documentation](https://numpy.org/doc/2.3/reference/generated/numpy.linalg.lstsq.html) — minimum-norm and rank-cutoff conventions; the experiment uses an explicit SVD instead of hiding the threshold in a default call.
- [mpmath matrix documentation](https://mpmath.org/doc/current/matrices.html) — arbitrary-precision QR solving.
