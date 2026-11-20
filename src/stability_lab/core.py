"""Separate solver arithmetic, input perturbation, and identifiability.

The reference is high-precision QR applied to the exact binary values in the
rounded NumPy input. It is not the generating coefficient vector.
"""
from dataclasses import dataclass
import numpy as np
import mpmath as mp

METHODS = ('normal_equations', 'qr', 'svd')


@dataclass(frozen=True)
class Problem:
    a: np.ndarray
    b: np.ndarray
    truth: np.ndarray
    target_singular_values: np.ndarray
    scenario: str


@dataclass(frozen=True)
class Solution:
    x: np.ndarray | None
    status: str
    effective_rank: int
    cutoff_ratio: float


def make_problem(seed: int, condition: float, noise: float = 0.0,
                 rows: int = 48, columns: int = 10,
                 rank_deficient: bool = False) -> Problem:
    if rows < columns or columns < 2 or condition < 1 or noise < 0:
        raise ValueError('Require rows >= columns >= 2, condition >= 1 and noise >= 0')
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.normal(size=(rows, columns)))
    v, _ = np.linalg.qr(rng.normal(size=(columns, columns)))
    spectrum = np.geomspace(1., 1./condition, columns)
    a = (u * spectrum) @ v.T
    if rank_deficient:
        # This exact duplicate remains identical after either dtype conversion.
        a[:, -1] = a[:, 0]
    truth = rng.normal(size=columns)
    truth /= np.linalg.norm(truth)
    b = a @ truth
    disturbance = rng.normal(size=rows)
    disturbance *= noise * np.linalg.norm(b) / np.linalg.norm(disturbance)
    b = b + disturbance
    return Problem(a, b, truth, spectrum, 'duplicate_column' if rank_deficient else 'full_rank')


def rounded(problem: Problem, dtype: str):
    if dtype not in ('float32', 'float64'):
        raise ValueError('Only float32 and float64 are studied')
    return problem.a.astype(dtype), problem.b.astype(dtype)


def numerical_rank64(a):
    singular = np.linalg.svd(a.astype(np.float64), compute_uv=False)
    cutoff = max(a.shape) * np.finfo(np.float64).eps * singular[0]
    return int(np.count_nonzero(singular > cutoff))


def solve(a: np.ndarray, b: np.ndarray, method: str) -> Solution:
    """Use the requested input/output array dtype, with a declared rank policy.

    NumPy linalg internals may promote precision. This compares concrete NumPy
    workflows, not pure float32 versus float64 hardware arithmetic kernels.

    Normal equations and unpivoted QR reject a matrix already numerically
    rank-deficient under a float64 diagnostic. SVD additionally uses an explicit
    dtype-dependent cutoff and reports truncation instead of hiding it.
    """
    if method not in METHODS:
        raise ValueError('Unknown method')
    if a.ndim != 2 or b.shape != (a.shape[0],) or a.shape[0] < a.shape[1]:
        raise ValueError('Expected a tall matrix and compatible vector')
    if a.dtype not in (np.dtype('float32'), np.dtype('float64')) or b.dtype != a.dtype:
        raise TypeError('A and b must have matching float32 or float64 dtype')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Inputs must be finite')
    rank = numerical_rank64(a)
    cutoff = float(max(a.shape) * np.finfo(a.dtype).eps)
    if method != 'svd' and rank < a.shape[1]:
        return Solution(None, 'rank_deficient_rejected', rank, cutoff)
    try:
        if method == 'normal_equations':
            x = np.linalg.solve(a.T @ a, a.T @ b)
        elif method == 'qr':
            q, r = np.linalg.qr(a, mode='reduced')
            x = np.linalg.solve(r, q.T @ b)
        else:
            # Explicit SVD exposes the exact cutoff and retained rank.
            u, singular, vt = np.linalg.svd(a, full_matrices=False)
            keep = singular > cutoff * singular[0]
            rank = int(keep.sum())
            projected = u.T @ b
            weighted = np.zeros_like(projected)
            weighted[keep] = projected[keep] / singular[keep]
            x = vt.T @ weighted
        if not np.isfinite(x).all():
            return Solution(None, 'nonfinite_result', rank, cutoff)
        status = 'truncated' if method == 'svd' and rank < a.shape[1] else 'ok'
        return Solution(x, status, rank, cutoff)
    except np.linalg.LinAlgError:
        return Solution(None, 'solver_failure', rank, cutoff)


def exact_mpf(value):
    numerator, denominator = float(value).as_integer_ratio()
    return mp.mpf(numerator) / denominator


def reference_solution(a, b, duplicate_column=False, digits=80):
    """Reference from exact binary inputs, using arbitrary-precision QR.

    The duplicate-column fixture has a known null direction. Solve the reduced
    independent matrix, then split the duplicated column's coefficient equally
    to obtain its unique minimum-norm representative.
    """
    with mp.workdps(digits):
        if duplicate_column and not np.array_equal(a[:, 0], a[:, -1]):
            raise ValueError('Duplicate-column reference requires exact repeated columns')
        reduced = a[:, :-1] if duplicate_column else a
        am = mp.matrix([[exact_mpf(v) for v in row] for row in reduced])
        bm = mp.matrix([exact_mpf(v) for v in b])
        x, _ = mp.qr_solve(am, bm)
        if duplicate_column:
            x = mp.matrix([x[0]/2] + list(x)[1:] + [x[0]/2])
        return np.array([float(value) for value in x], dtype=np.float64)


def metrics(a, b, x, reference, truth):
    """Diagnostic norms use float64 on the same stored input arrays.

    Reference gap includes deliberate SVD truncation, when that status is set.
    Recovery gap is a different quantity: it includes noise and input rounding.
    """
    af, bf, xf = a.astype(np.float64), b.astype(np.float64), x.astype(np.float64)
    residual = np.linalg.norm(af @ xf - bf)
    return {
        'relative_residual': float(residual / np.linalg.norm(bf)),
        'reference_gap': float(np.linalg.norm(xf-reference)/np.linalg.norm(reference)),
        'recovery_gap': float(np.linalg.norm(xf-truth)/np.linalg.norm(truth)),
        'reference_recovery_gap': float(np.linalg.norm(reference-truth)/np.linalg.norm(truth)),
        'coefficient_norm': float(np.linalg.norm(xf)),
    }
