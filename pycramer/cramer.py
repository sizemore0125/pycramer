# pycamer
# Copyright (C) 2025  Logan Sizemore

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from numbers import Integral
from typing import Callable, Optional, Sequence, Union

import numpy as np

from ._lookup import (
    calculate_lookup_matrix,
    calculate_lookup_matrix_with_kernel,
    cramer_statistic as _cramer_statistic_native,
    cramer_statistic_batch as _cramer_statistic_batch_native,
    cramer_bootstrap_random as _cramer_bootstrap_random,
    cramer_statistic_from_data as _cramer_statistic_from_data,
)

Kernel = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class KernelSpec:
    label: str
    fn: Kernel
    builtin_id: Optional[int]


@dataclass
class CramerTestResult:
    """Container for Cramér test outcomes."""

    method: str
    d: int
    m: int
    n: int
    statistic: float
    conf_level: float
    crit_value: float
    p_value: float
    result: int
    sim: str
    replicates: int
    hypdist_x: Optional[np.ndarray]
    hypdist_Fx: Optional[np.ndarray]
    eigenvalues: Optional[np.ndarray]

    def __str__(self) -> str:
        header = f"\n{self.d}-dimensional {self.method}\n"
        sizes = f"\tx-sample: {self.m} values        y-sample: {self.n} values\n"
        if self.crit_value > 0:
            decision = "ACCEPTED" if self.result == 0 else "REJECTED"
            details = (
                f"critical value for confidence level {self.conf_level * 100:.1f}% : {self.crit_value}\n"
                f"observed statistic {self.statistic}, so that\n"
                f'\t hypothesis ("x is distributed as y") is {decision}.\n'
                f"estimated p-value = {self.p_value}\n"
            )
            source = (
                f"\t[result based on {self.replicates} {self.sim} bootstrap-replicates]\n"
                if self.sim != "eigenvalue"
                else "\t[result based on eigenvalue decomposition and inverse fft]\n"
            )
            return header + sizes + details + source
        return header + sizes + f"observed statistic {self.statistic}\n"


def phi_cramer(x: np.ndarray) -> np.ndarray:
    """Compute the default Cramér kernel."""
    return 0.5 * np.sqrt(x)


def phi_bahr(x: np.ndarray) -> np.ndarray:
    """Compute the Bahr kernel transformation."""
    return 1.0 - np.exp(-0.5 * x)


def phi_log(x: np.ndarray) -> np.ndarray:
    """Compute the logarithmic kernel transformation."""
    return np.log1p(x)


def phi_frac_a(x: np.ndarray) -> np.ndarray:
    """Compute the fractional-A kernel transformation."""
    return 1.0 - 1.0 / (1.0 + x)


def phi_frac_b(x: np.ndarray) -> np.ndarray:
    """Compute the fractional-B kernel transformation."""
    return 1.0 - 1.0 / np.square(1.0 + x)


KERNELS: dict[str, Kernel] = {
    "phiCramer": phi_cramer,
    "phiBahr": phi_bahr,
    "phiLog": phi_log,
    "phiFracA": phi_frac_a,
    "phiFracB": phi_frac_b,
}

_BUILTIN_KERNEL_IDS: dict[str, int] = {
    "phiCramer": 0,
    "phiBahr": 1,
    "phiLog": 2,
    "phiFracA": 3,
    "phiFracB": 4,
}


RandomStateLike = Union[None, Integral, np.random.RandomState, np.random.Generator]
RNG = Union[np.random.RandomState, np.random.Generator]


def cramer_test(
    x: Union[Sequence[float], np.ndarray],
    y: Union[Sequence[float], np.ndarray],
    conf_level: float = 0.95,
    replicates: int = 1000,
    sim: str = "ordinary",
    just_statistic: bool = False,
    kernel: Union[str, Kernel] = "phiCramer",
    max_m: int = 1 << 14,
    K: int = 160,
    random_state: RandomStateLike = None,
    workers: Optional[int] = None,
    resamples: Optional[Sequence[Sequence[int]]] = None,
) -> CramerTestResult:
    """Perform the Cramér two-sample test.

    Args:
        x (Union[Sequence[float], np.ndarray]): First sample, 1D or 2D array-like.
        y (Union[Sequence[float], np.ndarray]): Second sample, matching shape to `x`.
        conf_level (float, optional): Confidence level for the rejection threshold. Defaults to 0.95.
        replicates (int, optional): Number of bootstrap draws when `resamples` is not provided. Defaults to 1000.
        sim (str, optional): Resampling strategy (`ordinary`, `permutation`, or `eigenvalue`). Defaults to "ordinary".
        just_statistic (bool, optional): If True, only compute the test statistic. Defaults to False.
        kernel (Union[str, Kernel], optional): Kernel identifier or callable. Defaults to "phiCramer".
        max_m (int, optional): Upper bound on FFT grid size for eigenvalue mode. Defaults to 1 << 14.
        K (int, optional): Integral upper limit used by the FFT inversion. Defaults to 160.
        random_state (RandomStateLike, optional): RNG seed or object for reproducibility. Defaults to None.
        workers (Optional[int], optional): Maximum worker threads for native bootstrap acceleration. Defaults to None (auto).
        resamples (Optional[Sequence[Sequence[int]]], optional): Explicit bootstrap index matrix. Defaults to None.

    Returns:
        CramerTestResult: Structured result containing statistic, critical value, p-value, and metadata.
    """
    x_arr = _coerce_sample(x)
    y_arr = _coerce_sample(y)

    if x_arr.shape[1] != y_arr.shape[1]:
        raise ValueError("x and y must have the same number of columns.")

    m, n = x_arr.shape[0], y_arr.shape[0]
    d = x_arr.shape[1]
    data = np.vstack([x_arr, y_arr])
    data_c = np.asarray(data, dtype=float, order="C")
    N = m + n

    kernel_spec = _resolve_kernel_spec(kernel)

    method = (
        f"nonparametric Cramer-Test with kernel "
        f"{kernel_spec.label}\n"
        "(on equality of two distributions)"
    )

    worker_count = _validate_workers(workers)

    statistic_fn = (
        _cramer_statistic_numpy if resamples is not None else _cramer_statistic
    )

    if just_statistic and resamples is None and kernel_spec.builtin_id is not None:
        statistic0 = float(
            _cramer_statistic_from_data(data_c, int(m), int(kernel_spec.builtin_id))
        )
        return CramerTestResult(
            method=method,
            d=d,
            m=m,
            n=n,
            statistic=statistic0,
            conf_level=conf_level,
            crit_value=0.0,
            p_value=0.0,
            result=0,
            sim=sim,
            replicates=0,
            hypdist_x=None,
            hypdist_Fx=None,
            eigenvalues=None,
        )

    lookup = _build_lookup_table(data_c, kernel_spec)
    base_indices = np.arange(N, dtype=int)
    statistic0 = statistic_fn(lookup, base_indices, m)

    if just_statistic:
        return CramerTestResult(
            method=method,
            d=d,
            m=m,
            n=n,
            statistic=statistic0,
            conf_level=conf_level,
            crit_value=0.0,
            p_value=0.0,
            result=0,
            sim=sim,
            replicates=0,
            hypdist_x=None,
            hypdist_Fx=None,
            eigenvalues=None,
        )

    if sim == "eigenvalue":
        if resamples is not None:
            raise ValueError("resamples should be None when sim is 'eigenvalue'.")
        eigenvalues, dist_x, dist_Fx, crit_value = _eigenvalue_path(
            lookup, conf_level, max_m, K
        )
        idx_limit = max(1, (3 * dist_x.size) // 4)
        target_idx = int(np.argmin(np.abs(dist_x[:idx_limit] - statistic0)))
        p_value = 1.0 - dist_Fx[target_idx]
        result = int(statistic0 > crit_value)
        return CramerTestResult(
            method=method,
            d=d,
            m=m,
            n=n,
            statistic=statistic0,
            conf_level=conf_level,
            crit_value=float(crit_value),
            p_value=float(p_value),
            result=result,
            sim=sim,
            replicates=replicates,
            hypdist_x=dist_x,
            hypdist_Fx=dist_Fx,
            eigenvalues=eigenvalues,
        )

    if sim not in {"ordinary", "permutation"}:
        raise ValueError("sim must be 'ordinary', 'permutation', or 'eigenvalue'.")

    stats = _bootstrap_statistics(
        lookup=lookup,
        m=m,
        N=N,
        replicates=replicates,
        sim=sim,
        random_state=random_state,
        resamples=resamples,
        workers=worker_count,
        statistic_fn=statistic_fn,
    )

    crit_value, p_value, hypdist_x, hypdist_Fx, decision = _evaluate_bootstrap(
        statistic0,
        stats,
        conf_level,
        use_tolerance=statistic_fn is _cramer_statistic,
    )

    return CramerTestResult(
        method=method,
        d=d,
        m=m,
        n=n,
        statistic=float(statistic0),
        conf_level=conf_level,
        crit_value=float(crit_value),
        p_value=float(p_value),
        result=decision,
        sim=sim,
        replicates=stats.size,
        hypdist_x=hypdist_x,
        hypdist_Fx=hypdist_Fx,
        eigenvalues=None,
    )


def _coerce_sample(sample: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Convert user input into a 2D floating-point array.

    Args:
        sample (Union[Sequence[float], np.ndarray]): Raw sample values.

    Returns:
        np.ndarray: Two-dimensional array with observations in rows.
    """
    arr = np.asarray(sample, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("Samples must be 1D or 2D arrays.")
    return arr


def _resolve_kernel_spec(kernel: Union[str, Kernel]) -> KernelSpec:
    """Resolve a kernel specification into a callable plus metadata."""
    if isinstance(kernel, str):
        try:
            fn = KERNELS[kernel]
        except KeyError as exc:
            raise ValueError(f"Unknown kernel '{kernel}'.") from exc
        return KernelSpec(
            label=kernel, fn=fn, builtin_id=_BUILTIN_KERNEL_IDS.get(kernel)
        )
    if callable(kernel):
        label = getattr(kernel, "__name__", "<lambda>")
        return KernelSpec(label=label, fn=kernel, builtin_id=None)
    raise TypeError("kernel must be a string or a callable.")


def _build_lookup_table(data: np.ndarray, kernel_spec: KernelSpec) -> np.ndarray:
    """Construct the pairwise lookup matrix, applying kernels in one pass when possible."""
    if kernel_spec.builtin_id is not None:
        return calculate_lookup_matrix_with_kernel(data, kernel_spec.builtin_id)
    lookup = calculate_lookup_matrix(data)
    return kernel_spec.fn(lookup)


def _cramer_statistic(
    lookup: np.ndarray,
    indices: np.ndarray,
    m: int,
) -> float:
    """Evaluate the Cramér statistic for a particular resample.

    Args:
        lookup (np.ndarray): Kernel-transformed distance matrix.
        indices (np.ndarray): Index vector partitioned into X then Y observations.
        m (int): Number of X observations.

    Returns:
        float: Value of the Cramér two-sample statistic.
    """
    indices_arr = np.asarray(indices, dtype=np.intp)
    if indices_arr.ndim != 1:
        raise ValueError("indices must be a one-dimensional array.")
    if indices_arr.size != lookup.shape[0]:
        raise ValueError("indices length must equal lookup dimension.")
    return float(_cramer_statistic_native(lookup, indices_arr, int(m)))


def _cramer_statistic_numpy(
    lookup: np.ndarray,
    indices: np.ndarray,
    m: int,
) -> float:
    """Python fallback for statistic evaluation, used when exact parity is required."""
    indices_arr = np.asarray(indices, dtype=np.intp)
    if indices_arr.ndim != 1:
        raise ValueError("indices must be a one-dimensional array.")
    if indices_arr.size != lookup.shape[0]:
        raise ValueError("indices length must equal lookup dimension.")
    x_idx = indices_arr[:m]
    y_idx = indices_arr[m:]
    term_xy = lookup[np.ix_(x_idx, y_idx)].sum()
    term_xx = lookup[np.ix_(x_idx, x_idx)].sum()
    term_yy = lookup[np.ix_(y_idx, y_idx)].sum()
    m_f = float(m)
    n_f = float(indices_arr.size - m)
    return (m_f * n_f / (m_f + n_f)) * (
        2.0 * term_xy / (m_f * n_f) - term_xx / (m_f * m_f) - term_yy / (n_f * n_f)
    )


def _bootstrap_statistics(
    lookup: np.ndarray,
    m: int,
    N: int,
    replicates: int,
    sim: str,
    random_state: RandomStateLike,
    resamples: Optional[Sequence[Sequence[int]]],
    workers: Optional[int],
    statistic_fn: Callable[[np.ndarray, np.ndarray, int], float],
) -> np.ndarray:
    """Generate bootstrap statistics using either supplied or random resamples."""
    if resamples is not None:
        return _bootstrap_from_resamples(resamples, lookup, m, N, statistic_fn)

    if replicates <= 0:
        raise ValueError("replicates must be positive when resamples are not provided.")

    seed_payload = _resolve_native_bootstrap_seed(random_state)
    if statistic_fn is _cramer_statistic and seed_payload is not None:
        native = _bootstrap_via_native(
            lookup=lookup,
            m=m,
            replicates=replicates,
            sim=sim,
            seed=seed_payload,
            workers=workers,
        )
        if native is not None:
            return native

    rng = _resolve_rng(random_state)
    return _bootstrap_via_python(
        lookup=lookup,
        m=m,
        N=N,
        replicates=replicates,
        sim=sim,
        rng=rng,
        statistic_fn=statistic_fn,
    )


def _bootstrap_from_resamples(
    resamples: Sequence[Sequence[int]],
    lookup: np.ndarray,
    m: int,
    N: int,
    statistic_fn: Callable[[np.ndarray, np.ndarray, int], float],
) -> np.ndarray:
    resample_array = np.asarray(resamples, dtype=np.intp, order="C")
    if resample_array.ndim != 2 or resample_array.shape[1] != N:
        raise ValueError("Each resample must have length m + n.")
    if statistic_fn is _cramer_statistic:
        return np.asarray(
            _cramer_statistic_batch_native(lookup, resample_array, int(m))
        )
    stats = np.empty(resample_array.shape[0], dtype=float)
    for idx, resample in enumerate(resample_array):
        stats[idx] = statistic_fn(lookup, resample, m)
    return stats


def _bootstrap_via_native(
    lookup: np.ndarray,
    m: int,
    replicates: int,
    sim: str,
    seed: int,
    workers: Optional[int],
) -> Optional[np.ndarray]:
    if sim not in {"ordinary", "permutation"}:
        return None
    sim_kind = 0 if sim == "ordinary" else 1
    worker_count = -1 if workers is None else int(workers)
    return np.asarray(
        _cramer_bootstrap_random(
            lookup,
            int(m),
            int(replicates),
            sim_kind,
            int(seed),
            worker_count,
        )
    )


def _bootstrap_via_python(
    lookup: np.ndarray,
    m: int,
    N: int,
    replicates: int,
    sim: str,
    rng: RNG,
    statistic_fn: Callable[[np.ndarray, np.ndarray, int], float],
) -> np.ndarray:
    stats = np.empty(replicates, dtype=float)
    for r in range(replicates):
        if sim == "ordinary":
            if hasattr(rng, "integers"):
                indices = rng.integers(0, N, size=N, dtype=np.intp)
            else:
                indices = rng.randint(0, N, size=N).astype(np.intp, copy=False)
        else:
            indices = np.asarray(rng.permutation(N), dtype=np.intp)
        stats[r] = statistic_fn(lookup, indices, m)
    return stats


def _resolve_rng(random_state: RandomStateLike) -> RNG:
    """Normalize ambiguous RNG specifications to a concrete generator.

    Args:
        random_state (RandomStateLike): Seed, Generator, RandomState, or None.

    Raises:
        TypeError: If the input cannot be converted to a supported RNG.

    Returns:
        RNG: Numpy random number generator.
    """
    if random_state is None:
        return np.random.default_rng()
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, np.random.RandomState):
        return random_state
    if isinstance(random_state, Integral):
        return np.random.RandomState(int(random_state))
    raise TypeError(
        "random_state must be None, an int, numpy.random.RandomState, or numpy.random.Generator."
    )


def _resolve_native_bootstrap_seed(random_state: RandomStateLike) -> Optional[int]:
    """Return a 64-bit seed for the native bootstrap, or None if fallback is required."""
    if random_state is None:
        seed_seq = np.random.SeedSequence()
        seed = int(seed_seq.generate_state(1, dtype=np.uint64)[0])
        return seed
    if isinstance(random_state, Integral):
        return int(random_state) & ((1 << 64) - 1)
    return None


def _validate_workers(workers: Optional[int]) -> Optional[int]:
    """Validate worker count input."""
    if workers is None:
        return None
    if not isinstance(workers, Integral):
        raise TypeError("workers must be an integer or None.")
    workers_int = int(workers)
    if workers_int <= 0:
        raise ValueError("workers must be positive when specified.")
    return workers_int


def _evaluate_bootstrap(
    statistic0: float,
    bootstrap_stats: np.ndarray,
    conf_level: float,
    *,
    use_tolerance: bool,
) -> tuple[float, float, np.ndarray, np.ndarray, int]:
    """Compute critical value, p-value, and empirical distribution from bootstrap draws.

    Args:
        statistic0 (float): Observed statistic for the original sample.
        bootstrap_stats (np.ndarray): Bootstrap statistic values.
        conf_level (float): Requested confidence level for the rejection threshold.

    Returns:
        tuple[float, float, np.ndarray, np.ndarray, int]: Critical value, p-value, sorted statistics, empirical CDF, and decision indicator.
    """
    bootstrap_stats = np.asarray(bootstrap_stats, dtype=float)
    sorted_stats = np.sort(bootstrap_stats)
    hypdist_x = sorted_stats.copy()
    hypdist_Fx = np.arange(1, sorted_stats.size + 1, dtype=float) / sorted_stats.size

    quantile_index = int(np.round(conf_level * sorted_stats.size))
    quantile_index = max(1, min(quantile_index, sorted_stats.size)) - 1
    crit_value = sorted_stats[quantile_index]

    if use_tolerance:
        eps = np.finfo(float).eps
        tol = max(1.0, abs(statistic0)) * eps * 32.0
        less_than = np.sum(bootstrap_stats < statistic0 - tol)
        equal_to = np.sum(np.abs(bootstrap_stats - statistic0) <= tol)
    else:
        less_than = np.sum(bootstrap_stats < statistic0)
        equal_to = np.sum(bootstrap_stats == statistic0)
    ties = equal_to + 1  # include statistic itself
    rank = less_than + (ties + 1) / 2.0
    p_value = 1.0 - rank / (bootstrap_stats.size + 1.0)

    decision = int(statistic0 > crit_value)
    return crit_value, p_value, hypdist_x, hypdist_Fx, decision


def _eigenvalue_path(
    lookup: np.ndarray,
    conf_level: float,
    max_m: int,
    K: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Solve the eigenvalue-based approximation to the limiting distribution.

    Args:
        lookup (np.ndarray): Kernel-transformed distance matrix.
        conf_level (float): Requested confidence level for the rejection threshold.
        max_m (int): Maximum FFT resolution for spectrum inversion.
        K (int): Integral limit, controlling frequency resolution.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, float]: Eigenvalues, distribution grid, CDF values, and estimated quantile.
    """
    N = lookup.shape[0]
    C1 = lookup.sum(axis=1) / N
    C2 = lookup.sum() / (N * N)
    B = (C1[:, None] + C1[None, :] - C2 - lookup) / N

    eigenvalues = np.linalg.eigvalsh(B)
    eigenvalues = np.real_if_close(eigenvalues)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]

    dist_x, dist_Fx, quantile = _cramer_kritwert_fft(eigenvalues, conf_level, max_m, K)
    return eigenvalues, dist_x, dist_Fx, quantile


def _cramer_kritwert_fft(
    lambda_sq: np.ndarray,
    conf_level: float,
    max_m: int,
    K: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Invert the characteristic function via FFT to obtain the limiting distribution.

    Args:
        lambda_sq (np.ndarray): Eigenvalues of the centered kernel matrix.
        conf_level (float): Requested confidence level.
        max_m (int): Upper cap on FFT grid size.
        K (int): Integral bound used in the discretization.

    Returns:
        tuple[np.ndarray, np.ndarray, float]: Grid points, CDF values, and quantile estimate.
    """
    lambda_sq = np.asarray(lambda_sq, dtype=float)
    if lambda_sq.size == 0:
        raise ValueError("Eigenvalue array is empty; cannot evaluate eigenvalue path.")

    M = 1 << 11
    threshold = 2.0 * np.sum(lambda_sq) + lambda_sq[0]
    while 150.0 * math.pi * M / (K * K) < threshold and M < max_m:
        M <<= 1
    M = min(M, max_m)
    good_limit = 150.0 * math.pi * M / (K * K)

    t = np.arange(M, dtype=float) * K / M
    t[0] = 1.0  # avoid division by zero
    x = np.arange(M, dtype=float) * (2.0 * math.pi / K)

    char = _cramer_characteristic_function(lambda_sq, t)
    h = char / t
    h *= np.exp(-0.0j * t)
    h[0] = 1j * np.sum(lambda_sq)

    fft_vals = np.fft.fft(h)
    Fx = (
        0.5
        - np.imag(K / (M * math.pi) * fft_vals)
        + K / (2.0 * M * math.pi) * (np.sum(lambda_sq) + x)
    )

    half = M // 2
    if Fx[half - 1] < conf_level:
        warnings.warn(
            "Quantile calculation discrepancy. Try to increase K!", RuntimeWarning
        )

    idx = int(np.argmin(np.abs(Fx[:half] - conf_level)))
    if Fx[idx] > conf_level:
        idx = max(idx - 1, 0)
    idx_next = min(idx + 1, Fx.size - 1)
    if idx_next == idx:
        quantile = x[idx]
    else:
        quantile = x[idx] + (conf_level - Fx[idx]) * (x[idx_next] - x[idx]) / (
            Fx[idx_next] - Fx[idx]
        )

    if quantile > good_limit:
        warnings.warn(
            "Quantile beyond good approximation limit. Try to increase maxM or decrease K!",
            RuntimeWarning,
        )

    return x, Fx.real, float(quantile)


def _cramer_characteristic_function(lambda_sq: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate the characteristic function of the limiting distribution.

    Args:
        lambda_sq (np.ndarray): Eigenvalues of the kernel matrix.
        t (np.ndarray): Evaluation points for the characteristic function.

    Returns:
        np.ndarray: Complex-valued characteristic function evaluated at `t`.
    """
    lambda_sq = np.asarray(lambda_sq, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float)
    z = -0.5 * np.log(1.0 - 2j * lambda_sq[:, None] * t[None, :])
    return np.exp(np.sum(z, axis=0))
