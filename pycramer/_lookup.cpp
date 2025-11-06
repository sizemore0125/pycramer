#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

namespace {

struct LookupAccessor {
    const char* data;
    npy_intp stride_row;
    npy_intp stride_col;
    npy_intp size;
};

inline double cramer_statistic_impl(const LookupAccessor& lookup,
                                    const npy_intp* indices,
                                    npy_intp m) {
    const npy_intp N = lookup.size;
    const npy_intp n = N - m;
    if (m <= 0 || n <= 0) {
        PyErr_SetString(PyExc_ValueError, "m must be between 1 and N - 1.");
        return std::numeric_limits<double>::quiet_NaN();
    }

    long double term_xy = 0.0L;
    long double term_xx = 0.0L;
    long double term_yy = 0.0L;

    for (npy_intp i = 0; i < m; ++i) {
        const npy_intp row_idx = indices[i];
        const char* row_ptr = lookup.data + row_idx * lookup.stride_row;
        for (npy_intp j = m; j < N; ++j) {
            const npy_intp col_idx = indices[j];
            term_xy += static_cast<long double>(
                *reinterpret_cast<const double*>(row_ptr + col_idx * lookup.stride_col));
        }
        for (npy_intp j = 0; j < m; ++j) {
            const npy_intp col_idx = indices[j];
            term_xx += static_cast<long double>(
                *reinterpret_cast<const double*>(row_ptr + col_idx * lookup.stride_col));
        }
    }

    for (npy_intp i = m; i < N; ++i) {
        const npy_intp row_idx = indices[i];
        const char* row_ptr = lookup.data + row_idx * lookup.stride_row;
        for (npy_intp j = m; j < N; ++j) {
            const npy_intp col_idx = indices[j];
            term_yy += static_cast<long double>(
                *reinterpret_cast<const double*>(row_ptr + col_idx * lookup.stride_col));
        }
    }

    const long double m_f = static_cast<long double>(m);
    const long double n_f = static_cast<long double>(n);
    const long double prefactor = (m_f * n_f) / (m_f + n_f);
    const long double result =
        prefactor *
        ((2.0L * term_xy) / (m_f * n_f) - term_xx / (m_f * m_f) - term_yy / (n_f * n_f));
    return static_cast<double>(result);
}

enum class KernelId : int {
    kPhiCramer = 0,
    kPhiBahr = 1,
    kPhiLog = 2,
    kPhiFracA = 3,
    kPhiFracB = 4,
};

enum class SimKind : int {
    kOrdinary = 0,
    kPermutation = 1,
};

inline bool apply_kernel_value(KernelId id, double& value) {
    const double half = 0.5;
    if (value < 0.0) {
        value = 0.0;
    }
    switch (id) {
        case KernelId::kPhiCramer:
            value = half * std::sqrt(value);
            return true;
        case KernelId::kPhiBahr:
            value = 1.0 - std::exp(-half * value);
            return true;
        case KernelId::kPhiLog:
            value = std::log1p(value);
            return true;
        case KernelId::kPhiFracA:
            value = 1.0 - 1.0 / (1.0 + value);
            return true;
        case KernelId::kPhiFracB: {
            const double denom = 1.0 + value;
            value = 1.0 - 1.0 / (denom * denom);
            return true;
        }
        default:
            return false;
    }
}

PyObject* calculate_lookup_matrix(PyObject*, PyObject* args) {
    PyObject* data_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &data_obj)) {
        return nullptr;
    }

    PyArrayObject* data = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(data_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (data == nullptr) {
        return nullptr;
    }

    const int ndim = PyArray_NDIM(data);
    if (ndim != 2) {
        Py_DECREF(data);
        PyErr_SetString(PyExc_ValueError, "data must be a 2D array.");
        return nullptr;
    }

    const npy_intp* dims = PyArray_DIMS(data);
    const npy_intp length = dims[0];
    const npy_intp cols = dims[1];

    npy_intp out_dims[2] = {length, length};
    PyArrayObject* lookup = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(2, out_dims, NPY_DOUBLE));
    if (lookup == nullptr) {
        Py_DECREF(data);
        return nullptr;
    }

    auto* data_ptr = static_cast<double*>(PyArray_DATA(data));
    auto* lookup_ptr = static_cast<double*>(PyArray_DATA(lookup));
    const npy_intp row_stride = PyArray_STRIDES(data)[0] /
                                static_cast<npy_intp>(sizeof(double));
    const npy_intp col_stride = PyArray_STRIDES(data)[1] /
                                static_cast<npy_intp>(sizeof(double));

    auto compute_block = [&](npy_intp start, npy_intp end) {
        for (npy_intp i = start; i < end; ++i) {
            double* row_i = lookup_ptr + i * length;
            row_i[i] = 0.0;
            const double* xi = data_ptr + i * row_stride;
            for (npy_intp j = 0; j < i; ++j) {
                const double* xj = data_ptr + j * row_stride;
                double sum_sq_diff = 0.0;
                for (npy_intp k = 0; k < cols; ++k) {
                    const double diff = xi[k * col_stride] - xj[k * col_stride];
                    sum_sq_diff += diff * diff;
                }
                row_i[j] = sum_sq_diff;
                lookup_ptr[j * length + i] = sum_sq_diff;
            }
        }
    };

    int worker_count = static_cast<int>(std::thread::hardware_concurrency());
    if (worker_count <= 1 || length < 256) {
        compute_block(0, length);
    } else {
        if (worker_count > static_cast<int>(length)) {
            worker_count = static_cast<int>(length);
        }
        std::vector<std::thread> threads;
        threads.reserve(worker_count);
        const npy_intp chunk = (length + worker_count - 1) / worker_count;
        Py_BEGIN_ALLOW_THREADS;
        for (int w = 0; w < worker_count; ++w) {
            const npy_intp start = w * chunk;
            const npy_intp end = std::min(length, start + chunk);
            if (start >= end) {
                continue;
            }
            threads.emplace_back(compute_block, start, end);
        }
        for (auto& th : threads) {
            if (th.joinable()) {
                th.join();
            }
        }
        Py_END_ALLOW_THREADS;
    }

    Py_DECREF(data);
    return reinterpret_cast<PyObject*>(lookup);
}

PyObject* calculate_lookup_matrix_with_kernel(PyObject*, PyObject* args) {
    PyObject* data_obj = nullptr;
    int kernel_id = 0;
    if (!PyArg_ParseTuple(args, "Oi", &data_obj, &kernel_id)) {
        return nullptr;
    }

    KernelId id = static_cast<KernelId>(kernel_id);

    PyArrayObject* data = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(data_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (data == nullptr) {
        return nullptr;
    }

    const int ndim = PyArray_NDIM(data);
    if (ndim != 2) {
        Py_DECREF(data);
        PyErr_SetString(PyExc_ValueError, "data must be a 2D array.");
        return nullptr;
    }

    const npy_intp* dims = PyArray_DIMS(data);
    const npy_intp length = dims[0];
    const npy_intp cols = dims[1];

    npy_intp out_dims[2] = {length, length};
    PyArrayObject* lookup = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(2, out_dims, NPY_DOUBLE));
    if (lookup == nullptr) {
        Py_DECREF(data);
        return nullptr;
    }

    auto* data_ptr = static_cast<double*>(PyArray_DATA(data));
    auto* lookup_ptr = static_cast<double*>(PyArray_DATA(lookup));
    const npy_intp row_stride = PyArray_STRIDES(data)[0] /
                                static_cast<npy_intp>(sizeof(double));
    const npy_intp col_stride = PyArray_STRIDES(data)[1] /
                                static_cast<npy_intp>(sizeof(double));

    auto compute_block = [&](npy_intp start, npy_intp end) {
        for (npy_intp i = start; i < end; ++i) {
            double* row_i = lookup_ptr + i * length;
            row_i[i] = 0.0;
            const double* xi = data_ptr + i * row_stride;
            for (npy_intp j = 0; j < i; ++j) {
                const double* xj = data_ptr + j * row_stride;
                double sum_sq_diff = 0.0;
                for (npy_intp k = 0; k < cols; ++k) {
                    const double diff = xi[k * col_stride] - xj[k * col_stride];
                    sum_sq_diff += diff * diff;
                }
                if (!apply_kernel_value(id, sum_sq_diff)) {
                    return false;
                }
                row_i[j] = sum_sq_diff;
                lookup_ptr[j * length + i] = sum_sq_diff;
            }
        }
        return true;
    };

    std::atomic<bool> ok(true);
    int worker_count = static_cast<int>(std::thread::hardware_concurrency());
    if (worker_count <= 1 || length < 256) {
        ok.store(compute_block(0, length), std::memory_order_relaxed);
    } else {
        if (worker_count > static_cast<int>(length)) {
            worker_count = static_cast<int>(length);
        }
        std::vector<std::thread> threads;
        threads.reserve(worker_count);
        const npy_intp chunk = (length + worker_count - 1) / worker_count;
        Py_BEGIN_ALLOW_THREADS;
        for (int w = 0; w < worker_count; ++w) {
            const npy_intp start = w * chunk;
            const npy_intp end = std::min(length, start + chunk);
            if (start >= end) {
                continue;
            }
            threads.emplace_back([&, start, end]() {
                if (!compute_block(start, end)) {
                    ok.store(false, std::memory_order_relaxed);
                }
            });
        }
        for (auto& th : threads) {
            if (th.joinable()) {
                th.join();
            }
        }
        Py_END_ALLOW_THREADS;
    }

    Py_DECREF(data);
    if (!ok.load(std::memory_order_relaxed)) {
        Py_DECREF(lookup);
        PyErr_SetString(PyExc_ValueError, "Unknown kernel id.");
        return nullptr;
    }
    return reinterpret_cast<PyObject*>(lookup);
}

PyObject* cramer_statistic(PyObject*, PyObject* args) {
    PyObject* lookup_obj = nullptr;
    PyObject* indices_obj = nullptr;
    Py_ssize_t m = 0;
    if (!PyArg_ParseTuple(args, "OOn", &lookup_obj, &indices_obj, &m)) {
        return nullptr;
    }

    PyArrayObject* lookup = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(lookup_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (lookup == nullptr) {
        return nullptr;
    }
    PyArrayObject* indices = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(indices_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    if (indices == nullptr) {
        Py_DECREF(lookup);
        return nullptr;
    }

    if (PyArray_NDIM(lookup) != 2) {
        Py_DECREF(indices);
        Py_DECREF(lookup);
        PyErr_SetString(PyExc_ValueError, "lookup must be a 2D array.");
        return nullptr;
    }

    const npy_intp N = PyArray_DIM(lookup, 0);
    if (PyArray_DIM(lookup, 1) != N) {
        Py_DECREF(indices);
        Py_DECREF(lookup);
        PyErr_SetString(PyExc_ValueError, "lookup must be square.");
        return nullptr;
    }
    if (PyArray_SIZE(indices) != N) {
        Py_DECREF(indices);
        Py_DECREF(lookup);
        PyErr_SetString(PyExc_ValueError, "indices must have length N.");
        return nullptr;
    }

    LookupAccessor accessor{
        PyArray_BYTES(lookup),
        PyArray_STRIDES(lookup)[0],
        PyArray_STRIDES(lookup)[1],
        N,
    };

    const auto* indices_ptr =
        reinterpret_cast<const npy_intp*>(PyArray_DATA(indices));
    const double stat = cramer_statistic_impl(accessor, indices_ptr,
                                              static_cast<npy_intp>(m));

    Py_DECREF(indices);
    Py_DECREF(lookup);
    if (!std::isfinite(stat)) {
        return nullptr;
    }
    return PyFloat_FromDouble(stat);
}

PyObject* cramer_statistic_batch(PyObject*, PyObject* args) {
    PyObject* lookup_obj = nullptr;
    PyObject* resamples_obj = nullptr;
    Py_ssize_t m = 0;
    if (!PyArg_ParseTuple(args, "OOn", &lookup_obj, &resamples_obj, &m)) {
        return nullptr;
    }

    PyArrayObject* lookup = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(lookup_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (lookup == nullptr) {
        return nullptr;
    }
    PyArrayObject* resamples = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(resamples_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    if (resamples == nullptr) {
        Py_DECREF(lookup);
        return nullptr;
    }

    if (PyArray_NDIM(lookup) != 2) {
        Py_DECREF(resamples);
        Py_DECREF(lookup);
        PyErr_SetString(PyExc_ValueError, "lookup must be a 2D array.");
        return nullptr;
    }
    if (PyArray_NDIM(resamples) != 2) {
        Py_DECREF(resamples);
        Py_DECREF(lookup);
        PyErr_SetString(PyExc_ValueError, "resamples must be a 2D array.");
        return nullptr;
    }

    const npy_intp N = PyArray_DIM(lookup, 0);
    if (PyArray_DIM(lookup, 1) != N) {
        Py_DECREF(resamples);
        Py_DECREF(lookup);
        PyErr_SetString(PyExc_ValueError, "lookup must be square.");
        return nullptr;
    }
    if (PyArray_DIM(resamples, 1) != N) {
        Py_DECREF(resamples);
        Py_DECREF(lookup);
        PyErr_SetString(PyExc_ValueError,
                        "Each resample must have length equal to N.");
        return nullptr;
    }

    LookupAccessor accessor{
        PyArray_BYTES(lookup),
        PyArray_STRIDES(lookup)[0],
        PyArray_STRIDES(lookup)[1],
        N,
    };

    const npy_intp replicates = PyArray_DIM(resamples, 0);
    npy_intp out_dims[1] = {replicates};
    PyArrayObject* output = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(1, out_dims, NPY_DOUBLE));
    if (output == nullptr) {
        Py_DECREF(resamples);
        Py_DECREF(lookup);
        return nullptr;
    }

    const char* resample_base = PyArray_BYTES(resamples);
    const npy_intp stride = PyArray_STRIDES(resamples)[0];
    auto* out_ptr = static_cast<double*>(PyArray_DATA(output));

    for (npy_intp r = 0; r < replicates; ++r) {
        const auto* idx = reinterpret_cast<const npy_intp*>(resample_base + r * stride);
        out_ptr[r] = cramer_statistic_impl(accessor, idx,
                                           static_cast<npy_intp>(m));
    }

    Py_DECREF(resamples);
    Py_DECREF(lookup);
    return reinterpret_cast<PyObject*>(output);
}

PyObject* cramer_bootstrap_random(PyObject*, PyObject* args) {
    PyObject* lookup_obj = nullptr;
    Py_ssize_t m = 0;
    Py_ssize_t replicates = 0;
    int sim_kind_int = 0;
    unsigned long long seed = 0;
    Py_ssize_t workers = 0;
    if (!PyArg_ParseTuple(args, "OnniK|n", &lookup_obj, &m, &replicates, &sim_kind_int, &seed, &workers)) {
        return nullptr;
    }

    if (replicates <= 0) {
        PyErr_SetString(PyExc_ValueError, "replicates must be positive.");
        return nullptr;
    }

    SimKind sim_kind = static_cast<SimKind>(sim_kind_int);
    if (sim_kind != SimKind::kOrdinary && sim_kind != SimKind::kPermutation) {
        PyErr_SetString(PyExc_ValueError, "Unknown bootstrap mode.");
        return nullptr;
    }

    PyArrayObject* lookup = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(lookup_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (lookup == nullptr) {
        return nullptr;
    }
    if (PyArray_NDIM(lookup) != 2) {
        Py_DECREF(lookup);
        PyErr_SetString(PyExc_ValueError, "lookup must be a 2D array.");
        return nullptr;
    }

    const npy_intp N = PyArray_DIM(lookup, 0);
    if (PyArray_DIM(lookup, 1) != N) {
        Py_DECREF(lookup);
        PyErr_SetString(PyExc_ValueError, "lookup must be square.");
        return nullptr;
    }
    if (m <= 0 || m >= N) {
        Py_DECREF(lookup);
        PyErr_SetString(PyExc_ValueError, "m must be between 1 and N - 1.");
        return nullptr;
    }

    LookupAccessor accessor{
        PyArray_BYTES(lookup),
        PyArray_STRIDES(lookup)[0],
        PyArray_STRIDES(lookup)[1],
        N,
    };

    npy_intp out_dims[1] = {replicates};
    PyArrayObject* output = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(1, out_dims, NPY_DOUBLE));
    if (output == nullptr) {
        Py_DECREF(lookup);
        return nullptr;
    }

    auto* out_ptr = static_cast<double*>(PyArray_DATA(output));

    int worker_count = 0;
    if (workers > 0) {
        worker_count = static_cast<int>(workers);
    } else {
        worker_count = static_cast<int>(std::thread::hardware_concurrency());
    }
    if (worker_count <= 0) {
        worker_count = 1;
    }
    if (static_cast<npy_intp>(worker_count) > replicates) {
        worker_count = static_cast<int>(replicates);
    }

    const bool is_ordinary = (sim_kind == SimKind::kOrdinary);

    auto worker_fn = [&](npy_intp start, npy_intp end, unsigned long long worker_seed) {
        std::mt19937_64 rng(worker_seed);
        std::uniform_int_distribution<npy_intp> dist(0, N - 1);
        std::vector<npy_intp> indices(N);

        for (npy_intp r = start; r < end; ++r) {
            if (is_ordinary) {
                for (npy_intp i = 0; i < N; ++i) {
                    indices[i] = dist(rng);
                }
            } else {
                std::iota(indices.begin(), indices.end(), static_cast<npy_intp>(0));
                std::shuffle(indices.begin(), indices.end(), rng);
            }
            out_ptr[r] = cramer_statistic_impl(accessor, indices.data(), static_cast<npy_intp>(m));
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(worker_count - 1);

    const npy_intp chunk = replicates / worker_count;
    npy_intp start = 0;

    Py_BEGIN_ALLOW_THREADS;
    for (int w = 0; w < worker_count; ++w) {
        npy_intp end = (w == worker_count - 1) ? replicates : start + chunk;
        if (end < start) {
            end = start;
        }
        const unsigned long long worker_seed = seed + static_cast<unsigned long long>(w) * 1315423911ULL;
        if (w == worker_count - 1) {
            worker_fn(start, end, worker_seed);
        } else {
            threads.emplace_back(worker_fn, start, end, worker_seed);
        }
        start = end;
    }
    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }
    Py_END_ALLOW_THREADS;

    Py_DECREF(lookup);
    return reinterpret_cast<PyObject*>(output);
}

PyObject* cramer_statistic_from_data(PyObject*, PyObject* args) {
    PyObject* data_obj = nullptr;
    Py_ssize_t m = 0;
    int kernel_id = 0;
    if (!PyArg_ParseTuple(args, "Oni", &data_obj, &m, &kernel_id)) {
        return nullptr;
    }

    KernelId id = static_cast<KernelId>(kernel_id);

    PyArrayObject* data = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(data_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (data == nullptr) {
        return nullptr;
    }

    if (PyArray_NDIM(data) != 2) {
        Py_DECREF(data);
        PyErr_SetString(PyExc_ValueError, "data must be a 2D array.");
        return nullptr;
    }

    const npy_intp N = PyArray_DIM(data, 0);
    if (m <= 0 || m >= N) {
        Py_DECREF(data);
        PyErr_SetString(PyExc_ValueError, "m must be between 1 and N - 1.");
        return nullptr;
    }
    const npy_intp n = N - m;
    const npy_intp cols = PyArray_DIM(data, 1);

    auto* data_ptr = static_cast<double*>(PyArray_DATA(data));
    const npy_intp row_stride = PyArray_STRIDES(data)[0] /
                                static_cast<npy_intp>(sizeof(double));
    const npy_intp col_stride = PyArray_STRIDES(data)[1] /
                                static_cast<npy_intp>(sizeof(double));

    const int worker_count = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    std::vector<long double> xx_part(worker_count, 0.0L);
    std::vector<long double> xy_part(worker_count, 0.0L);
    std::atomic<bool> ok(true);

    auto worker_x = [&](int worker_idx, npy_intp start, npy_intp end) {
        long double local_xx = 0.0L;
        long double local_xy = 0.0L;
        for (npy_intp i = start; i < end; ++i) {
            const double* xi = data_ptr + i * row_stride;
            for (npy_intp j = 0; j < i; ++j) {
                const double* xj = data_ptr + j * row_stride;
                double sum_sq_diff = 0.0;
                for (npy_intp k = 0; k < cols; ++k) {
                    const double diff = xi[k * col_stride] - xj[k * col_stride];
                    sum_sq_diff += diff * diff;
                }
                if (!apply_kernel_value(id, sum_sq_diff)) {
                    ok.store(false, std::memory_order_relaxed);
                    return;
                }
                local_xx += 2.0L * static_cast<long double>(sum_sq_diff);
            }
            for (npy_intp j = m; j < N; ++j) {
                const double* yj = data_ptr + j * row_stride;
                double sum_sq_diff = 0.0;
                for (npy_intp k = 0; k < cols; ++k) {
                    const double diff = xi[k * col_stride] - yj[k * col_stride];
                    sum_sq_diff += diff * diff;
                }
                if (!apply_kernel_value(id, sum_sq_diff)) {
                    ok.store(false, std::memory_order_relaxed);
                    return;
                }
                local_xy += static_cast<long double>(sum_sq_diff);
            }
        }
        xx_part[worker_idx] = local_xx;
        xy_part[worker_idx] = local_xy;
    };

    std::vector<std::thread> threads;
    threads.reserve(worker_count);
    const npy_intp chunk = (m + worker_count - 1) / worker_count;

    Py_BEGIN_ALLOW_THREADS;
    for (int w = 0; w < worker_count; ++w) {
        const npy_intp start = w * chunk;
        const npy_intp end = std::min(static_cast<npy_intp>(m), start + chunk);
        if (start >= end) {
            continue;
        }
        threads.emplace_back(worker_x, w, start, end);
    }
    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }
    Py_END_ALLOW_THREADS;

    if (!ok.load(std::memory_order_relaxed)) {
        Py_DECREF(data);
        PyErr_SetString(PyExc_ValueError, "Kernel application failed.");
        return nullptr;
    }

    long double term_xx = 0.0L;
    long double term_xy = 0.0L;
    for (int w = 0; w < worker_count; ++w) {
        term_xx += xx_part[w];
        term_xy += xy_part[w];
    }

    std::vector<long double> yy_part(worker_count, 0.0L);
    threads.clear();

    auto worker_y = [&](int worker_idx, npy_intp start, npy_intp end) {
        long double local_yy = 0.0L;
        for (npy_intp i = start; i < end; ++i) {
            const double* yi = data_ptr + (m + i) * row_stride;
            for (npy_intp j = 0; j < i; ++j) {
                const double* yj = data_ptr + (m + j) * row_stride;
                double sum_sq_diff = 0.0;
                for (npy_intp k = 0; k < cols; ++k) {
                    const double diff = yi[k * col_stride] - yj[k * col_stride];
                    sum_sq_diff += diff * diff;
                }
                if (!apply_kernel_value(id, sum_sq_diff)) {
                    ok.store(false, std::memory_order_relaxed);
                    return;
                }
                local_yy += 2.0L * static_cast<long double>(sum_sq_diff);
            }
        }
        yy_part[worker_idx] = local_yy;
    };

    const npy_intp chunk_y = (n + worker_count - 1) / worker_count;
    Py_BEGIN_ALLOW_THREADS;
    for (int w = 0; w < worker_count; ++w) {
        const npy_intp start = w * chunk_y;
        const npy_intp end = std::min(static_cast<npy_intp>(n), start + chunk_y);
        if (start >= end) {
            continue;
        }
        threads.emplace_back(worker_y, w, start, end);
    }
    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }
    Py_END_ALLOW_THREADS;

    Py_DECREF(data);

    if (!ok.load(std::memory_order_relaxed)) {
        PyErr_SetString(PyExc_ValueError, "Kernel application failed.");
        return nullptr;
    }

    long double term_yy = 0.0L;
    for (int w = 0; w < worker_count; ++w) {
        term_yy += yy_part[w];
    }

    const long double m_f = static_cast<long double>(m);
    const long double n_f = static_cast<long double>(n);
    const long double prefactor = (m_f * n_f) / (m_f + n_f);
    const long double result =
        prefactor *
        ((2.0L * term_xy) / (m_f * n_f) - term_xx / (m_f * m_f) - term_yy / (n_f * n_f));

    return PyFloat_FromDouble(static_cast<double>(result));
}

PyObject* apply_builtin_kernel(PyObject*, PyObject* args) {
    PyObject* lookup_obj = nullptr;
    int kernel_id = 0;
    if (!PyArg_ParseTuple(args, "Oi", &lookup_obj, &kernel_id)) {
        return nullptr;
    }

    PyArrayObject* lookup = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(lookup_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2));
    if (lookup == nullptr) {
        return nullptr;
    }

    auto* data = static_cast<double*>(PyArray_DATA(lookup));
    const npy_intp size = PyArray_SIZE(lookup);
    KernelId id = static_cast<KernelId>(kernel_id);

    auto transform_block = [&](npy_intp start, npy_intp end) {
        for (npy_intp idx = start; idx < end; ++idx) {
            double val = data[idx];
            if (!apply_kernel_value(id, val)) {
                return false;
            }
            data[idx] = val;
        }
        return true;
    };

    std::atomic<bool> ok(true);
    int worker_count = static_cast<int>(std::thread::hardware_concurrency());
    if (worker_count <= 1 || size < (1 << 15)) {
        ok.store(transform_block(0, size), std::memory_order_relaxed);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(worker_count);
        const npy_intp chunk = (size + worker_count - 1) / worker_count;
        Py_BEGIN_ALLOW_THREADS;
        for (int w = 0; w < worker_count; ++w) {
            const npy_intp start = w * chunk;
            const npy_intp end = std::min(size, start + chunk);
            if (start >= end) {
                continue;
            }
            threads.emplace_back([&, start, end]() {
                if (!transform_block(start, end)) {
                    ok.store(false, std::memory_order_relaxed);
                }
            });
        }
        for (auto& th : threads) {
            if (th.joinable()) {
                th.join();
            }
        }
        Py_END_ALLOW_THREADS;
    }

    if (!ok.load(std::memory_order_relaxed)) {
        PyArray_DiscardWritebackIfCopy(lookup);
        Py_DECREF(lookup);
        PyErr_SetString(PyExc_ValueError, "Unknown kernel id.");
        return nullptr;
    }

    if (PyArray_ResolveWritebackIfCopy(lookup) < 0) {
        Py_DECREF(lookup);
        return nullptr;
    }
    Py_DECREF(lookup);
    Py_RETURN_NONE;
}

PyMethodDef METHODS[] = {
    {"calculate_lookup_matrix", reinterpret_cast<PyCFunction>(calculate_lookup_matrix), METH_VARARGS,
     "Compute pairwise squared distances for rows of the input array."},
    {"calculate_lookup_matrix_with_kernel", reinterpret_cast<PyCFunction>(calculate_lookup_matrix_with_kernel), METH_VARARGS,
     "Compute pairwise distances and apply a built-in kernel in one pass."},
    {"cramer_statistic", reinterpret_cast<PyCFunction>(cramer_statistic), METH_VARARGS,
     "Compute the Cramér statistic for a single set of indices."},
    {"cramer_statistic_batch", reinterpret_cast<PyCFunction>(cramer_statistic_batch), METH_VARARGS,
     "Compute the Cramér statistic for multiple index sets."},
    {"cramer_bootstrap_random", reinterpret_cast<PyCFunction>(cramer_bootstrap_random), METH_VARARGS,
     "Generate bootstrap statistics using native resampling."},
    {"cramer_statistic_from_data", reinterpret_cast<PyCFunction>(cramer_statistic_from_data), METH_VARARGS,
     "Compute the Cramér statistic directly from raw data for built-in kernels."},
    {"apply_builtin_kernel", reinterpret_cast<PyCFunction>(apply_builtin_kernel), METH_VARARGS,
     "Apply an in-place kernel transformation to the lookup matrix."},
    {nullptr, nullptr, 0, nullptr}};

PyModuleDef MODULE = {
    PyModuleDef_HEAD_INIT,
    "_lookup",
    nullptr,
    -1,
    METHODS,
    nullptr,
    nullptr,
    nullptr,
    nullptr};

}  // namespace

PyMODINIT_FUNC PyInit__lookup(void) {
    import_array();
    return PyModule_Create(&MODULE);
}
