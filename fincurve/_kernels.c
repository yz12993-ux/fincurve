/*
 * Compiled kernels for fincurve.
 *
 * local_linear: tricube-weighted local linear regression with a nearest-neighbour
 * bandwidth. The training abscissae must be sorted ascending, which turns the
 * neighbour search into a two-pointer window walk: O(n_eval * m) instead of the
 * O(n_eval * n) partition the NumPy fallback performs for every evaluation point.
 *
 * The kernel follows the NumPy reference exactly:
 *   h      = distance to the m-th nearest training point (self included)
 *   weight = (1 - (d / (1.0001 h))^3)^3 for d < 1.0001 h, else 0
 *   slope  = 0 when the weighted x-variance is negligible
 * With loo != 0 the evaluation points are the training points and each point's
 * own weight is zeroed (leave-one-out prediction).
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <string.h>

static int
get_float64_buffer(PyObject *obj, Py_buffer *view, int writable, const char *name)
{
    int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT | (writable ? PyBUF_WRITABLE : 0);
    if (PyObject_GetBuffer(obj, view, flags) < 0) {
        return -1;
    }
    const char *fmt = view->format ? view->format : "";
    if (fmt[0] == '<' || fmt[0] == '=' || fmt[0] == '@') {
        fmt++;
    }
    if (view->ndim != 1 || view->itemsize != sizeof(double) || strcmp(fmt, "d") != 0) {
        PyBuffer_Release(view);
        PyErr_Format(PyExc_TypeError, "%s must be a contiguous 1-D float64 array", name);
        return -1;
    }
    return 0;
}

static Py_ssize_t
lower_bound(const double *x, Py_ssize_t n, double value)
{
    Py_ssize_t lo = 0, hi = n;
    while (lo < hi) {
        Py_ssize_t mid = lo + (hi - lo) / 2;
        if (x[mid] < value) {
            lo = mid + 1;
        }
        else {
            hi = mid;
        }
    }
    return lo;
}

static void
local_linear_core(const double *x, const double *y, Py_ssize_t n, const double *xe,
                  double *out, Py_ssize_t ne, Py_ssize_t m, int loo)
{
    double span = x[n - 1] - x[0];
    if (!(span > 0.0)) {
        span = 1.0;
    }
    for (Py_ssize_t i = 0; i < ne; i++) {
        double x0 = xe[i];
        Py_ssize_t pos = lower_bound(x, n, x0);
        Py_ssize_t lo = pos, hi = pos;
        while (hi - lo < m) {
            if (lo == 0) {
                hi++;
            }
            else if (hi == n) {
                lo--;
            }
            else if (x0 - x[lo - 1] <= x[hi] - x0) {
                lo--;
            }
            else {
                hi++;
            }
        }
        double h = fmax(x0 - x[lo], x[hi - 1] - x0);
        if (!(h > 0.0)) {
            lo = 0;
            hi = n;
            h = fmax(fabs(x0 - x[0]), fabs(x[n - 1] - x0));
            if (!(h > 0.0)) {
                h = 1.0;
            }
        }
        double hh = h * 1.0001;
        while (lo > 0 && x0 - x[lo - 1] < hh) {
            lo--;
        }
        while (hi < n && x[hi] - x0 < hh) {
            hi++;
        }

        double sw = 0.0, sx = 0.0, sy = 0.0;
        for (Py_ssize_t j = lo; j < hi; j++) {
            if (loo && j == i) {
                continue;
            }
            double d = fabs(x[j] - x0) / hh;
            if (d >= 1.0) {
                continue;
            }
            double t = 1.0 - d * d * d;
            double k = t * t * t;
            sw += k;
            sx += k * x[j];
            sy += k * y[j];
        }
        if (!(sw > 0.0)) {
            out[i] = NAN;
            continue;
        }
        double xm = sx / sw, ym = sy / sw, sxx = 0.0, sxy = 0.0;
        for (Py_ssize_t j = lo; j < hi; j++) {
            if (loo && j == i) {
                continue;
            }
            double d = fabs(x[j] - x0) / hh;
            if (d >= 1.0) {
                continue;
            }
            double t = 1.0 - d * d * d;
            double k = t * t * t;
            double dx = x[j] - xm;
            sxx += k * dx * dx;
            sxy += k * dx * (y[j] - ym);
        }
        double b = (sxx > 1e-12 * sw * span * span) ? sxy / sxx : 0.0;
        out[i] = ym + b * (x0 - xm);
    }
}

static PyObject *
local_linear(PyObject *self, PyObject *args)
{
    PyObject *xo, *yo, *eo, *oo;
    Py_ssize_t m;
    int loo;
    if (!PyArg_ParseTuple(args, "OOOOnp", &xo, &yo, &eo, &oo, &m, &loo)) {
        return NULL;
    }
    Py_buffer xb, yb, eb, ob;
    if (get_float64_buffer(xo, &xb, 0, "x") < 0) {
        return NULL;
    }
    if (get_float64_buffer(yo, &yb, 0, "y") < 0) {
        PyBuffer_Release(&xb);
        return NULL;
    }
    if (get_float64_buffer(eo, &eb, 0, "x_eval") < 0) {
        PyBuffer_Release(&xb);
        PyBuffer_Release(&yb);
        return NULL;
    }
    if (get_float64_buffer(oo, &ob, 1, "out") < 0) {
        PyBuffer_Release(&xb);
        PyBuffer_Release(&yb);
        PyBuffer_Release(&eb);
        return NULL;
    }
    Py_ssize_t n = xb.len / (Py_ssize_t)sizeof(double);
    Py_ssize_t ne = eb.len / (Py_ssize_t)sizeof(double);
    PyObject *result = NULL;
    if (n < 1 || yb.len != xb.len || ob.len != eb.len || (loo && ne != n)) {
        PyErr_SetString(PyExc_ValueError, "inconsistent array lengths");
        goto done;
    }
    if (m > n) {
        m = n;
    }
    if (m < 1) {
        m = 1;
    }
    Py_BEGIN_ALLOW_THREADS
    local_linear_core((const double *)xb.buf, (const double *)yb.buf, n,
                      (const double *)eb.buf, (double *)ob.buf, ne, m, loo);
    Py_END_ALLOW_THREADS
    Py_INCREF(Py_None);
    result = Py_None;
done:
    PyBuffer_Release(&xb);
    PyBuffer_Release(&yb);
    PyBuffer_Release(&eb);
    PyBuffer_Release(&ob);
    return result;
}

static PyMethodDef kernel_methods[] = {
    {"local_linear", local_linear, METH_VARARGS,
     "local_linear(x_sorted, y, x_eval, out, m, loo): tricube local linear regression into out."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef kernel_module = {
    PyModuleDef_HEAD_INIT, "_kernels", "Compiled numerical kernels for fincurve.", -1, kernel_methods,
};

PyMODINIT_FUNC
PyInit__kernels(void)
{
    return PyModule_Create(&kernel_module);
}
