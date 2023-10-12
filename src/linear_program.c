/*
 * Lua Linear program functions
 *
 * Copyright (C) 2017-2023 Andre Naef
 */


#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <lapacke.h>
#include <lauxlib.h>
#include "linear_core.h"
#include "linear_program.h"


#if LUA_VERSION_NUM < 502
#define lua_rawlen  lua_objlen
#endif


typedef struct linear_spline_s {
	size_t   n;              /* number of polynomials */
	int      extrapolation;  /* extrapolation mode */
	double  *x;              /* x cut-ins; n + 1 values */
	double  *a;              /* constant coefficients; equals y; n + 1 values */
	double  *b;              /* linear coefficients; n values */
	double  *c;              /* quadratic coefficients; n values */
	double  *d;              /* cubic coefficients; n values */
} linear_spline_t;


static inline CBLAS_TRANSPOSE linear_checktranspose(lua_State *L, int index);
static inline char linear_lapacktranspose(CBLAS_TRANSPOSE transpose);
static int linear_dot(lua_State *L);
static int linear_ger(lua_State *L);
static int linear_gemv(lua_State *L);
static int linear_gemm(lua_State *L);
static int linear_gesv(lua_State *L);
static int linear_gels(lua_State *L);
static int linear_inv(lua_State *L);
static int linear_det(lua_State *L);
static int linear_cov(lua_State *L);
static int linear_corr(lua_State *L);
static int linear_ranks(lua_State *L);
static int linear_quantile(lua_State *L);
static int linear_rank(lua_State *L);
static int linear_interpolant(lua_State *L);
static int linear_spline(lua_State *L);


static const char *const LINEAR_TRANSPOSES[] = {"notrans", "trans", NULL};
static const char *const LINEAR_BOUNDARIES[] = {"not-a-knot", "natural", "clamped", NULL};
static const char *const LINEAR_EXTRAPOLATIONS[] = {"none", "const", "linear", "cubic", NULL};


static inline CBLAS_TRANSPOSE linear_checktranspose (lua_State *L, int index) {
	return luaL_checkoption(L, index, "notrans", LINEAR_TRANSPOSES) == 0 ? CblasNoTrans
			: CblasTrans;
}

static inline char linear_lapacktranspose (CBLAS_TRANSPOSE transpose) {
	return transpose == CblasNoTrans ? 'N' : 'T';
}

static int linear_dot (lua_State *L) {
	linear_vector_t  *x, *y;

	x = luaL_checkudata(L, 1, LINEAR_VECTOR);
	y = luaL_checkudata(L, 2, LINEAR_VECTOR);
	luaL_argcheck(L, y->length == x->length, 2, "dimension mismatch");
	lua_pushnumber(L, cblas_ddot(x->length, x->values, x->inc, y->values, y->inc));
	return 1;
}

static int linear_ger (lua_State *L) {
	double            alpha;
	linear_vector_t  *x, *y;
	linear_matrix_t  *A;

	x = luaL_checkudata(L, 1, LINEAR_VECTOR);
	y = luaL_checkudata(L, 2, LINEAR_VECTOR);
	A = luaL_checkudata(L, 3, LINEAR_MATRIX);
	luaL_argcheck(L, x->length == A->rows, 1, "dimension mismatch");
	luaL_argcheck(L, y->length == A->cols, 2, "dimension mismatch");
	alpha = luaL_optnumber(L, 4, 1.0);
	cblas_dger(A->order, A->rows, A->cols, alpha, x->values, x->inc, y->values, y->inc,
			A->values, A->ld);
	return 0;
}

static int linear_gemv (lua_State *L) {
	size_t            m, n;
	double            alpha, beta;
	CBLAS_TRANSPOSE   ta;
	linear_matrix_t  *A;
	linear_vector_t  *x, *y;

	A = luaL_checkudata(L, 1, LINEAR_MATRIX);
	x = luaL_checkudata(L, 2, LINEAR_VECTOR);
	y = luaL_checkudata(L, 3, LINEAR_VECTOR);
	ta = linear_checktranspose(L, 4);
	m = ta == CblasNoTrans ? A->rows : A->cols;
	n = ta == CblasNoTrans ? A->cols : A->rows;
	luaL_argcheck(L, x->length == n, 2, "dimension mismatch");
	luaL_argcheck(L, y->length == m, 3, "dimension mismatch");
	alpha = luaL_optnumber(L, 5, 1.0);
	beta = luaL_optnumber(L, 6, 0.0);
	cblas_dgemv(A->order, ta, A->rows, A->cols, alpha, A->values, A->ld, x->values, x->inc,
			beta, y->values, y->inc);
	return 0;
}

static int linear_gemm (lua_State *L) {
	size_t            m, n, ka, kb;
	double            alpha, beta;
	CBLAS_TRANSPOSE   ta, tb;
	linear_matrix_t  *A, *B, *C;

	A = luaL_checkudata(L, 1, LINEAR_MATRIX);
	B = luaL_checkudata(L, 2, LINEAR_MATRIX);
	luaL_argcheck(L, B->order == A->order, 2, "order mismatch");
	C = luaL_checkudata(L, 3, LINEAR_MATRIX);
	luaL_argcheck(L, C->order == A->order, 3, "order mismatch");
	ta = linear_checktranspose(L, 4);
	tb = linear_checktranspose(L, 5);
	m = ta == CblasNoTrans ? A->rows : A->cols;
	n = tb == CblasNoTrans ? B->cols : B->rows;
	ka = ta == CblasNoTrans ? A->cols : A->rows;
	kb = tb == CblasNoTrans ? B->rows : B->cols;
	luaL_argcheck(L, ka == kb, 2, "dimension mismatch");
	alpha = luaL_optnumber(L, 6, 1.0);
	beta = luaL_optnumber(L, 7, 0.0);
	cblas_dgemm(A->order, ta, tb, m, n, ka, alpha, A->values, A->ld, B->values, B->ld, beta,
			C->values, C->ld);
	return 0;
}

static int linear_gesv (lua_State *L) {
	int              *ipiv, result;
	linear_matrix_t  *A, *B;

	A = luaL_checkudata(L, 1, LINEAR_MATRIX);
	luaL_argcheck(L, A->rows == A->cols, 1, "not square");
	B = luaL_checkudata(L, 2, LINEAR_MATRIX);
	luaL_argcheck(L, B->order == A->order, 2, "order mismatch");
	luaL_argcheck(L, B->rows == A->rows, 2, "dimension mismatch");
	ipiv = calloc(A->rows, sizeof(lapack_int));
	if (ipiv == NULL) {
		return luaL_error(L, "cannot allocate indexes");
	}
	result = LAPACKE_dgesv(A->order, A->rows, B->cols, A->values, A->ld, ipiv, B->values,
		B->ld);
	free(ipiv);
	if (result < 0) {
		return luaL_error(L, "internal error");
	}
	lua_pushboolean(L, result == 0);
	return 1;
}

static int linear_gels (lua_State *L) {
	int               result;
	char              ta;
	linear_matrix_t  *A, *B;

	A = luaL_checkudata(L, 1, LINEAR_MATRIX);
	B = luaL_checkudata(L, 2, LINEAR_MATRIX);
	luaL_argcheck(L, B->order == A->order, 2, "order mismatch");
	ta = linear_lapacktranspose(linear_checktranspose(L, 3));
	luaL_argcheck(L, B->rows == (A->rows >= A->cols ? A->rows : A->cols), 2,
			"dimension mismatch");
	result = LAPACKE_dgels(A->order, ta, A->rows, A->cols, B->cols, A->values, A->ld,
			B->values, B->ld);
	if (result < 0) {
		return luaL_error(L, "internal error");
	}
	lua_pushboolean(L, result == 0);
	return 1;
}

static int linear_inv (lua_State *L) {
	int              *ipiv, result;
	linear_matrix_t  *A;

	A = luaL_checkudata(L, 1, LINEAR_MATRIX);
	luaL_argcheck(L, A->rows == A->cols, 1, "not square");
	ipiv = calloc(A->rows, sizeof(lapack_int));
	if (ipiv == NULL) {
		return luaL_error(L, "cannot allocate indexes");
	}
	result = LAPACKE_dgetrf(A->order, A->rows, A->cols, A->values, A->ld, ipiv);
	if (result != 0) {
		free(ipiv);
		if (result < 0) {
			return luaL_error(L, "internal error");
		}
		lua_pushboolean(L, 0);  /* matrix is singular at machine precision */
		return 1;
	}
	result = LAPACKE_dgetri(A->order, A->rows, A->values, A->ld, ipiv);
	free(ipiv);
	if (result < 0) {
		return luaL_error(L, "internal error");
	}
	lua_pushboolean(L, result == 0);
	return 1;
}

static int linear_det (lua_State *L) {
	int              *ipiv, result, neg;
	size_t            n, i;
	double           *copy, *d, *s, det;
	linear_matrix_t  *A;

	/* check and process arguments */
	A = luaL_checkudata(L, 1, LINEAR_MATRIX);
	luaL_argcheck(L, A->rows == A->cols, 1, "not square");
	n = A->rows;

	/* copy matrix */
	copy = calloc(n * n, sizeof(double));
	if (copy == NULL) {
		return luaL_error(L, "cannot allocate values");
	}
	d = copy;
	s = A->values;
	for (i = 0; i < n; i++) {
		memcpy(d, s, n * sizeof(double));
		d += n;
		s += A->ld;
	}

	/* invoke subprograms */
	ipiv = calloc(n, sizeof(lapack_int));
	if (ipiv == NULL) {
		free(copy);
		return luaL_error(L, "cannot allocate indexes");
	}
	result = LAPACKE_dgetrf(A->order, n, n, copy, n, ipiv);
	if (result != 0) {
		free(copy);
		free(ipiv);
		if (result < 0) {
			return luaL_error(L, "internal error");
		}
		lua_pushnumber(L, 0.0);  /* matrix is singular at machine precision */
		return 1;
	}

	/* calculate determinant */
	det = 1.0;
	neg = 0;
	for (i = 0; i < n; i++) {
		det *= copy[i * n + i];
		if ((size_t)ipiv[i] != i + 1) {
			neg = !neg;
		}
	}
	free(copy);
	free(ipiv);
	lua_pushnumber(L, neg ? -det : det);
	return 1;
}

static int linear_cov (lua_State *L) {
	size_t            i, j, k, ddof;
	double           *means, *v, *vi, *vj, sum;
	linear_matrix_t  *A, *B;

	/* check and process arguments */
	A = luaL_checkudata(L, 1, LINEAR_MATRIX);
	B = luaL_checkudata(L, 2, LINEAR_MATRIX);
	luaL_argcheck(L, A->cols == B->rows, 2, "dimension mismatch");
	luaL_argcheck(L, B->rows == B->cols, 2, "not square");
	ddof = luaL_optinteger(L, 3, 0);
	luaL_argcheck(L, ddof < A->rows, 3, "bad ddof");

	/* calculate means */
	means = calloc(A->cols, sizeof(double));
	if (means == NULL) {
		return luaL_error(L, "cannot allocate values");
	}
	if (A->order == CblasColMajor) {
		for (i = 0; i < A->cols; i++) {
			sum = 0.0;
			v = &A->values[i * A->ld];
			for (j = 0; j < A->rows; j++) {
				sum += v[j];
			}
			means[i] = sum / A->rows;
		}
	} else {
		for (i = 0; i < A->cols; i++) {
			sum = 0.0;
			v = &A->values[i];
			for (j = 0; j < A->rows; j++) {
				sum += *v;
				v += A->ld;
			}
			means[i] = sum / A->rows;
		}
	}

	/* calculate covariances */
	if (A->order == CblasColMajor) {
		for (i = 0; i < A->cols; i++) {
			for (j = i; j < A->cols; j++) {
				sum = 0.0;
				vi = &A->values[i * A->ld];
				vj = &A->values[j * A->ld];
				for (k = 0; k < A->rows; k++) {
					sum += (vi[k] - means[i]) * (vj[k] - means[j]);
				}
				B->values[i * B->ld + j] = B->values[j * B->ld + i] = sum
						/ (A->rows - ddof);
			}
		}
	} else {
		for (i = 0; i < A->cols; i++) {
			for (j = i; j < A->cols; j++) {
				sum = 0.0;
				vi = &A->values[i];
				vj = &A->values[j];
				for (k = 0; k < A->rows; k++) {
					sum += (*vi - means[i]) * (*vj - means[j]);
					vi += A->ld;
					vj += A->ld;
				}
				B->values[i * B->ld + j] = B->values[j * B->ld + i] = sum
						/ (A->rows - ddof);
			}
		}
	}
	free(means);
	return 0;
}

static int linear_corr (lua_State *L) {
	size_t            i, j, k;
	double           *means, *stds, *v, *vi, *vj, sum;
	linear_matrix_t  *A, *B;

	/* check and process arguments */
	A = luaL_checkudata(L, 1, LINEAR_MATRIX);
	B = luaL_checkudata(L, 2, LINEAR_MATRIX);
	luaL_argcheck(L, A->cols == B->rows, 2, "dimension mismatch");
	luaL_argcheck(L, B->rows == B->cols, 2, "not square");

	/* calculate means and stds */
	means = calloc(A->cols, sizeof(double));
	if (means == NULL) {
		return luaL_error(L, "cannot allocate values");
	}
	stds = calloc(A->cols, sizeof(double));
	if (stds == NULL) {
		free(means);
		return luaL_error(L, "cannot allocate values");
	}
	if (A->order == CblasColMajor) {
		for (i = 0; i < A->cols; i++) {
			sum = 0.0;
			v = &A->values[i * A->ld];
			for (j = 0; j < A->rows; j++) {
				sum += v[j];
			}
			means[i] = sum / A->rows;
			sum = 0.0;
			v = &A->values[i * A->ld];
			for (j = 0; j < A->rows; j++) {
				sum += (v[j] - means[i]) * (v[j] - means[i]);
			}
			stds[i] = sqrt(sum);
		}
	} else {
		for (i = 0; i < A->cols; i++) {
			sum = 0.0;
			v = &A->values[i];
			for (j = 0; j < A->rows; j++) {
				sum += *v;
				v += A->ld;
			}
			means[i] = sum / A->rows;
			sum = 0.0;
			v = &A->values[i];
			for (j = 0; j < A->rows; j++) {
				sum += (*v - means[i]) * (*v - means[i]);
				v += A->ld;
			}
			stds[i] = sqrt(sum);
		}
	}

	/* calculate Pearson product-moment correlation coefficients */
	if (A->order == CblasColMajor) {
		for (i = 0; i < A->cols; i++) {
			for (j = i; j < A->cols; j++) {
				sum = 0.0;
				vi = &A->values[i * A->ld];
				vj = &A->values[j * A->ld];
				for (k = 0; k < A->rows; k++) {
					sum += (vi[k] - means[i]) * (vj[k] - means[j]);
				}
				B->values[i * B->ld + j] = B->values[j * B->ld + i] = sum
						/ (stds[i] * stds[j]);
			}
		}
	} else {
		for (i = 0; i < A->cols; i++) {
			for (j = i; j < A->cols; j++) {
				sum = 0.0;
				vi = &A->values[i];
				vj = &A->values[j];
				for (k = 0; k < A->rows; k++) {
					sum += (*vi - means[i]) * (*vj - means[j]);
					vi += A->ld;
					vj += A->ld;
				}
				B->values[i * B->ld + j] = B->values[j * B->ld + i] = sum
						/ (stds[i] * stds[j]);
			}
		}
	}
	free(means);
	free(stds);
	return 0;
}

static int linear_ranks (lua_State *L) {
	const char  *mode;
	int          q, k, l, u;

	q = luaL_checkinteger(L, 1);
	luaL_argcheck(L, q > 0, 1, "bad sign");
	mode = luaL_optstring(L, 2, "");
	l = strchr(mode, 'z') ? 0 : 1;
	u = strchr(mode, 'q') ? q : q - 1;
	lua_createtable(L, u - l + 1, 0);
	for (k = l; k <= u; k++) {
		lua_pushnumber(L, (double)k / q);
		lua_rawseti(L, -2, k - l + 1);
	}
	return 1;
}

static int linear_quantile (lua_State *L) {
	size_t            i, index, count;
	double           *copy, *v;
	double            rank, pos, frac;
	linear_vector_t  *x;

	/* copy vector */
	x = luaL_checkudata(L, 1, LINEAR_VECTOR);
	copy = lua_newuserdata(L, x->length * sizeof(double));
	v = x->values;
	for (i = 0; i < x->length; i++) {
		if (isnan(*v)) {
			return luaL_error(L, "bad value");
		}
		copy[i] = *v;
		v += x->inc;
	}
	qsort(copy, x->length, sizeof(double), linear_comparison_handler);

	/* handle cases */
	switch (lua_type(L, 2)) {
	case LUA_TNUMBER:
		rank = lua_tonumber(L, 2);
		luaL_argcheck(L, rank >= 0 && rank <= 1, 2, "bad rank");
		pos = rank * (x->length - 1);
		frac = fmod(pos, 1);
		index = floor(pos);
		if (frac > 0) {
			lua_pushnumber(L, copy[index] + (copy[index + 1] - copy[index]) * frac);
		} else {
			lua_pushnumber(L, copy[index]);
		}
		break;

	case LUA_TTABLE:
		count = lua_rawlen(L, 2);
		lua_createtable(L, count, 0);
		for (i = 0; i < count; i++) {
			if (linear_rawgeti(L, 2, i + 1) != LUA_TNUMBER) {
				return luaL_error(L, "bad rank at index %d", i + 1);
			}
			rank = lua_tonumber(L, -1);
			if (!(rank >= 0 && rank <= 1)) {
				return luaL_error(L, "bad rank at index %d", i + 1);
			}
			pos = rank * (x->length - 1);
			frac = fmod(pos, 1);
			index = floor(pos);
			if (frac > 0) {
				lua_pushnumber(L, copy[index] + (copy[index + 1] - copy[index])
						* frac);
			} else {
				lua_pushnumber(L, copy[index]);
			}
			lua_rawseti(L, -3, i + 1);
			lua_pop(L, 1);
		}
		break;

	default:
		luaL_argerror(L, 2, "number or table expected");
	}
	return 1;
}

static int linear_rank (lua_State *L) {
	size_t            i, lower, upper, mid, count;
	double           *copy, *v;
	double            q;
	linear_vector_t  *x;

	/* copy vector */
	x = luaL_checkudata(L, 1, LINEAR_VECTOR);
	luaL_argcheck(L, 1, x->length >= 2, "bad vector");
	copy = lua_newuserdata(L, x->length * sizeof(double));
	v = x->values;
	for (i = 0; i < x->length; i++) {
		if (isnan(*v)) {
			return luaL_error(L, "bad value");
		}
		copy[i] = *v;
		v += x->inc;
	}
	qsort(copy, x->length, sizeof(double), linear_comparison_handler);

	/* handle cases */
	switch (lua_type(L, 2)) {
	case LUA_TNUMBER:
		q = lua_tonumber(L, 2);
		if (q <= copy[0]) {
			lua_pushnumber(L, 0);
		} else if (q >= copy[x->length - 1]) {
			lua_pushnumber(L, 1);
		} else if (!isnan(q)) {
			lower = 0;
			upper = x->length - 1;
			while (lower <= upper) {
				mid = (lower + upper) / 2;
				if (copy[mid] < q) {
					lower = mid + 1;
				} else {
					upper = mid - 1;
				}
			}
			lua_pushnumber(L, (upper + (q - copy[upper]) / (copy[lower]
					- copy[upper])) / (x->length - 1));
		} else {
			return luaL_error(L, "bad quantile");
		}
		break;

	case LUA_TTABLE:
		count = lua_rawlen(L, 2);
		lua_createtable(L, count, 0);
		for (i = 0; i < count; i++) {
			if (linear_rawgeti(L, 2, i + 1) != LUA_TNUMBER) {
				return luaL_error(L, "bad quantile at index %d", i + 1);
			}
			q = lua_tonumber(L, -1);
			if (q <= copy[0]) {
				lua_pushnumber(L, 0);
			} else if (q >= copy[x->length - 1]) {
				lua_pushnumber(L, 1);
			} else if (!isnan(q)) {
				lower = 0;
				upper = x->length - 1;
				while (lower <= upper) {
					mid = (lower + upper) / 2;
					if (copy[mid] < q) {
						lower = mid + 1;
					} else {
						upper = mid - 1;
					}
				}
				lua_pushnumber(L, (upper + (q - copy[upper]) / (copy[lower]
						- copy[upper])) / (x->length - 1));
			} else {
				return luaL_error(L, "bad quantile at index %d", i + 1);
			}
			lua_rawseti(L, -3, i + 1);
			lua_pop(L, 1);
		}
		break;

	default:
		luaL_argerror(L, 2, "number or table expected");
	}
	return 1;
}

static int linear_interpolant (lua_State *L) {
	double            x, y;
	size_t            lower, upper, mid;
	linear_spline_t  *spline;

	spline = (void *)lua_topointer(L, lua_upvalueindex(1));
	x = luaL_checknumber(L, 1);
	if (x >= spline->x[0] && x <= spline->x[spline->n]) {
		/* interpolation */
		lower = 0;
		upper = spline->n - 1;
		while (lower <= upper) {
			mid = (lower + upper) / 2;
			if (spline->x[mid] <= x) {
				lower = mid + 1;
			} else {
				upper = mid - 1;
			}
		}
		x -= spline->x[upper];
		y = ((spline->d[upper] * x + spline->c[upper]) * x + spline->b[upper]) * x
				+ spline->a[upper];
	} else if (x < spline->x[0]) {
		/* left extrapolation */
		switch (spline->extrapolation) {
		case 0:  /* none */
			return luaL_argerror(L, 1, "too small");

		case 1:  /* const */
			y = spline->a[0];
			break;

		case 2:  /* linear */
			x -= spline->x[0];
			y = spline->b[0] * x + spline->a[0];
			break;

		case 3:  /* cubic */
			x -= spline->x[0];
			y = ((spline->d[0] * x + spline->c[0]) * x + spline->b[0]) * x
				+ spline->a[0];
			break;

		default:
			return 0;  /* not reached */
		}
	} else if (x > spline->x[spline->n]) {
		/* right extrapolation */
		switch (spline->extrapolation) {
			case 0:  /* none */
				return luaL_argerror(L, 1, "too large");

			case 1:  /* const */
				y = spline->a[spline->n];
				break;

			case 2:  /* linear */
				x -= spline->x[spline->n];
				y = spline->b[spline->n - 1] * x + spline->a[spline->n];
				break;

			case 3:  /* cubic */
				x -= spline->x[spline->n - 1];
				y = ((spline->d[spline->n - 1] * x + spline->c[spline->n - 1]) * x
						+ spline->b[spline->n - 1]) * x
						+ spline->a[spline->n - 1];
				break;

			default:
				return 0;  /* not reached */
		}
	} else {
		return luaL_argerror(L, 1, "bad value");
	}
	lua_pushnumber(L, y);
	return 1;
}

static int linear_spline (lua_State *L) {
	int                    boundary, extrapolation;
	double                *h, *dl, *d, *du, *b;
	double                 da, db;
	size_t                 i, n;
	linear_vector_t       *x, *y;
	linear_spline_t       *spline;

	/* process arguments */
	x = luaL_checkudata(L, 1, LINEAR_VECTOR);
	y = luaL_checkudata(L, 2, LINEAR_VECTOR);
	boundary = luaL_checkoption(L, 3, "not-a-knot", LINEAR_BOUNDARIES);
	extrapolation = luaL_checkoption(L, 4, "none", LINEAR_EXTRAPOLATIONS);
	da = boundary == 2 ? luaL_checknumber(L, 5) : 0.0;  /* clamped */
	db = boundary == 2 ? luaL_checknumber(L, 6) : 0.0;
	luaL_argcheck(L, x->length >= (boundary == 0 ? 4 : 3), 0, "bad dimension");
	luaL_argcheck(L, x->length == y->length, 2, "dimension mismatch");
	n = x->length - 1;  /* number of polynomials */

	/* prepare the tridiagonal system */
	spline = lua_newuserdata(L, sizeof(linear_spline_t) + (5 * n + 2) * sizeof(double));
	spline->n = n;
	spline->extrapolation = extrapolation;
	spline->x = (double *)((char *)(spline) + sizeof(linear_spline_t));
	spline->a = spline->x + (n + 1);
	spline->b = spline->a + (n + 1);
	spline->c = spline->b + n;
	spline->d = spline->c + n;
	h = spline->d;
	dl = spline->b;
	d = spline->x;
	du = spline->c;
	b = spline->a;
	for (i = 0; i < n; i++) {
		h[i] = x->values[(i + 1) * x->inc] - x->values[i * x->inc];
		if (!(h[i] > 0)) {
			return luaL_argerror(L, 1, "bad order");
		}
	}
	for (i = 1; i < n; i++) {
		dl[i - 1] = h[i - 1];
		d[i] = 2 * (h[i - 1] + h[i]);
		du[i] = h[i];
		b[i] = 3 * ((y->values[(i + 1) * y->inc] - y->values[i * y->inc]) / h[i]
				- (y->values[i * y->inc] - y->values[(i - 1) * y->inc]) / h[i- 1]);
	}
	switch (boundary) {
	case 0:  /* not-a-knot*/
		d[0] = h[0] - (h[1] * h[1]) / h[0];
		du[0] = 3 * h[1] + 2 * h[0] + (h[1] * h[1]) / h[0];
		b[0] = 3 * ((y->values[2 * y->inc] - y->values[1 * y->inc]) / h[1]
				- (y->values[1 * y->inc] - y->values[0 * y->inc]) / h[0]);
		dl[n - 1] = 3 * h[n - 2] + 2 * h[n - 1] + (h[n - 2] * h[n - 2]) / h[n - 1];
		d[n] = h[n - 1] - (h[n - 2] * h[n - 2]) / h[n - 1];
		b[n] = 3 * ((y->values[n * y->inc] - y->values[(n - 1) * y->inc]) / h[n - 1]
				- (y->values[(n - 1) * y->inc] - y->values[(n - 2) * y->inc])
				/ h[n - 2]);
		break;

	case 1:  /* natural */
		d[0] = 1;
		du[0] = 0;
		b[0] = 0;
		dl[n - 1] = 0;
		d[n] = 1;
		b[n] = 0;
		break;

	case 2:  /* clamped */
		d[0] = 2 * h[0];
		du[0] = h[0];
		b[0] = 3 * ((y->values[1 * y->inc] - y->values[0 * y->inc]) / h[0] - da);
		dl[n - 1] = h[n - 1];
		d[n] = 2 * h[n - 1];
		b[n] = 3 * (db - (y->values[n * y->inc] - y->values[(n - 1) * y->inc])
				/ h[n - 1]);
		break;
	}

	/* solve the tridiagonal system */
	if (LAPACKE_dgtsv(LAPACK_ROW_MAJOR, n + 1, 1, dl, d, du, b, 1) != 0) {
		return luaL_error(L, "internal error");
	}

	/* make polynomials */
	for (i = 0; i < n; i++) {
		spline->b[i] = (y->values[(i + 1) * y->inc] - y->values[i * y->inc]) / h[i]
				- (2 * b[i] + b[i + 1]) * h[i] / 3;
		spline->c[i] = b[i];
		spline->d[i] = (b[i + 1] - b[i]) / (3 * h[i]);  /* overwrites h[i] */
		spline->x[i] = x->values[i * x->inc];
		spline->a[i] = y->values[i * y->inc];  /* overwrites b[i] */
	}
	spline->x[n] = x->values[n * x->inc];
	spline->a[n] = y->values[n * y->inc];

	/* return interpolant */
	lua_pushcclosure(L, linear_interpolant, 1);
	return 1;
}

int linear_open_program  (lua_State *L) {
	static const luaL_Reg FUNCTIONS[] = {
		{"dot", linear_dot},
		{"ger", linear_ger},
		{"gemv", linear_gemv},
		{"gemm", linear_gemm},
		{"gesv", linear_gesv},
		{"gels", linear_gels},
		{"inv", linear_inv},
		{"det", linear_det},
		{"cov", linear_cov},
		{"corr", linear_corr},
		{"ranks", linear_ranks},
		{"quantile", linear_quantile},
		{"rank", linear_rank},
		{"spline", linear_spline},
		{ NULL, NULL }
	};
#if LUA_VERSION_NUM >= 502
	luaL_setfuncs(L, FUNCTIONS, 0);
#else
	const luaL_Reg  *reg;

	for (reg = FUNCTIONS; reg->name; reg++) {
		lua_pushcfunction(L, reg->func);
		lua_setfield(L, -2, reg->name);
	}
#endif
	return 0;
}
