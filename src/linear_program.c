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


static const char *const LINEAR_TRANSPOSES[] = {"notrans", "trans", NULL};


static inline CBLAS_TRANSPOSE linear_checktranspose (lua_State *L, int index) {
	return luaL_checkoption(L, index, "notrans", LINEAR_TRANSPOSES) == 0 ? CblasNoTrans
			: CblasTrans;
}

static inline char linear_lapacktranspose (CBLAS_TRANSPOSE transpose) {
	return transpose == CblasNoTrans ? 'N' : 'T';
}

static int linear_dot (lua_State *L) {
	struct linear_vector  *x, *y;

	x = luaL_checkudata(L, 1, LINEAR_VECTOR);
	y = luaL_checkudata(L, 2, LINEAR_VECTOR);
	luaL_argcheck(L, y->length == x->length, 2, "dimension mismatch");
	lua_pushnumber(L, cblas_ddot(x->length, x->values, x->inc, y->values, y->inc));
	return 1;
}

static int linear_ger (lua_State *L) {
	double                 alpha;
	struct linear_vector  *x, *y;
	struct linear_matrix  *A;

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
	size_t                  m, n;
	double                  alpha, beta;
	CBLAS_TRANSPOSE         ta;
	struct linear_matrix   *A;
	struct linear_vector   *x, *y;

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
	size_t                   m, n, ka, kb;
	double                   alpha, beta;
	CBLAS_TRANSPOSE          ta, tb;
	struct linear_matrix    *A, *B, *C;

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
	int                   *ipiv, result;
	struct linear_matrix  *A, *B;

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
	int                    result;
	char                   ta;
	struct linear_matrix  *A, *B;

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
	int                   *ipiv, result;
	struct linear_matrix  *A;

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
	int                   *ipiv, result, neg;
	size_t                 n, i;
	double                *copy, *d, *s, det;
	struct linear_matrix  *A;

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
	size_t                 i, j, k, ddof;
	double                *means, *v, *vi, *vj, sum;
	struct linear_matrix  *A, *B;

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
	size_t                 i, j, k;
	double                *means, *stds, *v, *vi, *vj, sum;
	struct linear_matrix  *A, *B;

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

int linear_open_program  (lua_State *L) {
	static const luaL_Reg FUNCTIONS[] = {
		{ "dot", linear_dot },
		{ "ger", linear_ger },
		{ "gemv", linear_gemv },
		{ "gemm", linear_gemm },
		{ "gesv", linear_gesv },
		{ "gels", linear_gels },
		{ "inv", linear_inv },
		{ "det", linear_det },
		{ "cov", linear_cov },
		{ "corr", linear_corr },
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
