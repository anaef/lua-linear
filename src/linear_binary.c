/*
 * Lua Linear binary vector functions
 *
 * Copyright (C) 2017-2023 Andre Naef
 */


#include <math.h>
#include <lauxlib.h>
#include "linear_core.h"
#include "linear_binary.h"


#if LUA_VERSION_NUM < 502
#define luaL_testudata  linear_testudata
#endif


static void linear_axpy_handler(int size, double *x, int incx, double *y, int incy,
		linear_arg_u *args);
static int linear_axpy(lua_State *L);
static void linear_axpby_handler(int size, double *x, int incx, double *y, int incy,
		linear_arg_u *args);
static int linear_axpby(lua_State *L);
static void linear_mul_handler(int size, double *x, int incx, double *y, int incy,
		linear_arg_u *args);
static int linear_mul(lua_State *L);
static void linear_swap_handler(int size, double *x, int incx, double *y, int incy,
		linear_arg_u *args);
static int linear_swap(lua_State *L);
static void linear_copy_handler(int size, double *x, int incx, double *y, int incy,
		linear_arg_u *args);
static int linear_copy(lua_State *L);


static linear_param_t LINEAR_PARAMS_NONE[] = {
	LINEAR_PARAMS_LAST
};
static linear_param_t LINEAR_PARAMS_ALPHA[] = {
	{'n', {.n = 1.0}},
	LINEAR_PARAMS_LAST
};
static linear_param_t LINEAR_PARAMS_ALPHA_BETA[] = {
	{'n', {.n = 1.0}},
	{'n', {.n = 1.0}},
	LINEAR_PARAMS_LAST
};


int linear_binary (lua_State *L, linear_binary_function f, linear_param_t *params) {
	size_t                 i;
	linear_arg_u       args[LINEAR_PARAMS_MAX];
	linear_vector_t  *x, *y;
	linear_matrix_t  *X, *Y;

	x = luaL_testudata(L, 1, LINEAR_VECTOR);
	if (x != NULL) {
		y = luaL_testudata(L, 2, LINEAR_VECTOR);
		if (y != NULL) {
			/* vector-vector */
			luaL_argcheck(L, y->length == x->length, 2, "dimension mismatch");
			linear_checkargs(L, 3, x->length, params, args);
			f(x->length, x->values, x->inc, y->values, y->inc, args);
			return 0;
		}
		Y = luaL_testudata(L, 2, LINEAR_MATRIX);
		if (Y != NULL) {
			/* vector-matrix */
			linear_checkargs(L, 4, x->length, params, args);
			if (linear_checkorder(L, 3) == CblasRowMajor) {
				luaL_argcheck(L, x->length == Y->cols, 1, "dimension mismatch");
				if (Y->order == CblasRowMajor) {
					for (i = 0; i < Y->rows; i++) {
						f(x->length, x->values, x->inc,
								&Y->values[i * Y->ld], 1, args);
					}
				} else {
					for (i = 0; i < Y->rows; i++) {
						f(x->length, x->values, x->inc, 
								&Y->values[i], Y->ld, args);
					}
				}
			} else {
				luaL_argcheck(L, x->length == Y->rows, 1, "dimension mismatch");
				if (Y->order == CblasColMajor) {
					for (i = 0; i < Y->cols; i++) {
						f(x->length, x->values, x->inc,
								&Y->values[i * Y->ld], 1, args);
					}
				} else {
					for (i = 0; i < Y->cols; i++) {
						f(x->length, x->values, x->inc,
								&Y->values[i], Y->ld, args);
					}
				}
			}
			return 0;
		}
		return linear_argerror(L, 2, 0);
	}
	X = luaL_testudata(L, 1, LINEAR_MATRIX);
	if (X != NULL) {
		/* matrix-matrix */
		Y = luaL_checkudata(L, 2, LINEAR_MATRIX);
		luaL_argcheck(L, X->order == Y->order, 2, "order mismatch");
		luaL_argcheck(L, X->rows == Y->rows && X->cols == Y->cols, 2, "dimension mismatch");
		if (X->order == CblasRowMajor) {
			linear_checkargs(L, 3, X->cols, params, args);
			if (X->ld == X->cols && Y->ld == Y->cols && X->rows * X->cols <= INT_MAX) {
				f(X->rows * X->cols, X->values, 1, Y->values, 1, args);
			} else {
				for (i = 0; i < X->rows; i++) {
					f(X->cols, &X->values[i * X->ld], 1, &Y->values[i * Y->ld],
							1, args);
				}
			}
		} else {
			linear_checkargs(L, 3, X->rows, params, args);
			if (X->ld == X->rows && Y->ld == Y->rows && X->cols * X->rows <= INT_MAX) {
				f(X->cols * X->rows, X->values, 1, Y->values, 1, args);
			} else {
				for (i = 0; i < X->cols; i++) {
					f(X->rows, &X->values[i * X->ld], 1, &Y->values[i * Y->ld],
							1, args);
				}
			}
		}
		return 0;
	}
	return linear_argerror(L, 1, 0);
}

static void linear_axpy_handler (int size, double *x, int incx, double *y, int incy,
		linear_arg_u *args) {
	cblas_daxpy(size, args[0].n, x, incx, y, incy);
}

static int linear_axpy (lua_State *L) {
	return linear_binary(L, linear_axpy_handler, LINEAR_PARAMS_ALPHA);
}

static void linear_axpby_handler (int size, double *x, int incx, double *y, int incy,
		linear_arg_u *args) {
#if LINEAR_USE_AXPBY
	cblas_daxpby(size, args[0].n, x, incx, args[1].n, y, incy);
#else
	if (args[1].n != 1.0) {
		cblas_dscal(size, args[1].n, y, incy);
	}
	cblas_daxpy(size, args[0].n, x, incx, y, incy);
#endif
}

static int linear_axpby (lua_State *L) {
	return linear_binary(L, linear_axpby_handler, LINEAR_PARAMS_ALPHA_BETA);
}

static void linear_mul_handler (int size, double *x, int incx, double *y, int incy,
		linear_arg_u *args) {
	int     i;
	double  alpha;

	alpha = args[0].n;
	if (alpha == 1.0) {
		if (incx == 1 && incy == 1) {
			for (i = 0; i < size; i++) {
				y[i] *= x[i];
			}
		} else {
			for (i = 0; i < size; i++) {
				*y *= *x;
				x += incx;
				y += incy;
			}
		}
	} else if (alpha == -1.0) {
		if (incx == 1 && incy == 1) {
			for (i = 0; i < size; i++) {
				y[i] /= x[i];
			}
		} else {
			for (i = 0; i < size; i++) {
				*y /= *x;
				x += incx;
				y += incy;
			}
		}
	} else if (alpha == 0.5) {
		for (i = 0; i < size; i++) {
			*y *= sqrt(*x);
			x += incx;
			y += incy;
		}
	} else if (alpha != 0.0) {
		for (i = 0; i < size; i++) {
			*y *= pow(*x, alpha);
			x += incx;
			y += incy;
		}
	}
}

static int linear_mul (lua_State *L) {
	return linear_binary(L, linear_mul_handler, LINEAR_PARAMS_ALPHA);
}

static void linear_swap_handler (int size, double *x, int incx, double *y, int incy,
		linear_arg_u *args) {
	(void)args;
	cblas_dswap(size, x, incx, y, incy);
}

static int linear_swap (lua_State *L) {
	return linear_binary(L, linear_swap_handler, LINEAR_PARAMS_NONE);
}

static void linear_copy_handler (int size, double *x, int incx, double *y, int incy,
		linear_arg_u *args) {
	(void)args;
	cblas_dcopy(size, x, incx, y, incy);
}

static int linear_copy (lua_State *L) {
	return linear_binary(L, linear_copy_handler, LINEAR_PARAMS_NONE);
}

int linear_open_binary (lua_State *L) {
	static const luaL_Reg FUNCTIONS[] = {
		{"axpy", linear_axpy},    /* deprecated */
		{"axpby", linear_axpby},
		{"mul", linear_mul},
		{"swap", linear_swap},
		{"copy", linear_copy},
		{NULL, NULL}
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
