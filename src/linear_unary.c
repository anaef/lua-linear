/*
 * Lua Linear unary vector functions
 *
 * Copyright (C) 2017-2023 Andre Naef
 */


#include <stdlib.h>
#include <math.h>
#include <lauxlib.h>
#include "linear_core.h"
#include "linear_unary.h"


#if LUA_VERSION_NUM < 502
#define luaL_testudata  linear_testudata
#endif


static double linear_sum_handler(int size, double *values, int inc, linear_arg_u *args);
static int linear_sum(lua_State *L);
static double linear_mean_handler(int size, double *values, int inc, linear_arg_u *args);
static int linear_mean(lua_State *L);
static double linear_var_handler(int size, double *values, int inc, linear_arg_u *args);
static int linear_var(lua_State *L);
static double linear_std_handler(int size, double *values, int inc, linear_arg_u *args);
static int linear_std(lua_State *L);
static double linear_skew_handler(int size, double *values, int inc, linear_arg_u *args);
static int linear_skew(lua_State *L);
static double linear_kurt_handler(int size, double *values, int inc, linear_arg_u *args);
static int linear_kurt(lua_State *L);
static double linear_nrm2_handler(int size, double *values, int inc, linear_arg_u *args);
static int linear_median(lua_State *L);
static double linear_median_handler(int size, double *values, int inc, linear_arg_u *args);
static int linear_mad(lua_State *L);
static double linear_mad_handler(int size, double *values, int inc, linear_arg_u *args);
static int linear_nrm2(lua_State *L);
static double linear_asum_handler(int size, double *values, int inc, linear_arg_u *args);
static int linear_asum(lua_State *L);
static double linear_min_handler(int size, double *x, int incx, linear_arg_u *args);
static int linear_min(lua_State *L);
static double linear_max_handler(int size, double *x, int incx, linear_arg_u *args);
static int linear_max(lua_State *L);


static linear_param_t linear_params_none[] = {
	LINEAR_PARAMS_LAST
};
static linear_param_t linear_params_ddof[] = {
	{'d', {.d = 0}},
	LINEAR_PARAMS_LAST
};
static const char *linear_sets[] = {"p", "s", NULL};
static linear_param_t linear_params_set[] = {
	{'e', {.e = linear_sets}},
	LINEAR_PARAMS_LAST
};
static linear_param_t linear_params_lua[] = {
	{'L', {0.0}},
	LINEAR_PARAMS_LAST
};


int linear_unary (lua_State *L, linear_unary_function f, linear_param_t *params) {
	size_t            i;
	linear_arg_u      args[LINEAR_PARAMS_MAX];
	linear_vector_t  *x, *y;
	linear_matrix_t  *X;

	x = luaL_testudata(L, 1, LINEAR_VECTOR);
	if (x != NULL) {
		/* vector */
		linear_checkargs(L, 2, x->length, params, args);
		lua_pushnumber(L, f(x->length, x->values, x->inc, args));
		return 1;
	}
	X = luaL_testudata(L, 1, LINEAR_MATRIX);
	if (X != NULL) {
		/* matrix-vector */
		y = luaL_checkudata(L, 2, LINEAR_VECTOR);
		if (linear_checkorder(L, 3) == CblasRowMajor) {
			luaL_argcheck(L, y->length == X->rows, 2, "dimension mismatch");
			linear_checkargs(L, 4, X->cols, params, args);
			if (X->order == CblasRowMajor) {
				for (i = 0; i < X->rows; i++) {
					y->values[i * y->inc] = f(X->cols, &X->values[i * X->ld],
							1, args);
				}
			} else {
				for (i = 0; i < X->rows; i++) {
					y->values[i * y->inc] = f(X->cols, &X->values[i], X->ld,
							args);
				}
			}
		} else {
			luaL_argcheck(L, y->length == X->cols, 2, "dimension mismatch");
			linear_checkargs(L, 4, X->rows, params, args);
			if (X->order == CblasColMajor) {
				for (i = 0; i < X->cols; i++) {
					y->values[i * y->inc] = f(X->rows, &X->values[i * X->ld],
							1, args);
				}
			} else {
				for (i = 0; i < X->cols; i++) {
					y->values[i * y->inc] = f(X->rows, &X->values[i], X->ld,
							args);
				}
			}
		}
		return 0;
	}
	return linear_argerror(L, 1, 0);
}

static double linear_sum_handler (int size, double *x, int incx, linear_arg_u *args) {
	int     i;
	double  sum;

	(void)args;
	sum = 0.0;
	if (incx == 1) {
		for (i = 0; i < size; i++) {
			sum += x[i];
		}
	} else {
		for (i = 0; i < size; i++) {
			sum += *x;
			x += incx;
		}
	}
	return sum;
}

static int linear_sum (lua_State *L) {
	return linear_unary(L, linear_sum_handler, linear_params_none);
}

static double linear_mean_handler (int size, double *x, int incx, linear_arg_u *args) {
	return linear_sum_handler(size, x, incx, args) / size;
}

static int linear_mean (lua_State *L) {
	return linear_unary(L, linear_mean_handler, linear_params_none);
}

static double linear_var_handler (int size, double *x, int incx, linear_arg_u *args) {
	int     i;
	double  sum, mean;

	sum = 0.0;
	if (incx == 1) {
		for (i = 0; i < size; i++) {
			sum += x[i];
		}
		mean = sum / size;
		sum = 0.0;
		for (i = 0; i < size; i++) {
			sum += (x[i] - mean) * (x[i] - mean);
		}
	} else {
		for (i = 0; i < size; i++) {
			sum += *x;
			x += incx;
		}
		mean = sum / size;
		x -= (size_t)size * (size_t)incx;
		sum = 0.0;
		for (i = 0; i < size; i++) {
			sum += (*x - mean) * (*x - mean);
			x += incx;
		}
	}
	return sum / (size - args[0].d);
}

static int linear_var (lua_State *L) {
	return linear_unary(L, linear_var_handler, linear_params_ddof);
}

static double linear_std_handler (int size, double *x, int incx, linear_arg_u *args) {
	return sqrt(linear_var_handler(size, x, incx, args));
}

static int linear_std (lua_State *L) {
	return linear_unary(L, linear_std_handler, linear_params_ddof);
}

static double linear_skew_handler (int size, double *x, int incx, linear_arg_u *args) {
	int     i;
	double  sum, mean, m3, m2, skew;

	sum = 0.0;
	m3 = 0.0;
	m2 = 0.0;
	if (incx == 1) {
		for (i = 0; i < size; i++) {
			sum += x[i];
		}
		mean = sum / size;
		for (i = 0; i < size; i++) {
			m3 += pow(x[i] - mean, 3);
			m2 += (x[i] - mean) * (x[i] - mean);
		}
	} else {
		for (i = 0; i < size; i++) {
			sum += *x;
			x += incx;
		}
		mean = sum / size;
		x -= (size_t)size * (size_t)incx;
		for (i = 0; i < size; i++) {
			m3 += pow(*x - mean, 3);
			m2 += (*x - mean) * (*x - mean);
			x += incx;
		}
	}
	m3 /= size;
	m2 /= size;
	skew = m3 / pow(m2, 1.5);
	if (args[0].e == 1) {
		skew *= sqrt((double)size * (size - 1)) / (size - 2);
	}
	return skew;
}

static int linear_skew (lua_State *L) {
	return linear_unary(L, linear_skew_handler, linear_params_set);
}

static double linear_kurt_handler (int size, double *x, int incx, linear_arg_u *args) {
	int     i;
	double  sum, mean, m4, m2, kurt;

	sum = 0.0;
	m4 = 0.0;
	m2 = 0.0;
	if (incx == 1) {
		for (i = 0; i < size; i++) {
			sum += x[i];
		}
		mean = sum / size;
		for (i = 0; i < size; i++) {
			m4 += pow(x[i] - mean, 4);
			m2 += (x[i] - mean) * (x[i] - mean);
		}
	} else {
		for (i = 0; i < size; i++) {
			sum += *x;
			x += incx;
		}
		mean = sum / size;
		x -= (size_t)size * (size_t)incx;
		for (i = 0; i < size; i++) {
			m4 += pow(*x - mean, 4);
			m2 += (*x - mean) * (*x - mean);
			x += incx;
		}
	}
	m4 /= size;
	m2 /= size;
	kurt = m4 / (m2 * m2) - 3;  /* excess kurtosis */
	if (args[0].e == 1) {
		kurt = ((double)(size - 1) / ((size - 2) * (size - 3))) * ((size + 1) * kurt + 6);
	}
	return kurt;
}

static int linear_kurt (lua_State *L) {
	return linear_unary(L, linear_kurt_handler, linear_params_set);
}

static double linear_median_handler (int size, double *x, int incx, linear_arg_u *args) {
	int      i, mid;
	double  *s, median;

	s = malloc(size * sizeof(double));
	if (s == NULL) {
		return luaL_error(args[0].L, "cannot allocate components");
	}
	for (i = 0; i < size; i++) {
		if (isnan(*x)) {
			free(s);
			return NAN;
		}
		s[i] = *x;
		x += incx;
	}
	qsort(s, size, sizeof(double), linear_comparison_handler);
	mid = size / 2;
	if (size % 2 == 0) {
		median = (s[mid - 1] + s[mid]) / 2;
	} else {
		median = s[mid];
	}
	free(s);
	return median;
}

static int linear_median (lua_State *L) {
	return linear_unary(L, linear_median_handler, linear_params_lua);
}

static double linear_mad_handler (int size, double *x, int incx, linear_arg_u *args) {
	int      i, mid;
	double  *s, median, mad;

	/* calculate the median */
	s = malloc(size * sizeof(double));
	if (s == NULL) {
		return luaL_error(args[0].L, "cannot allocate components");
	}
	for (i = 0; i < size; i++) {
		if (isnan(*x)) {
			free(s);
			return NAN;
		}
		s[i] = *x;
		x += incx;
	}
	qsort(s, size, sizeof(double), linear_comparison_handler);
	mid = size / 2;
	if (size % 2 == 0) {
		median = (s[mid - 1] + s[mid]) / 2;
	} else {
		median = s[mid];
	}

	/* calculate the median absolute deviation */
	for (i = 0; i < size; i++) {
		s[i] = fabs(s[i] - median);
	}
	qsort(s, size, sizeof(double), linear_comparison_handler);
	if (size % 2 == 0) {
		mad = (s[mid - 1] + s[mid]) / 2;
	} else {
		mad = s[mid];
	}
	free(s);
	return mad;
}

static int linear_mad (lua_State *L) {
	return linear_unary(L, linear_mad_handler, linear_params_lua);
}

static double linear_nrm2_handler (int size, double *x, int incx, linear_arg_u *args) {
	(void)args;
	return cblas_dnrm2(size, x, incx);
}

static int linear_nrm2 (lua_State *L) {
	return linear_unary(L, linear_nrm2_handler, linear_params_none);
}

static double linear_asum_handler (int size, double *x, int incx, linear_arg_u *args) {
	(void)args;
	return cblas_dasum(size, x, incx);
}

static int linear_asum (lua_State *L) {
	return linear_unary(L, linear_asum_handler, linear_params_none);
}

static double linear_min_handler (int size, double *x, int incx, linear_arg_u *args) {
	int     i;
	double  min;

	(void)args;
	min = *x;
	for (i = 1; i < size; i++) {
		x += incx;
		if (*x < min) {
			min = *x;
		}
	}
	return min;
}

static int linear_min (lua_State *L) {
	return linear_unary(L, linear_min_handler, linear_params_none);
}

static double linear_max_handler (int size, double *x, int incx, linear_arg_u *args) {
	int     i;
	double  max;

	(void)args;
	max = *x;
	for (i = 1; i < size; i++) {
		x += incx;
		if (*x > max) {
			max = *x;
		}
	}
	return max;
}

static int linear_max (lua_State *L) {
	return linear_unary(L, linear_max_handler, linear_params_none);
}

int linear_open_unary (lua_State *L) {
	static const luaL_Reg functions[] = {
		{"sum", linear_sum},
		{"mean", linear_mean},
		{"var", linear_var},
		{"std", linear_std},
		{"skew", linear_skew},
		{"kurt", linear_kurt},
		{"median", linear_median},
		{"mad", linear_mad},
		{"nrm2", linear_nrm2},
		{"asum", linear_asum},
		{"min", linear_min},
		{"max", linear_max},
		{NULL, NULL}
	};
#if LUA_VERSION_NUM >= 502
	luaL_setfuncs(L, functions, 0);
#else
	const luaL_Reg  *reg;

	for (reg = functions; reg->name; reg++) {
		lua_pushcfunction(L, reg->func);
		lua_setfield(L, -2, reg->name);
	}
#endif
	return 0;
}
