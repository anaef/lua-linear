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


static int linear_comparison_handler(const void *a, const void *b);
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


static linear_param_t LINEAR_PARAMS_NONE[] = {
	LINEAR_PARAMS_LAST
};
static linear_param_t LINEAR_PARAMS_DDOF[] = {
	{'d', {.d = 0}},
	LINEAR_PARAMS_LAST
};
static const char *LINEAR_SETS[] = {"p", "s", NULL};
static linear_param_t LINEAR_PARAMS_SET[] = {
	{'e', {.e = LINEAR_SETS}},
	LINEAR_PARAMS_LAST
};
static linear_param_t LINEAR_PARAMS_LUA[] = {
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

static int linear_comparison_handler (const void *a, const void *b) {
	double  da, db;

	da = *(const double *)a;
	db = *(const double *)b;
	return da < db ? -1 : (da > db ? 1 : 0);
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
	return linear_unary(L, linear_sum_handler, LINEAR_PARAMS_NONE);
}

static double linear_mean_handler (int size, double *x, int incx, linear_arg_u *args) {
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
	return sum / size;
}

static int linear_mean (lua_State *L) {
	return linear_unary(L, linear_mean_handler, LINEAR_PARAMS_NONE);
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
	return linear_unary(L, linear_var_handler, LINEAR_PARAMS_DDOF);
}

static double linear_std_handler (int size, double *x, int incx, linear_arg_u *args) {
	return sqrt(linear_var_handler(size, x, incx, args));
}

static int linear_std (lua_State *L) {
	return linear_unary(L, linear_std_handler, LINEAR_PARAMS_DDOF);
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
	return linear_unary(L, linear_skew_handler, LINEAR_PARAMS_SET);
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
	return linear_unary(L, linear_kurt_handler, LINEAR_PARAMS_SET);
}

static double linear_median_handler (int size, double *x, int incx, linear_arg_u *args) {
	int      i, mid;
	double  *copy, median;

	copy = malloc(size * sizeof(double));
	if (copy == NULL) {
		return luaL_error(args[0].L, "cannot allocate values");
	}
	if (incx == 1) {
		for (i = 0; i < size; i++) {
			copy[i] = x[i];
		}
	} else {
		for (i = 0; i < size; i++) {
			copy[i] = *x;
			x += incx;
		}
	}
	qsort(copy, size, sizeof(double), linear_comparison_handler);
	mid = size / 2;
	if (size % 2 == 0) {
		median = (copy[mid - 1] + copy[mid]) / 2;
	} else {
		median = copy[mid];
	}
	free(copy);
	return median;
}

static int linear_median (lua_State *L) {
	return linear_unary(L, linear_median_handler, LINEAR_PARAMS_LUA);
}

static double linear_mad_handler (int size, double *x, int incx, linear_arg_u *args) {
	int      i, mid;
	double  *copy, median, mad;

	/* calculat the median */
	copy = malloc(size * sizeof(double));
	if (copy == NULL) {
		return luaL_error(args[0].L, "cannot allocate values");
	}
	if (incx == 1) {
		for (i = 0; i < size; i++) {
			copy[i] = x[i];
		}
	} else {
		for (i = 0; i < size; i++) {
			copy[i] = *x;
			x += incx;
		}
	}
	qsort(copy, size, sizeof(double), linear_comparison_handler);
	mid = size / 2;
	if (size % 2 == 0) {
		median = (copy[mid - 1] + copy[mid]) / 2;
	} else {
		median = copy[mid];
	}

	/* calculate the median absolute deviation */
	for (i = 0; i < size; i++) {
		copy[i] = fabs(copy[i] - median);
	}
	qsort(copy, size, sizeof(double), linear_comparison_handler);
	if (size % 2 == 0) {
		mad = (copy[mid - 1] + copy[mid]) / 2;
	} else {
		mad = copy[mid];
	}
	free(copy);
	return mad;
}

static int linear_mad (lua_State *L) {
	return linear_unary(L, linear_mad_handler, LINEAR_PARAMS_LUA);
}

static double linear_nrm2_handler (int size, double *x, int incx, linear_arg_u *args) {
	(void)args;
	return cblas_dnrm2(size, x, incx);
}

static int linear_nrm2 (lua_State *L) {
	return linear_unary(L, linear_nrm2_handler, LINEAR_PARAMS_NONE);
}

static double linear_asum_handler (int size, double *x, int incx, linear_arg_u *args) {
	(void)args;
	return cblas_dasum(size, x, incx);
}

static int linear_asum (lua_State *L) {
	return linear_unary(L, linear_asum_handler, LINEAR_PARAMS_NONE);
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
	return linear_unary(L, linear_min_handler, LINEAR_PARAMS_NONE);
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
	return linear_unary(L, linear_max_handler, LINEAR_PARAMS_NONE);
}

int linear_open_unary (lua_State *L) {
	static const luaL_Reg FUNCTIONS[] = {
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
