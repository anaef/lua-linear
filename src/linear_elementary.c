/*
 * Lua Linear elementary functions
 *
 * Copyright (C) 2017-2023 Andre Naef
 */


#include <stdlib.h>
#include <math.h>
#include <lauxlib.h>
#include "linear_core.h"
#include "linear_elementary.h"


#if LUA_VERSION_NUM < 502
#define luaL_testudata  linear_testudata
#endif


static void linear_inc_handler(size_t size, double *x, size_t incx, linear_arg_u *args);
static int linear_inc(lua_State *L);
static void linear_scal_handler(size_t size, double *x, size_t incx, linear_arg_u *args);
static int linear_scal(lua_State *L);
static void linear_pow_handler(size_t size, double *x, size_t incx, linear_arg_u *args);
static int linear_pow(lua_State *L);
static void linear_exp_handler(size_t size, double *x, size_t incx, linear_arg_u *args);
static int linear_exp(lua_State *L);
static void linear_log_handler(size_t size, double *x, size_t incx, linear_arg_u *args);
static int linear_log(lua_State *L);
static void linear_sgn_handler(size_t size, double *x, size_t incx, linear_arg_u *args);
static int linear_sgn(lua_State *L);
static void linear_abs_handler(size_t size, double *x, size_t incx, linear_arg_u *args);
static int linear_abs(lua_State *L);
static void linear_logistic_handler(size_t size, double *x, size_t incx, linear_arg_u *args);
static int linear_logistic(lua_State *L);
static void linear_tanh_handler(size_t size, double *x, size_t incx, linear_arg_u *args);
static int linear_tanh(lua_State *L);
static void linear_apply_handler(size_t size, double *x, size_t incx, linear_arg_u *args);
static int linear_apply(lua_State *L);
static void linear_set_handler(size_t size, double *x, size_t incx, linear_arg_u *args);
static int linear_set(lua_State *L);
static void linear_clip_handler(size_t size, double *x, size_t incx, linear_arg_u *args);
static int linear_clip(lua_State *L);
static void linear_uniform_handler(size_t size, double *x, size_t incx, linear_arg_u *args);
static int linear_uniform(lua_State *L);
static void linear_normal_handler(size_t size, double *x, size_t incx, linear_arg_u *args);
static int linear_normal(lua_State *L);
static void linear_normalpdf_handler(size_t size, double *x, size_t incx, linear_arg_u *args);
static int linear_normalpdf(lua_State *L);
static void linear_normalcdf_handler(size_t size, double *x, size_t incx, linear_arg_u *args);
static int linear_normalcdf(lua_State *L);
static void linear_normalqf_handler(size_t size, double *x, size_t incx, linear_arg_u *args);
static int linear_normalqf(lua_State *L);


static linear_param_t linear_params_none[] = {
	LINEAR_PARAMS_LAST
};
static linear_param_t linear_params_alpha[] = {
	{'n', {.n = 1.0}},
	LINEAR_PARAMS_LAST
};
static linear_param_t linear_params_min_max[] = {
	{'n', {.n = 0.0}},
	{'n', {.n = 1.0}},
	LINEAR_PARAMS_LAST
};
static linear_param_t linear_params_random[] = {
	{'r', {0.0}},
	LINEAR_PARAMS_LAST
};
static linear_param_t linear_params_mu_sigma[] = {
	{'n', {.n = 0.0}},
	{'n', {.n = 1.0}},
	LINEAR_PARAMS_LAST
};

static __thread lua_State  *linear_TL;


int linear_elementary (lua_State *L, linear_elementary_function f, linear_param_t *params) {
	int               isnum;
	size_t            i;
	double            n;
	linear_arg_u      args[LINEAR_PARAMS_MAX];
	linear_vector_t  *x;
	linear_matrix_t  *X;

#if LUA_VERSION_NUM >= 502
	n = lua_tonumberx(L, 1, &isnum);
#else
	isnum = lua_isnumber(L, 1);
	n = isnum ? lua_tonumber(L, 1) : 0.0;
#endif
	if (isnum) {
		linear_checkargs(L, 2, 1, params, args);
		f(1, &n, 1, args);
		lua_pushnumber(L, n);
		return 1;
	}
	x = luaL_testudata(L, 1, LINEAR_VECTOR);
	if (x != NULL) {
		linear_checkargs(L, 2, x->length, params, args);
		f(x->length, x->values, x->inc, args);
		return 0;
	}
	X = luaL_testudata(L, 1, LINEAR_MATRIX);
	if (X != NULL) {
		if (X->order == CblasRowMajor) {
			linear_checkargs(L, 2, X->cols, params, args);
			if (X->cols == X->ld && X->rows * X->cols <= INT_MAX) {
				f(X->rows * X->cols, X->values, 1, args);
			} else {
				for (i = 0; i < X->rows; i++) {
					f(X->cols, &X->values[i * X->ld], 1, args);
				}
			}
		} else {
			linear_checkargs(L, 2, X->rows, params, args);
			if (X->rows == X->ld && X->cols * X->rows <= INT_MAX) {
				f(X->cols * X->rows, X->values, 1, args);
			} else {
				for (i = 0; i < X->cols; i++) {
					f(X->rows, &X->values[i * X->ld], 1, args);
				}
			}
		}
		return 0;
	}
	return linear_argerror(L, 0, 1);
}

static void linear_inc_handler (size_t size, double *x, size_t incx, linear_arg_u *args) {
	size_t  i;
	double  alpha;

	alpha = args[0].n;
	if (incx == 1) {
		for (i = 0; i < size; i++) {
			x[i] += alpha;
		}
	} else {
		for (i = 0; i < size; i++) {
			*x += alpha;
			x += incx;
		}
	}
}

static int linear_inc (lua_State *L) {
	return linear_elementary(L, linear_inc_handler, linear_params_alpha);
}

static void linear_scal_handler (size_t size, double *x, size_t incx, linear_arg_u *args) {
	cblas_dscal(size, args[0].n, x, incx);
}

static int linear_scal (lua_State *L) {
	return linear_elementary(L, linear_scal_handler, linear_params_alpha);
}

static void linear_pow_handler (size_t size, double *x, size_t incx, linear_arg_u *args) {
	size_t  i;
	double  alpha;

	alpha = args[0].n;
	if (alpha == -1.0) {
		if (incx == 1) {
			for (i = 0; i < size; i++) {
				x[i] = 1 / x[i];
			}
		} else {
			for (i = 0; i < size; i++) {
				*x = 1 / *x;
				x += incx;
			}
		}
	} else if (alpha == 0.0) {
		if (incx == 1) {
			for (i = 0; i < size; i++) {
				x[i] = 1.0;
			}
		} else {
			for (i = 0; i < size; i++) {
				*x = 1.0;
				x += incx;
			}
		}
	} else if (alpha == 0.5) {
		for (i = 0; i < size; i++) {
			*x = sqrt(*x);
			x += incx;
		}
	} else if (alpha != 1.0) {
		for (i = 0; i < size; i++) {
			*x = pow(*x, alpha);
			x += incx;
		}
	}
}

static int linear_pow (lua_State *L) {
	return linear_elementary(L, linear_pow_handler, linear_params_alpha);
}

static void linear_exp_handler (size_t size, double *x, size_t incx, linear_arg_u *args) {
	size_t  i;

	(void)args;
	if (incx == 1) {
		for (i = 0; i < size; i++) {
			x[i] = exp(x[i]);
		}
	} else {
		for (i = 0; i < size; i++) {
			*x = exp(*x);
			x += incx;
		}
	}
}

static int linear_exp (lua_State *L) {
	return linear_elementary(L, linear_exp_handler, linear_params_none);
}

static void linear_log_handler (size_t size, double *x, size_t incx, linear_arg_u *args) {
	size_t  i;

	(void)args;
	if (incx == 1) {
		for (i = 0; i < size; i++) {
			x[i] = log(x[i]);
		}
	} else {
		for (i = 0; i < size; i++) {
			*x = log(*x);
			x += incx;
		}
	}
}

static int linear_log (lua_State *L) {
	return linear_elementary(L, linear_log_handler, linear_params_none);
}

static void linear_sgn_handler (size_t size, double *x, size_t incx, linear_arg_u *args) {
	size_t  i;

	(void)args;
	for (i = 0; i < size; i++) {
		if (*x > 0) {
			*x = 1;
		} else if (*x < 0) {
			*x = -1;
		}
		x += incx;
	}
}

static int linear_sgn (lua_State *L) {
	return linear_elementary(L, linear_sgn_handler, linear_params_none);
}

static void linear_abs_handler (size_t size, double *x, size_t incx, linear_arg_u *args) {
	size_t  i;

	(void)args;
	for (i = 0; i < size; i++) {
		*x = fabs(*x);
		x += incx;
	}
}

static int linear_abs (lua_State *L) {
	return linear_elementary(L, linear_abs_handler, linear_params_none);
}

static void linear_logistic_handler (size_t size, double *x, size_t incx, linear_arg_u *args) {
	size_t  i;

	(void)args;
	for (i = 0; i < size; i++) {
		*x = 1.0 / (1.0 + exp(-*x));
		x += incx;
	}
}

static int linear_logistic (lua_State *L) {
	return linear_elementary(L, linear_logistic_handler, linear_params_none);
}

static void linear_tanh_handler (size_t size, double *x, size_t incx, linear_arg_u *args) {
	size_t  i;

	(void)args;
	for (i = 0; i < size; i++) {
		*x = tanh(*x);
		x += incx;
	}
}

static int linear_tanh (lua_State *L) {
	return linear_elementary(L, linear_tanh_handler, linear_params_none);
}

static void linear_apply_handler (size_t size, double *x, size_t incx, linear_arg_u *args) {
	size_t  i;

	(void)args;
	for (i = 0; i < size; i++) {
		lua_pushvalue(linear_TL, -1);
		lua_pushnumber(linear_TL, *x);
		lua_call(linear_TL, 1, 1);
		*x = lua_tonumber(linear_TL, -1);
		x += incx;
		lua_pop(linear_TL, 1);
	}
}

static int linear_apply (lua_State *L) {
	luaL_checktype(L, 2, LUA_TFUNCTION);
	lua_settop(L, 2);
	linear_TL = L;
	return linear_elementary(L, linear_apply_handler, linear_params_none);
}

static void linear_set_handler (size_t size, double *x, size_t incx, linear_arg_u *args) {
	size_t  i;
	double  alpha;

	alpha = args[0].n;
	if (incx == 1) {
		for (i = 0; i < size; i++) {
			x[i] = alpha;
		}
	} else {
		for (i = 0; i < size; i++) {
			*x = alpha;
			x += incx;
		}
	}
}

static int linear_set (lua_State *L) {
	return linear_elementary(L, linear_set_handler, linear_params_alpha);
}

static void linear_clip_handler (size_t size, double *x, size_t incx, linear_arg_u *args) {
	size_t  i;
	double  min, max;

	min = args[0].n;
	max = args[1].n;
	for (i = 0; i < size; i++) {
		if (*x < min) {
			*x = min;
		} else if (*x > max) {
			*x = max;
		}
		x += incx;
	}
}

static int linear_clip (lua_State *L) {
	return linear_elementary(L, linear_clip_handler, linear_params_min_max);
}

static void linear_uniform_handler (size_t size, double *x, size_t incx, linear_arg_u *args) {
	size_t     i;
	uint64_t  *r;

	r = args[0].r;
	for (i = 0; i < size; i++) {
		*x = linear_random(r);
		x += incx;
	}
}

static int linear_uniform (lua_State *L) {
	return linear_elementary(L, linear_uniform_handler, linear_params_random);
}

static void linear_normal_handler (size_t size, double *x, size_t incx, linear_arg_u *args) {
	size_t     i;
	double     u1, u2, r, s, c;
	uint64_t  *rs;

	rs = args[0].r;

	/* Box-Muller transform */
	for (i = 0; i < size - 1; i += 2) {
		u1 = linear_random(rs);
		u2 = linear_random(rs);
		r = sqrt(-2.0 * log(1 - u1));
		sincos(2 * M_PI * u2, &s, &c);
		*x = r * c;
		x += incx;
		*x = r * s;
		x += incx;
	}
	if (i < size) {
		u1 = linear_random(rs);
		u2 = linear_random(rs);
		*x = sqrt(-2.0 * log(1 - u1)) * cos(2 * M_PI * u2);
	}
}

static int linear_normal (lua_State *L) {
	return linear_elementary(L, linear_normal_handler, linear_params_random);
}

static void linear_normalpdf_handler (size_t size, double *x, size_t incx, linear_arg_u *args) {
	size_t  i;
	double  mu, sigma;

	mu = args[0].n;
	sigma = args[1].n;
	for (i = 0; i < size; i++) {
		*x = (M_2_SQRTPI / (sigma * M_SQRT2 * 2)) * exp(-0.5 * ((*x - mu) / sigma)
				* ((*x - mu) / sigma));
		x += incx;
	}
}

static int linear_normalpdf (lua_State *L) {
	return linear_elementary(L, linear_normalpdf_handler, linear_params_mu_sigma);
}

static void linear_normalcdf_handler (size_t size, double *x, size_t incx, linear_arg_u *args) {
	size_t  i;
	double  mu, sigma;

	mu = args[0].n;
	sigma = args[1].n;
	for (i = 0; i < size; i++) {
		*x = 0.5 * (1 + erf((*x - mu) / (sigma * M_SQRT2)));
		x += incx;
	}
}

static int linear_normalcdf (lua_State *L) {
	return linear_elementary(L, linear_normalcdf_handler, linear_params_mu_sigma);
}

static void linear_normalqf_handler (size_t size, double *x, size_t incx, linear_arg_u *args) {
	size_t  i;
	double  mu, sigma, p, inverf, inverf_prev, f, fx;

	mu = args[0].n;
	sigma = args[1].n;
	for (i = 0; i < size; i++) {
		p = 2 * *x - 1;
		if (p < -1 || p > 1) {
			inverf = NAN;
		} else if (p == -1) {
			inverf = -INFINITY;
		} else if (p == 1) {
			inverf = INFINITY;
		} else {
			/* Newton-Raphson; ~ 4-8 iterations */
			inverf = sqrt(-log((1 - p) * (1 + p))) * (p >= 0 ? 1 : -1);
			do {
				inverf_prev = inverf;
				f = erf(inverf) - p;
				fx = M_2_SQRTPI * exp(-(inverf * inverf));
				inverf -= f / fx;
			} while (fabs(inverf - inverf_prev) > 1e-16);
		}
		*x = mu + sigma * M_SQRT2 * inverf;
		x += incx;
	}
}

static int linear_normalqf (lua_State *L) {
	return linear_elementary(L, linear_normalqf_handler, linear_params_mu_sigma);
}

int linear_open_elementary (lua_State *L) {
	static const luaL_Reg functions[] = {
		{"inc", linear_inc},
		{"scal", linear_scal},
		{"pow", linear_pow},
		{"exp", linear_exp},
		{"log", linear_log},
		{"sgn", linear_sgn},
		{"abs", linear_abs},
		{"logistic", linear_logistic},
		{"tanh", linear_tanh},
		{"apply", linear_apply},
		{"set", linear_set},
		{"clip", linear_clip},
		{"uniform", linear_uniform},
		{"normal", linear_normal},
		{"normalpdf", linear_normalpdf},
		{"normalcdf", linear_normalcdf},
		{"normalqf", linear_normalqf},
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
