/*
 * Lua Linear
 *
 * Copyright (C) 2017-2023 Andre Naef
 */


#include "linear_core.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <lauxlib.h>
#include <cblas.h>
#include <lapacke.h>


/* arguments */
static inline CBLAS_ORDER linear_checkorder(lua_State *L, int index);
static inline CBLAS_TRANSPOSE linear_checktranspose(lua_State *L, int index);
static inline char linear_lapacktranspose(CBLAS_TRANSPOSE transpose);
static inline int linear_checkargs(lua_State *L, struct linear_param *params, size_t size,
		int index, union linear_arg *args);
static int linear_argerror(lua_State *L, int index, int numok);
static inline int linear_rawgeti(lua_State *L, int index, int n);
#if LUA_VERSION_NUM < 502
#define lua_rawlen  lua_objlen
void *luaL_testudata(lua_State *L, int index, const char *name);
#endif

/* vector */
static void linear_push_vector(lua_State *L, size_t length, size_t inc, struct linear_data *data,
		double *values);
static int linear_vector_len(lua_State *L);
static int linear_vector_index(lua_State *L);
static int linear_vector_newindex(lua_State *L);
#if LUA_VERSION_NUM < 504
static int linear_vector_next(lua_State *L);
static int linear_vector_ipairs(lua_State *L);
#endif
static int linear_vector_tostring(lua_State *L);
static int linear_vector_gc(lua_State *L);

/* matrix */
static void linear_push_matrix(lua_State *L, size_t rows, size_t cols, size_t ld, CBLAS_ORDER order,
		struct linear_data *data, double *values);
static int linear_matrix_len(lua_State *L);
static int linear_matrix_index(lua_State *L);
#if LUA_VERSION_NUM < 504
static int linear_matrix_next(lua_State *L);
static int linear_matrix_ipairs(lua_State *L);
#endif
static int linear_matrix_tostring(lua_State *L);
static int linear_matrix_gc(lua_State *L);

/* structural functions */
static int linear_vector(lua_State *L);
static int linear_matrix(lua_State *L);
static int linear_totable(lua_State *L);
static int linear_tolinear(lua_State *L);
static int linear_type(lua_State *L);
static int linear_size(lua_State *L);
static int linear_tvector(lua_State *L);
static int linear_sub(lua_State *L);
static int linear_unwind(lua_State *L);
static int linear_reshape(lua_State *L);
#if LUA_VERSION_NUM < 502
static int linear_ipairs(lua_State *L);
#endif

/* elementary functions */
static void linear_inc_handler(int size, double *x, int incx, union linear_arg *args);
static int linear_inc(lua_State *L);
static void linear_scal_handler(int size, double *x, int incx, union linear_arg *args);
static int linear_scal(lua_State *L);
static void linear_pow_handler(int size, double *x, int incx, union linear_arg *args);
static int linear_pow(lua_State *L);
static void linear_exp_handler(int size, double *x, int incx, union linear_arg *args);
static int linear_exp(lua_State *L);
static void linear_log_handler(int size, double *x, int incx, union linear_arg *args);
static int linear_log(lua_State *L);
static void linear_sgn_handler(int size, double *x, int incx, union linear_arg *args);
static int linear_sgn(lua_State *L);
static void linear_abs_handler(int size, double *x, int incx, union linear_arg *args);
static int linear_abs(lua_State *L);
static void linear_logistic_handler(int size, double *x, int incx, union linear_arg *args);
static int linear_logistic(lua_State *L);
static void linear_tanh_handler(int size, double *x, int incx, union linear_arg *args);
static int linear_tanh(lua_State *L);
static void linear_apply_handler(int size, double *x, int incx, union linear_arg *args);
static int linear_apply(lua_State *L);
static void linear_set_handler(int size, double *x, int incx, union linear_arg *args);
static int linear_set(lua_State *L);
static void linear_uniform_handler(int size, double *x, int incx, union linear_arg *args);
static int linear_uniform(lua_State *L);
static void linear_normal_handler(int size, double *x, int incx, union linear_arg *args);
static int linear_normal(lua_State *L);

/* unary vector functions */
static double linear_sum_handler(int size, double *values, int inc, union linear_arg *args);
static int linear_sum(lua_State *L);
static double linear_mean_handler(int size, double *values, int inc, union linear_arg *args);
static int linear_mean(lua_State *L);
static double linear_var_handler(int size, double *values, int inc, union linear_arg *args);
static int linear_var(lua_State *L);
static double linear_std_handler(int size, double *values, int inc, union linear_arg *args);
static int linear_std(lua_State *L);
static double linear_nrm2_handler(int size, double *values, int inc, union linear_arg *args);
static int linear_nrm2(lua_State *L);
static double linear_asum_handler(int size, double *values, int inc, union linear_arg *args);
static int linear_asum(lua_State *L);

/* binary vector functions */
static void linear_axpy_handler(int size, double *x, int incx, double *y, int incy,
		union linear_arg *args);
static int linear_axpy(lua_State *L);
static void linear_axpby_handler(int size, double *x, int incx, double *y, int incy,
		union linear_arg *args);
static int linear_axpby(lua_State *L);
static void linear_mul_handler(int size, double *x, int incx, double *y, int incy,
		union linear_arg *args);
static int linear_mul(lua_State *L);
static void linear_swap_handler(int size, double *x, int incx, double *y, int incy,
		union linear_arg *args);
static int linear_swap(lua_State *L);
static void linear_copy_handler(int size, double *x, int incx, double *y, int incy,
		union linear_arg *args);
static int linear_copy(lua_State *L);

/* program functions */
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


static const char *const LINEAR_ORDERS[] = {"row", "col", NULL};
static const char *const LINEAR_TRANSPOSES[] = {"notrans", "trans", NULL};
static struct linear_param LINEAR_PARAMS_NONE[] = {
	{NULL, '\0', {0.0}}
};
static struct linear_param LINEAR_PARAMS_ALPHA[] = {
	{"alpha", 'n', {1.0}},
	{NULL, '\0', {0.0}}
};
static struct linear_param LINEAR_PARAMS_ALPHA_BETA[] = {
	{"alpha", 'n', {1.0}},
	{"beta", 'n', {0.0}},
	{NULL, '\0', {0.0}}
};
static struct linear_param LINEAR_PARAMS_DDOF[] = {
	{"ddof", 'd', {.defd = 0}},
	{NULL, '\0', {0.0}}
};

static __thread lua_State  *LINEAR_TL;


/*
 * arguments
 */

static inline CBLAS_ORDER linear_checkorder (lua_State *L, int index) {
	return luaL_checkoption(L, index, "row", LINEAR_ORDERS) == 0 ? CblasRowMajor
			: CblasColMajor;
}

static inline CBLAS_TRANSPOSE linear_checktranspose (lua_State *L, int index) {
	return luaL_checkoption(L, index, "notrans", LINEAR_TRANSPOSES) == 0 ? CblasNoTrans
			: CblasTrans;
}

static inline char linear_lapacktranspose (CBLAS_TRANSPOSE transpose) {
	return transpose == CblasNoTrans ? 'N' : 'T';
}

static inline int linear_checkargs (lua_State *L, struct linear_param *params, size_t size,
		int index, union linear_arg *args) {
	while (params->name) {
		switch (params->type) {
		case 'n':
			args->n = luaL_optnumber(L, index, params->defn);
			break;

		case 'd':
			args->d = luaL_optinteger(L, index, params->defd);
			luaL_argcheck(L, args->d < size, index, "bad ddof");
			break;

		default:
			return luaL_error(L, "bad param type");
		}
		params++;
		index++;
		args++;
	}
	return 0;
}

static int linear_argerror (lua_State *L, int index, int numok) {
	const char  *fmt;

	fmt = !numok ? "vector, or matrix expected, got %s"
			: "number, vector, or matrix expected, got %s";
	return luaL_argerror(L, index, lua_pushfstring(L, fmt, luaL_typename(L, index)));
}

static inline int linear_rawgeti (lua_State *L, int index, int n) {
#if LUA_VERSION_NUM >= 503
	return lua_rawgeti(L, index, n);
#else
	lua_rawgeti(L, index, n);
	return lua_type(L, -1);
#endif
}

#if LUA_VERSION_NUM < 502
void *luaL_testudata (lua_State *L, int index, const char *name) {
	void  *userdata;

	userdata = lua_touserdata(L, index);
	if (!userdata || !lua_getmetatable(L, index)) {
		return NULL;
	}
	luaL_getmetatable(L, name);
	if (!lua_rawequal(L, -1, -2)) {
		userdata = NULL;
	}
	lua_pop(L, 2);
	return userdata;
}
#endif


/*
 * vector
 */

struct linear_vector *linear_create_vector (lua_State *L, size_t length) {
	struct linear_vector  *vector;

	assert(length >= 1 && length <= INT_MAX);
	vector = lua_newuserdata(L, sizeof(struct linear_vector));
	vector->length = length;
	vector->inc = 1;
	vector->data = NULL;
	luaL_getmetatable(L, LINEAR_VECTOR);
	lua_setmetatable(L, -2);
	vector->data = calloc(1, sizeof(struct linear_data) + length * sizeof(double));
	if (vector->data == NULL) {
		luaL_error(L, "cannot allocate data");
	}
	vector->data->refs = 1;
	vector->values = (double *)((char *)vector->data + sizeof(struct linear_data));
	return vector;
}

static void linear_push_vector (lua_State *L, size_t length, size_t inc, struct linear_data *data,
		double *values) {
	struct linear_vector  *vector;

	assert(length >= 1 && length <= INT_MAX);
	vector = lua_newuserdata(L, sizeof(struct linear_vector));
	vector->length = length;
	vector->inc = inc;
	vector->data = NULL;
	luaL_getmetatable(L, LINEAR_VECTOR);
	lua_setmetatable(L, -2);
	vector->data = data;
	data->refs++;
	vector->values = values;
}

static int linear_vector_len (lua_State *L) {
	struct linear_vector  *x;

	x = luaL_checkudata(L, 1, LINEAR_VECTOR);
	lua_pushinteger(L, x->length);
	return 1;
}

static int linear_vector_index (lua_State *L) {
	size_t             index;
	struct linear_vector  *x;

	x = luaL_checkudata(L, 1, LINEAR_VECTOR);
	index = luaL_checkinteger(L, 2);
	if (index >= 1 && index <= x->length) {
		lua_pushnumber(L, x->values[(index - 1) * x->inc]);
	} else {
		lua_pushnil(L);
	}
	return 1;
}

static int linear_vector_newindex (lua_State *L) {
	size_t             index;
	double             value;
	struct linear_vector  *x;

	x = luaL_checkudata(L, 1, LINEAR_VECTOR);
	index = luaL_checkinteger(L, 2);
	luaL_argcheck(L, index >= 1 && index <= x->length, 2, "bad index");
	value = luaL_checknumber(L, 3);
	x->values[(index - 1) * x->inc] = value;
	return 0;
}

#if LUA_VERSION_NUM < 504
static int linear_vector_next (lua_State *L) {
	size_t             index;
	struct linear_vector  *x;

	x = luaL_checkudata(L, 1, LINEAR_VECTOR);
	index = luaL_checkinteger(L, 2);
	if (index < x->length) {
		lua_pushinteger(L, index + 1);
		lua_pushnumber(L, x->values[index * x->inc]);
		return 2;
	}
	lua_pushnil(L);
	return 1;
}

static int linear_vector_ipairs (lua_State *L) {
	luaL_checkudata(L, 1, LINEAR_VECTOR);
	lua_pushcfunction(L, linear_vector_next);
	lua_pushvalue(L, 1);
	lua_pushinteger(L, 0);
	return 3;
}
#endif

static int linear_vector_tostring (lua_State *L) {
	struct linear_vector  *x;

	x = luaL_checkudata(L, 1, LINEAR_VECTOR);
	lua_pushfstring(L, LINEAR_VECTOR ": %p", x);
	return 1;
}

static int linear_vector_gc (lua_State *L) {
	struct linear_vector  *x;

	x = luaL_checkudata(L, 1, LINEAR_VECTOR);
	if (x->data) {
		x->data->refs--;
		if (x->data->refs == 0) {
			free(x->data);
		}
	}
	return 0;
}


/*
 * matrix
 */

struct linear_matrix *linear_create_matrix (lua_State *L, size_t rows, size_t cols,
		CBLAS_ORDER order) {
	struct linear_matrix  *matrix;

	assert(rows >= 1 && rows <= INT_MAX && cols >= 1 && cols <= INT_MAX);
	matrix = lua_newuserdata(L, sizeof(struct linear_matrix));
	matrix->rows = rows;
	matrix->cols = cols;
	matrix->ld = order == CblasRowMajor ? cols : rows;
	matrix->order = order;
	matrix->data = NULL;
	luaL_getmetatable(L, LINEAR_MATRIX);
	lua_setmetatable(L, -2);
	matrix->data = calloc(1, sizeof(struct linear_data) + rows * cols * sizeof(double));
	if (matrix->data == NULL) {
		luaL_error(L, "cannot allocate data");
	}
	matrix->data->refs = 1;
	matrix->values = (double *)((char *)matrix->data + sizeof(struct linear_data));
	return matrix;
}

static void linear_push_matrix (lua_State *L, size_t rows, size_t cols, size_t ld,
		CBLAS_ORDER order, struct linear_data *data, double *values) {
	struct linear_matrix  *matrix;

	assert(rows >= 1 && rows <= INT_MAX && cols >= 1 && cols <= INT_MAX);
	matrix = (struct linear_matrix *)lua_newuserdata(L, sizeof(struct linear_matrix));
	matrix->rows = rows;
	matrix->cols = cols;
	matrix->ld = ld;
	matrix->order = order;
	matrix->data = NULL;
	luaL_getmetatable(L, LINEAR_MATRIX);
	lua_setmetatable(L, -2);
	matrix->data = data;
	data->refs++;
	matrix->values = values;
}

static int linear_matrix_len (lua_State *L) {
	struct linear_matrix  *X;

	X = luaL_checkudata(L, 1, LINEAR_MATRIX);
	if (X->order == CblasRowMajor) {
		lua_pushinteger(L, X->rows);
	} else {
		lua_pushinteger(L, X->cols);
	}
	return 1;
}

static int linear_matrix_index (lua_State *L) {
	size_t                 index;
	struct linear_matrix  *X;

	X = luaL_checkudata(L, 1, LINEAR_MATRIX);
	index = luaL_checkinteger(L, 2);
	if (X->order == CblasRowMajor) {
		if (index >= 1 && index <= X->rows) {
			linear_push_vector(L, X->cols, 1, X->data, &X->values[(index - 1) * X->ld]);
		} else {
			lua_pushnil(L);
		}
	} else {
		if (index >= 1 && index <= X->cols) {
			linear_push_vector(L, X->rows, 1, X->data, &X->values[(index - 1) * X->ld]);
		} else {
			lua_pushnil(L);
		}
	}
	return 1;
}

#if LUA_VERSION_NUM < 504
static int linear_matrix_next (lua_State *L) {
	size_t                 index, majorsize, minorsize;
	struct linear_matrix  *X;

	X = luaL_checkudata(L, 1, LINEAR_MATRIX);
	index = luaL_checkinteger(L, 2);
	if (X->order == CblasRowMajor) {
		majorsize = X->rows;
		minorsize = X->cols;
	} else {
		majorsize = X->cols;
		minorsize = X->rows;
	}
	if (index < majorsize) {
		lua_pushinteger(L, index + 1);
		linear_push_vector(L, minorsize, 1, X->data, &X->values[index * X->ld]);
		return 2;
	}
	lua_pushnil(L);
	return 1;
}

static int linear_matrix_ipairs (lua_State *L) {
	luaL_checkudata(L, 1, LINEAR_MATRIX);
	lua_pushcfunction(L, linear_matrix_next);
	lua_pushvalue(L, 1);
	lua_pushinteger(L, 0);
	return 3;
}
#endif

static int linear_matrix_tostring (lua_State *L) {
	struct linear_matrix  *X;

	X = luaL_checkudata(L, 1, LINEAR_MATRIX);
	lua_pushfstring(L, LINEAR_MATRIX ": %p", X);
	return 1;
}

static int linear_matrix_gc (lua_State *L) {
	struct linear_matrix  *X;

	X = luaL_checkudata(L, 1, LINEAR_MATRIX);
	if (X->data) {
		X->data->refs--;
		if (X->data->refs == 0) {
			free(X->data);
		}
	}
	return 0;
}


/*
 * structural functions
 */

static int linear_vector (lua_State *L) {
	size_t  size;

	size = luaL_checkinteger(L, 1);
	luaL_argcheck(L, size >= 1 && size <= INT_MAX, 1, "bad dimension");
	linear_create_vector(L, size);
	return 1;
}

static int linear_matrix (lua_State *L) {
	size_t       rows, cols;
	CBLAS_ORDER  order;

	rows = luaL_checkinteger(L, 1);
	luaL_argcheck(L, rows >= 1 && rows <= INT_MAX, 1, "bad dimension");
	cols = luaL_checkinteger(L, 2);
	luaL_argcheck(L, cols >= 1 && cols <= INT_MAX, 2, "bad dimension");
	order = linear_checkorder(L, 3);
	linear_create_matrix(L, rows, cols, order);
	return 1;
}

static int linear_totable (lua_State *L) {
	size_t                 i, j;
	const double          *value;
	struct linear_vector  *x;
	struct linear_matrix  *X;

	x = luaL_testudata(L, 1, LINEAR_VECTOR);
	if (x != NULL) {
		lua_createtable(L, x->length, 0);
		value = x->values;
		for (i = 0; i < x->length; i++) {
			lua_pushnumber(L, *value);
			lua_rawseti(L, -2, i + 1);
			value += x->inc;
		}
		return 1;
	}
	X = luaL_testudata(L, 1, LINEAR_MATRIX);
	if (X != NULL) {
		if (X->order == CblasRowMajor) {
			lua_createtable(L, X->rows, 0);
			for (i = 0; i < X->rows; i++) {
				lua_createtable(L, X->cols, 0);
				value = &X->values[i * X->ld];
				for (j = 0; j < X->cols; j++) {
					lua_pushnumber(L, *value++);
					lua_rawseti(L, -2, j + 1);
				}
				lua_rawseti(L, -2, i + 1);
			}
		} else {
			lua_createtable(L, X->cols, 0);
			for (i = 0; i < X->cols; i++) {
				lua_createtable(L, X->rows, 0);
				value = &X->values[i * X->ld];
				for (j = 0; j < X->rows; j++) {
					lua_pushnumber(L, *value++);
					lua_rawseti(L, -2, j + 1);
				}
				lua_rawseti(L, -2, i + 1);
			}
		}
		return 1;
	}
	return linear_argerror(L, 1, 0);
}

static int linear_tolinear (lua_State *L) {
	double                *value;
	size_t                 size, rows, cols, major, minor, i, j;
	CBLAS_ORDER            order;
	struct linear_vector  *x;
	struct linear_matrix  *X;

	luaL_checktype(L, 1, LUA_TTABLE);
	switch (linear_rawgeti(L, 1, 1)) {
	case LUA_TNUMBER:
		size = lua_rawlen(L, 1);
		if (size < 1 || size > INT_MAX) {
			return luaL_error(L, "bad size");
		}
		x = linear_create_vector(L, size);
		value = x->values;
		for (i = 0; i < size; i++) {
			if (linear_rawgeti(L, 1, i + 1) != LUA_TNUMBER) {
				return luaL_error(L, "bad value at index %d", i + 1);
			}
			*value++ = lua_tonumber(L, -1);
			lua_pop(L, 1);
		}
		return 1;

	case LUA_TTABLE:
		major = lua_rawlen(L, 1);
		if (major < 1 || major > INT_MAX) {
			return luaL_error(L, "bad rows");
		}
		minor = lua_rawlen(L, -1);
		if (minor < 1 || minor > INT_MAX) {
			return luaL_error(L, "bad columns");
		}
		lua_pop(L, 1);
		order = linear_checkorder(L, 2);
		if (order == CblasRowMajor) {
			rows = major;
			cols = minor;
		} else {
			rows = minor;
			cols = major;
		}
		X = linear_create_matrix(L, rows, cols, order);
		for (i = 0; i < major; i++) {
			value = &X->values[i * X->ld];
			if (linear_rawgeti(L, 1, i + 1) != LUA_TTABLE
					 || lua_rawlen(L, -1) != minor) {
				return luaL_error(L, "bad value at index %d", i + 1);
			}
			for (j = 0; j < minor; j++) {
				if (linear_rawgeti(L, -1, j + 1) != LUA_TNUMBER) {
					return luaL_error(L, "bad value at index (%d,%d)", i + 1,
							j + 1);
				}
				*value++ = lua_tonumber(L, -1);
				lua_pop(L, 1);
			}
			lua_pop(L, 1);
		}
		return 1;

	default:
		return luaL_argerror(L, 1, "bad table");
	}
}

static int linear_type (lua_State *L) {
	if (luaL_testudata(L, 1, LINEAR_VECTOR) != NULL) {
		lua_pushliteral(L, "vector");
		return 1;
	}
	if (luaL_testudata(L, 1, LINEAR_MATRIX) != NULL) {
		lua_pushliteral(L, "matrix");
		return 1;
	}
	lua_pushnil(L);
	return 1;
}

static int linear_size (lua_State *L) {
	struct linear_vector  *x;
	struct linear_matrix  *X;

	x = luaL_testudata(L, 1, LINEAR_VECTOR);
	if (x != NULL) {
		lua_pushinteger(L, x->length);
		return 1;
	}
	X = luaL_testudata(L, 1, LINEAR_MATRIX);
	if (X != NULL) {
		lua_pushinteger(L, X->rows);
		lua_pushinteger(L, X->cols);
		lua_pushstring(L, LINEAR_ORDERS[X->order == CblasRowMajor ? 0 : 1]);
		return 3;
	}
	return linear_argerror(L, 1, 0);
}

static int linear_tvector (lua_State *L) {
	size_t                 index, length;
	struct linear_matrix  *X;

	X = luaL_checkudata(L, 1, LINEAR_MATRIX);
	index = luaL_checkinteger(L, 2);
	if (X->order == CblasRowMajor) {
		luaL_argcheck(L, index >= 1 && index <= X->cols, 2, "bad index");
		length = X->rows;
	} else {
		luaL_argcheck(L, index >= 1 && index <= X->rows, 2, "bad index");
		length = X->cols;
	}
	linear_push_vector(L, length, X->ld, X->data, &X->values[index - 1]);
	return 1;
}

static int linear_sub (lua_State *L) {
	struct linear_vector  *x;
	struct linear_matrix  *X;

	x = luaL_testudata(L, 1, LINEAR_VECTOR);
	if (x != NULL) {
		size_t  start, end;

		start = luaL_optinteger(L, 2, 1);
		luaL_argcheck(L, start >= 1 && start <= x->length, 2, "bad index");
		end = luaL_optinteger(L, 3, x->length);
		luaL_argcheck(L, end >= start && end <= x->length, 3, "bad index");
		linear_push_vector(L, end - start + 1, x->inc, x->data,
				&x->values[(start - 1) * x->inc]);
		return 1;
	}
	X = luaL_testudata(L, 1, LINEAR_MATRIX);
	if (X != NULL) {
		size_t  rowstart, rowend, colstart, colend;

		rowstart = luaL_optinteger(L, 2, 1);
		luaL_argcheck(L, rowstart >= 1 && rowstart <= X->rows, 2, "bad index");
		colstart = luaL_optinteger(L, 3, 1);
		luaL_argcheck(L, colstart >= 1 && colstart <= X->cols, 3, "bad index");
		rowend = luaL_optinteger(L, 4, X->rows);
		luaL_argcheck(L, rowend >= rowstart && rowend <= X->rows, 4, "bad index");
		colend = luaL_optinteger(L, 5, X->cols);
		luaL_argcheck(L, colend >= colstart && colend <= X->cols, 5, "bad index");
		if (X->order == CblasRowMajor) {
			linear_push_matrix(L, rowend - rowstart + 1, colend - colstart + 1, X->ld,
					X->order, X->data, &X->values[(rowstart - 1) * X->ld
					+ (colstart - 1)]);
		} else {
			linear_push_matrix(L, rowend - rowstart + 1, colend - colstart + 1, X->ld,
					X->order, X->data, &X->values[(colstart - 1) * X->ld
					+ (rowstart - 1)]);
		}
		return 1;
	}
	return linear_argerror(L, 1, 0);
}

static int linear_unwind (lua_State *L) {
	int                    index;
	double                *s, *d, *last;
	size_t                 i, j;
	struct linear_vector  *x;
	struct linear_matrix  *X;

	if (lua_gettop(L) == 0) {
		return luaL_error(L, "wrong number of arguments");
	}
	x = luaL_checkudata(L, lua_gettop(L), LINEAR_VECTOR);
	d = x->values;
	last = d + x->length * x->inc;
	index = 1;
	while (d < last) {
		X = luaL_checkudata(L, index, LINEAR_MATRIX);
		luaL_argcheck(L, d + X->rows * X->cols * x->inc <= last, index, "matrix too large");
		if (X->order == CblasRowMajor) {
			for (i = 0; i < X->rows; i++) {
				s = &X->values[i * X->ld];
				for (j = 0; j < X->cols; j++) {
					*d = *s++;
					d += x->inc;
				}
			}
		} else {
			for (i = 0; i < X->cols; i++) {
				s = &X->values[i * X->ld];
				for (j = 0; j < X->rows; j++) {
					*d = *s++;
					d += x->inc;
				}
			}
		}
		index++;
	}
	return 0;
}

static int linear_reshape (lua_State *L) {
	int                    index;
	double                *d, *s, *last;
	size_t                 i, j;
	struct linear_vector  *x;
	struct linear_matrix  *X;

	x = luaL_checkudata(L, 1, LINEAR_VECTOR);
	s = x->values;
	last = x->values + x->length * x->inc;
	index = 2;
	while (s < last) {
		X = luaL_checkudata(L, index, LINEAR_MATRIX);
		luaL_argcheck(L, s + X->rows * X->cols * x->inc <= last, index,	"matrix too large");
		if (X->order == CblasRowMajor) {
			for (i = 0; i < X->rows; i++) {
				d = &X->values[i * X->ld];
				for (j = 0; j < X->cols; j++) {
					*d++ = *s;
					s += x->inc;
				}
			}
		} else {
			for (i = 0; i < X->cols; i++) {
				d = &X->values[i * X->ld];
				for (j = 0; j < X->rows; j++) {
					*d++ = *s;
					s += x->inc;
				}
			}
		}
		index++;
	}
	return 0;
}

#if LUA_VERSION_NUM < 502
static int linear_ipairs (lua_State *L) {
	if (luaL_testudata(L, 1, LINEAR_VECTOR)) {
		return linear_vector_ipairs(L);
	}
	if (luaL_testudata(L, 1, LINEAR_MATRIX)) {
		return linear_matrix_ipairs(L);
	}
	return argerror(L, 1);
}
#endif


/*
 * elementary functions
 */

int linear_elementary (lua_State *L, linear_elementary_function f, struct linear_param *params) {
	int                    isnum;
	size_t                 i;
	double                 n;
	union linear_arg       args[LINEAR_PARAM_MAX];
	struct linear_vector  *x;
	struct linear_matrix  *X;

	linear_checkargs(L, params, 0, 2, args);
#if LUA_VERSION_NUM >= 502
	n = lua_tonumberx(L, 1, &isnum);
#else
	isnum = lua_isnumber(L, 1);
	n = lua_tonumber(L, 1);
#endif
	if (isnum) {
		f(1, &n, 1, args);
		lua_pushnumber(L, n);
		return 1;
	}
	x = luaL_testudata(L, 1, LINEAR_VECTOR);
	if (x != NULL) {
		f(x->length, x->values, x->inc, args);
		return 0;
	}
	X = luaL_testudata(L, 1, LINEAR_MATRIX);
	if (X != NULL) {
		if (X->order == CblasRowMajor) {
			if (X->cols == X->ld && X->rows * X->cols <= INT_MAX) {
				f(X->rows * X->cols, X->values, 1, args);
			} else {
				for (i = 0; i < X->rows; i++) {
					f(X->cols, &X->values[i * X->ld], 1, args);
				}
			}
		} else {
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

static void linear_inc_handler (int size, double *x, int incx, union linear_arg *args) {
	int     i;
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
	return linear_elementary(L, linear_inc_handler, LINEAR_PARAMS_ALPHA);
}

static void linear_scal_handler (int size, double *x, int incx, union linear_arg *args) {
	cblas_dscal(size, args[0].n, x, incx);
}

static int linear_scal (lua_State *L) {
	return linear_elementary(L, linear_scal_handler, LINEAR_PARAMS_ALPHA);
}

static void linear_pow_handler (int size, double *x, int incx, union linear_arg *args) {
	int     i;
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
	return linear_elementary(L, linear_pow_handler, LINEAR_PARAMS_ALPHA);
}

static void linear_exp_handler (int size, double *x, int incx, union linear_arg *args) {
	int  i;

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
	return linear_elementary(L, linear_exp_handler, LINEAR_PARAMS_NONE);
}

static void linear_log_handler (int size, double *x, int incx, union linear_arg *args) {
	int  i;

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
	return linear_elementary(L, linear_log_handler, LINEAR_PARAMS_NONE);
}

static void linear_sgn_handler (int size, double *x, int incx, union linear_arg *args) {
	int  i;

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
	return linear_elementary(L, linear_sgn_handler, LINEAR_PARAMS_NONE);
}

static void linear_abs_handler (int size, double *x, int incx, union linear_arg *args) {
	int  i;

	(void)args;
	for (i = 0; i < size; i++) {
		*x = fabs(*x);
		x += incx;
	}
}

static int linear_abs (lua_State *L) {
	return linear_elementary(L, linear_abs_handler, LINEAR_PARAMS_NONE);
}

static void linear_logistic_handler (int size, double *x, int incx, union linear_arg *args) {
	int  i;

	(void)args;
	for (i = 0; i < size; i++) {
		*x = 1.0 / (1.0 + exp(-*x));
		x += incx;
	}
}

static int linear_logistic (lua_State *L) {
	return linear_elementary(L, linear_logistic_handler, LINEAR_PARAMS_NONE);
}

static void linear_tanh_handler (int size, double *x, int incx, union linear_arg *args) {
	int  i;

	(void)args;
	for (i = 0; i < size; i++) {
		*x = tanh(*x);
		x += incx;
	}
}

static int linear_tanh (lua_State *L) {
	return linear_elementary(L, linear_tanh_handler, LINEAR_PARAMS_NONE);
}

static void linear_apply_handler (int size, double *x, int incx, union linear_arg *args) {
	int  i;

	(void)args;
	for (i = 0; i < size; i++) {
		lua_pushvalue(LINEAR_TL, -1);
		lua_pushnumber(LINEAR_TL, *x);
		lua_call(LINEAR_TL, 1, 1);
		*x = lua_tonumber(LINEAR_TL, -1);
		x += incx;
		lua_pop(LINEAR_TL, 1);
	}
}

static int linear_apply (lua_State *L) {
	luaL_checktype(L, 2, LUA_TFUNCTION);
	lua_settop(L, 2);
	LINEAR_TL = L;
	return linear_elementary(L, linear_apply_handler, LINEAR_PARAMS_NONE);
}

static void linear_set_handler (int size, double *x, int incx, union linear_arg *args) {
	int     i;
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
	return linear_elementary(L, linear_set_handler, LINEAR_PARAMS_ALPHA);
}

static void linear_uniform_handler (int size, double *x, int incx, union linear_arg *args) {
	int  i;

	(void)args;
	for (i = 0; i < size; i++) {
		*x = random() / (RAND_MAX + 1.0);
		x += incx;
	}
}

static int linear_uniform (lua_State *L) {
	return linear_elementary(L, linear_uniform_handler, LINEAR_PARAMS_NONE);
}

static void linear_normal_handler (int size, double *x, int incx, union linear_arg *args) {
	int     i;
	double  u1, u2, r, s, c;

	(void)args;

	/* Box-Muller transform */
	for (i = 0; i < size - 1; i += 2) {
		u1 = (random() + 1.0) / (RAND_MAX + 1.0);
		u2 = (random() + 1.0) / (RAND_MAX + 1.0);
		r = sqrt(-2.0 * log(u1));
		sincos(2 * M_PI * u2, &s, &c);
		*x = r * c;
		x += incx;
		*x = r * s;
		x += incx;
	}
	if (i < size) {
		u1 = (random() + 1.0) / (RAND_MAX + 1.0);
		u2 = (random() + 1.0) / (RAND_MAX + 1.0);
		*x = sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);
	}
}

static int linear_normal (lua_State *L) {
	return linear_elementary(L, linear_normal_handler, LINEAR_PARAMS_NONE);
}


/*
 * unary vector functions
 */

int linear_unary (lua_State *L, linear_unary_function f, struct linear_param *params) {
	size_t                 i;
	union linear_arg       args[LINEAR_PARAM_MAX];
	struct linear_vector  *x, *y;
	struct linear_matrix  *X;

	x = luaL_testudata(L, 1, LINEAR_VECTOR);
	if (x != NULL) {
		/* vector */
		linear_checkargs(L, params, x->length, 2, args);
		lua_pushnumber(L, f(x->length, x->values, x->inc, args));
		return 1;
	}
	X = luaL_testudata(L, 1, LINEAR_MATRIX);
	if (X != NULL) {
		/* matrix-vector */
		y = luaL_checkudata(L, 2, LINEAR_VECTOR);
		if (linear_checkorder(L, 3) == CblasRowMajor) {
			luaL_argcheck(L, y->length == X->rows, 2, "dimension mismatch");
			linear_checkargs(L, params, X->cols, 4, args);
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
			linear_checkargs(L, params, X->rows, 4, args);
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

static double linear_sum_handler (int size, double *x, int incx, union linear_arg *args) {
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

static double linear_mean_handler (int size, double *x, int incx, union linear_arg *args) {
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

static double linear_var_handler (int size, double *x, int incx, union linear_arg *args) {
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

static double linear_std_handler (int size, double *x, int incx, union linear_arg *args) {
	return sqrt(linear_var_handler(size, x, incx, args));
}

static int linear_std (lua_State *L) {
	return linear_unary(L, linear_std_handler, LINEAR_PARAMS_DDOF);
}

static double linear_nrm2_handler (int size, double *x, int incx, union linear_arg *args) {
	(void)args;
	return cblas_dnrm2(size, x, incx);
}

static int linear_nrm2 (lua_State *L) {
	return linear_unary(L, linear_nrm2_handler, LINEAR_PARAMS_NONE);
}

static double linear_asum_handler (int size, double *x, int incx, union linear_arg *args) {
	(void)args;
	return cblas_dasum(size, x, incx);
}

static int linear_asum (lua_State *L) {
	return linear_unary(L, linear_asum_handler, LINEAR_PARAMS_NONE);
}


/*
 * binary vector functions
 */

int linear_binary (lua_State *L, linear_binary_function f, struct linear_param *params) {
	size_t                 i;
	union linear_arg       args[LINEAR_PARAM_MAX];
	struct linear_vector  *x, *y;
	struct linear_matrix  *X, *Y;

	x = luaL_testudata(L, 1, LINEAR_VECTOR);
	if (x != NULL) {
		y = luaL_testudata(L, 2, LINEAR_VECTOR);
		if (y != NULL) {
			/* vector-vector */
			luaL_argcheck(L, y->length == x->length, 2, "dimension mismatch");
			linear_checkargs(L, params, 0, 3, args);
			f(x->length, x->values, x->inc, y->values, y->inc, args);
			return 0;
		}
		Y = luaL_testudata(L, 2, LINEAR_MATRIX);
		if (Y != NULL) {
			/* vector-matrix */
			linear_checkargs(L, params, 0, 4, args);
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
		linear_checkargs(L, params, 0, 3, args);
		if (X->order == CblasRowMajor) {
			if (X->ld == X->cols && Y->ld == Y->cols && X->rows * X->cols <= INT_MAX) {
				f(X->rows * X->cols, X->values, 1, Y->values, 1, args);
			} else {
				for (i = 0; i < X->rows; i++) {
					f(X->cols, &X->values[i * X->ld], 1, &Y->values[i * Y->ld],
							1, args);
				}
			}
		} else {
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
		union linear_arg *args) {
	cblas_daxpy(size, args[0].n, x, incx, y, incy);
}

/* deprecated */
static int linear_axpy (lua_State *L) {
	return linear_binary(L, linear_axpy_handler, LINEAR_PARAMS_ALPHA);
}

static void linear_axpby_handler (int size, double *x, int incx, double *y, int incy,
		union linear_arg *args) {
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
		union linear_arg *args) {
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
		union linear_arg *args) {
	(void)args;
	cblas_dswap(size, x, incx, y, incy);
}

static int linear_swap (lua_State *L) {
	return linear_binary(L, linear_swap_handler, LINEAR_PARAMS_NONE);
}

static void linear_copy_handler (int size, double *x, int incx, double *y, int incy,
		union linear_arg *args) {
	(void)args;
	cblas_dcopy(size, x, incx, y, incy);
}

static int linear_copy (lua_State *L) {
	return linear_binary(L, linear_copy_handler, LINEAR_PARAMS_NONE);
}


/*
 * program functions
 */

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


/*
 * library
 */

int luaopen_linear (lua_State *L) {
	static const luaL_Reg FUNCTIONS[] = {
		/* structural functions */
		{ "vector", linear_vector },
		{ "matrix", linear_matrix },
		{ "totable", linear_totable },
		{ "tolinear", linear_tolinear },
		{ "type", linear_type },
		{ "size", linear_size },
		{ "tvector", linear_tvector },
		{ "sub", linear_sub },
		{ "unwind", linear_unwind },
		{ "reshape", linear_reshape },
#if LUA_VERSION_NUM < 502
		{ "ipairs", ipairs },
#endif

		/* elementary functions */
		{ "inc", linear_inc },
		{ "scal", linear_scal },
		{ "pow", linear_pow },
		{ "exp", linear_exp },
		{ "log", linear_log },
		{ "sgn", linear_sgn },
		{ "abs", linear_abs },
		{ "logistic", linear_logistic },
		{ "tanh", linear_tanh },
		{ "apply", linear_apply },
		{ "set", linear_set },
		{ "uniform", linear_uniform },
		{ "normal", linear_normal },

		/* unary vector functions */
		{ "sum", linear_sum },
		{ "mean", linear_mean },
		{ "var", linear_var },
		{ "std", linear_std },
		{ "nrm2", linear_nrm2 },
		{ "asum", linear_asum },

		/* binary vector functions */
		{ "axpy", linear_axpy },    /* deprecated */
		{ "axpby", linear_axpby },
		{ "mul", linear_mul },
		{ "swap", linear_swap },
		{ "copy", linear_copy },

		/* program functions */
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

	/* register functions */
#if LUA_VERSION_NUM >= 502
	luaL_newlib(L, FUNCTIONS);
#else
	luaL_register(L, luaL_checkstring(L, 1), FUNCTIONS);
#endif

	/* vector metatable */
	luaL_newmetatable(L, LINEAR_VECTOR);
	lua_pushcfunction(L, linear_vector_len);
	lua_setfield(L, -2, "__len");
	lua_pushcfunction(L, linear_vector_index);
	lua_setfield(L, -2, "__index");
	lua_pushcfunction(L, linear_vector_newindex);
	lua_setfield(L, -2, "__newindex");
#if LUA_VERSION_NUM >= 502 && LUA_VERSION_NUM < 504
	lua_pushcfunction(L, linear_vector_ipairs);
	lua_setfield(L, -2, "__ipairs");
#endif
	lua_pushcfunction(L, linear_vector_tostring);
	lua_setfield(L, -2, "__tostring");
	lua_pushcfunction(L, linear_vector_gc);
	lua_setfield(L, -2, "__gc");
	lua_pop(L, 1);

	/* matrix metatable */
	luaL_newmetatable(L, LINEAR_MATRIX);
	lua_pushcfunction(L, linear_matrix_len);
	lua_setfield(L, -2, "__len");
	lua_pushcfunction(L, linear_matrix_index);
	lua_setfield(L, -2, "__index");
#if LUA_VERSION_NUM >= 502 && LUA_VERSION_NUM < 504
	lua_pushcfunction(L, linear_matrix_ipairs);
	lua_setfield(L, -2, "__ipairs");
#endif
	lua_pushcfunction(L, linear_matrix_tostring);
	lua_setfield(L, -2, "__tostring");
	lua_pushcfunction(L, linear_matrix_gc);
	lua_setfield(L, -2, "__gc");
	lua_pop(L, 1);

	return 1;
}
