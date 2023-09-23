/*
 * Lua Linear
 *
 * Copyright (C) 2017-2023 Andre Naef
 */


#include "linear.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <lauxlib.h>
#include <cblas.h>
#include <lapacke.h>


typedef void (*ll_elementary_function)(const int size, double alpha, double *x, const int incx);
typedef double (*ll_unary_function)(const int size, const double *x, const int incx,
		const int ddof);
typedef void (*ll_binary_function)(const int size, const double alpha, double *x,
		const int incx, const double beta, double *y, const int incy);


static inline CBLAS_ORDER ll_checkorder(lua_State *L, int index);
static inline CBLAS_TRANSPOSE ll_checktranspose(lua_State *L, int index);
static inline char ll_lapacktranspose(CBLAS_TRANSPOSE transpose);
static int argerror(lua_State *L, int index);
static inline int ll_rawgeti(lua_State *L, int index, int n);
#if LUA_VERSION_NUM < 502
#define lua_rawlen  lua_objlen
void *luaL_testudata(lua_State *L, int index, const char *name);
#endif

static void ll_push_vector(lua_State *L, size_t length, size_t inc, struct ll_data *data, double *values);
static int ll_vector_len(lua_State *L);
static int ll_vector_index(lua_State *L);
static int ll_vector_newindex(lua_State *L);
#if LUA_VERSION_NUM < 504
static int ll_vector_next(lua_State *L);
static int ll_vector_ipairs(lua_State *L);
#endif
static int ll_vector_tostring(lua_State *L);
static int ll_vector_gc(lua_State *L);

static void ll_push_matrix(lua_State *L, size_t rows, size_t cols, size_t ld, CBLAS_ORDER order,
		struct ll_data *data, double *values);
static int ll_matrix_len(lua_State *L);
static int ll_matrix_index(lua_State *L);
#if LUA_VERSION_NUM < 504
static int ll_matrix_next(lua_State *L);
static int ll_matrix_ipairs(lua_State *L);
#endif
static int ll_matrix_tostring(lua_State *L);
static int ll_matrix_gc(lua_State *L);

static int ll_vector(lua_State *L);
static int ll_matrix(lua_State *L);
static int ll_totable(lua_State *L);
static int ll_tolinear(lua_State *L);
static int ll_type(lua_State *L);
static int ll_size(lua_State *L);
static int ll_tvector(lua_State *L);
static int ll_sub(lua_State *L);
static int ll_unwind(lua_State *L);
static int ll_reshape(lua_State *L);
#if LUA_VERSION_NUM < 502
static int ll_ipairs(lua_State *L);
#endif

static int ll_elementary(lua_State *L, ll_elementary_function f, int hasalpha);
static void ll_inc_handler(const int size, double alpha, double *x, const int incx);
static int ll_inc(lua_State *L);
static int ll_scal(lua_State *L);
static void ll_pow_handler(const int size, double alpha, double *x, const int incx);
static int ll_pow(lua_State *L);
static void ll_exp_handler(const int size, double alpha, double *x, const int incx);
static int ll_exp(lua_State *L);
static void ll_log_handler(const int size, double alpha, double *x, const int incx);
static int ll_log(lua_State *L);
static void ll_sgn_handler(const int size, double alpha, double *x, const int incx);
static int ll_sgn(lua_State *L);
static void ll_abs_handler(const int size, double alpha, double *x, const int incx);
static int ll_abs(lua_State *L);
static void ll_logistic_handler(const int size, double alpha, double *x, const int incx);
static int ll_logistic(lua_State *L);
static void ll_tanh_handler(const int size, double alpha, double *x, const int incx);
static int ll_tanh(lua_State *L);
static void ll_apply_handler(const int size, double alpha, double *x, const int incx);
static int ll_apply(lua_State *L);
static void ll_set_handler(const int size, double alpha, double *x, const int incx);
static int ll_set(lua_State *L);
static void ll_uniform_handler(const int size, double alpha, double *x, const int incx);
static int ll_uniform(lua_State *L);
static void ll_normal_handler(const int size, double alpha, double *x, const int incx);
static int ll_normal(lua_State *L);

static int ll_unary(lua_State *L, ll_unary_function f, int hasddof);
static double ll_sum_handler(int size, const double *values, const int inc, const int ddof);
static int ll_sum(lua_State *L);
static double ll_mean_handler(int size, const double *values, const int inc, const int ddof);
static int ll_mean(lua_State *L);
static double ll_var_handler(int size, const double *values, const int inc, const int ddof);
static int ll_var(lua_State *L);
static double ll_std_handler(int size, const double *values, const int inc, const int ddof);
static int ll_std(lua_State *L);
static double ll_nrm2_handler(int size, const double *values, const int inc, const int ddof);
static int ll_nrm2(lua_State *L);
static double ll_asum_handler(int size, const double *values, const int inc, const int ddof);
static int ll_asum(lua_State *L);

static int ll_binary(lua_State *L, ll_binary_function s, int hasalpha, int hasbeta);
static void ll_axpy_handler(const int size, const double alpha, double *x, int incx, const double beta,
		double *y, int incy);
static int ll_axpy(lua_State *L);
static void ll_axpby_handler(const int size, const double alpha, double *x, int incx, const double beta,
		double *y, int incy);
static int ll_axpby(lua_State *L);
static void ll_mul_handler(const int size, const double alpha, double *x, int incx, const double beta,
		 double *y, int incy);
static int ll_mul(lua_State *L);
static void ll_swap_handler(const int size, const double alpha, double *x, int incx, const double beta,
		double *y, int incy);
static int ll_swap(lua_State *L);
static void ll_copy_handler(const int size, const double alpha, double *x, int incx, const double beta,
		double *y, int incy);
static int ll_copy(lua_State *L);

static int ll_dot(lua_State *L);
static int ll_ger(lua_State *L);
static int ll_gemv(lua_State *L);
static int ll_gemm(lua_State *L);
static int ll_gesv(lua_State *L);
static int ll_gels(lua_State *L);
static int ll_inv(lua_State *L);
static int ll_det(lua_State *L);
static int ll_cov(lua_State *L);
static int ll_corr(lua_State *L);


static const char *const LL_ORDERS[] = {"row", "col", NULL};
static const char *const LL_TRANSPOSES[] = {"notrans", "trans", NULL};
static __thread lua_State  *LL_TL;


/*
 * arguments
 */

static inline CBLAS_ORDER ll_checkorder (lua_State *L, int index) {
	return luaL_checkoption(L, index, "row", LL_ORDERS) == 0 ? CblasRowMajor : CblasColMajor;
}

static inline CBLAS_TRANSPOSE ll_checktranspose (lua_State *L, int index) {
	return luaL_checkoption(L, index, "notrans", LL_TRANSPOSES) == 0 ? CblasNoTrans
			: CblasTrans;
}

static inline char ll_lapacktranspose (CBLAS_TRANSPOSE transpose) {
	return transpose == CblasNoTrans ? 'N' : 'T';
}

static int argerror (lua_State *L, int index) {
	return luaL_argerror(L, index, lua_pushfstring(L, "vector, or matrix expected, got %s",
			luaL_typename(L, index)));
}

static inline int ll_rawgeti (lua_State *L, int index, int n) {
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

struct ll_vector *ll_create_vector (lua_State *L, size_t length) {
	struct ll_vector  *vector;

	assert(length >= 1 && length <= INT_MAX);
	vector = lua_newuserdata(L, sizeof(struct ll_vector));
	vector->length = length;
	vector->inc = 1;
	vector->data = NULL;
	luaL_getmetatable(L, LUALINEAR_VECTOR);
	lua_setmetatable(L, -2);
	vector->data = calloc(1, sizeof(struct ll_data) + length * sizeof(double));
	if (vector->data == NULL) {
		luaL_error(L, "cannot allocate data");
	}
	vector->data->refs = 1;
	vector->values = (double *)((char *)vector->data + sizeof(struct ll_data));
	return vector;
}

static void ll_push_vector (lua_State *L, size_t length, size_t inc, struct ll_data *data,
		double *values) {
	struct ll_vector  *vector;

	assert(length >= 1 && length <= INT_MAX);
	vector = lua_newuserdata(L, sizeof(struct ll_vector));
	vector->length = length;
	vector->inc = inc;
	vector->data = NULL;
	luaL_getmetatable(L, LUALINEAR_VECTOR);
	lua_setmetatable(L, -2);
	vector->data = data;
	data->refs++;
	vector->values = values;
}

static int ll_vector_len (lua_State *L) {
	struct ll_vector  *x;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR);
	lua_pushinteger(L, x->length);
	return 1;
}

static int ll_vector_index (lua_State *L) {
	size_t             index;
	struct ll_vector  *x;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR);
	index = luaL_checkinteger(L, 2);
	if (index >= 1 && index <= x->length) {
		lua_pushnumber(L, x->values[(index - 1) * x->inc]);
	} else {
		lua_pushnil(L);
	}
	return 1;
}

static int ll_vector_newindex (lua_State *L) {
	size_t             index;
	double             value;
	struct ll_vector  *x;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR);
	index = luaL_checkinteger(L, 2);
	luaL_argcheck(L, index >= 1 && index <= x->length, 2, "bad index");
	value = luaL_checknumber(L, 3);
	x->values[(index - 1) * x->inc] = value;
	return 0;
}

#if LUA_VERSION_NUM < 504
static int ll_vector_next (lua_State *L) {
	size_t             index;
	struct ll_vector  *x;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR);
	index = luaL_checkinteger(L, 2);
	if (index < x->length) {
		lua_pushinteger(L, index + 1);
		lua_pushnumber(L, x->values[index * x->inc]);
		return 2;
	}
	lua_pushnil(L);
	return 1;
}

static int ll_vector_ipairs (lua_State *L) {
	luaL_checkudata(L, 1, LUALINEAR_VECTOR);
	lua_pushcfunction(L, ll_vector_next);
	lua_pushvalue(L, 1);
	lua_pushinteger(L, 0);
	return 3;
}
#endif

static int ll_vector_tostring (lua_State *L) {
	struct ll_vector  *x;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR);
	lua_pushfstring(L, LUALINEAR_VECTOR ": %p", x);
	return 1;
}

static int ll_vector_gc (lua_State *L) {
	struct ll_vector  *x;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR);
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

struct ll_matrix *ll_create_matrix (lua_State *L, size_t rows, size_t cols, CBLAS_ORDER order) {
	struct ll_matrix  *matrix;

	assert(rows >= 1 && rows <= INT_MAX && cols >= 1 && cols <= INT_MAX);
	matrix = lua_newuserdata(L, sizeof(struct ll_matrix));
	matrix->rows = rows;
	matrix->cols = cols;
	matrix->ld = order == CblasRowMajor ? cols : rows;
	matrix->order = order;
	matrix->data = NULL;
	luaL_getmetatable(L, LUALINEAR_MATRIX);
	lua_setmetatable(L, -2);
	matrix->data = calloc(1, sizeof(struct ll_data) + rows * cols * sizeof(double));
	if (matrix->data == NULL) {
		luaL_error(L, "cannot allocate data");
	}
	matrix->data->refs = 1;
	matrix->values = (double *)((char *)matrix->data + sizeof(struct ll_data));
	return matrix;
}

static void ll_push_matrix (lua_State *L, size_t rows, size_t cols, size_t ld, CBLAS_ORDER order,
		struct ll_data *data, double *values) {
	struct ll_matrix  *matrix;

	assert(rows >= 1 && rows <= INT_MAX && cols >= 1 && cols <= INT_MAX);
	matrix = (struct ll_matrix *)lua_newuserdata(L, sizeof(struct ll_matrix));
	matrix->rows = rows;
	matrix->cols = cols;
	matrix->ld = ld;
	matrix->order = order;
	matrix->data = NULL;
	luaL_getmetatable(L, LUALINEAR_MATRIX);
	lua_setmetatable(L, -2);
	matrix->data = data;
	data->refs++;
	matrix->values = values;
}

static int ll_matrix_len (lua_State *L) {
	struct ll_matrix  *X;

	X = luaL_checkudata(L, 1, LUALINEAR_MATRIX);
	if (X->order == CblasRowMajor) {
		lua_pushinteger(L, X->rows);
	} else {
		lua_pushinteger(L, X->cols);
	}
	return 1;
}

static int ll_matrix_index (lua_State *L) {
	size_t             index;
	struct ll_matrix  *X;

	X = luaL_checkudata(L, 1, LUALINEAR_MATRIX);
	index = luaL_checkinteger(L, 2);
	if (X->order == CblasRowMajor) {
		if (index >= 1 && index <= X->rows) {
			ll_push_vector(L, X->cols, 1, X->data, &X->values[(index - 1) * X->ld]);
		} else {
			lua_pushnil(L);
		}
	} else {
		if (index >= 1 && index <= X->cols) {
			ll_push_vector(L, X->rows, 1, X->data, &X->values[(index - 1) * X->ld]);
		} else {
			lua_pushnil(L);
		}
	}
	return 1;
}

#if LUA_VERSION_NUM < 504
static int ll_matrix_next (lua_State *L) {
	size_t             index, majorsize, minorsize;
	struct ll_matrix  *X;

	X = luaL_checkudata(L, 1, LUALINEAR_MATRIX);
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
		ll_push_vector(L, minorsize, 1, X->data, &X->values[index * X->ld]);
		return 2;
	}
	lua_pushnil(L);
	return 1;
}

static int ll_matrix_ipairs (lua_State *L) {
	luaL_checkudata(L, 1, LUALINEAR_MATRIX);
	lua_pushcfunction(L, ll_matrix_next);
	lua_pushvalue(L, 1);
	lua_pushinteger(L, 0);
	return 3;
}
#endif

static int ll_matrix_tostring (lua_State *L) {
	struct ll_matrix  *X;

	X = luaL_checkudata(L, 1, LUALINEAR_MATRIX);
	lua_pushfstring(L, LUALINEAR_MATRIX ": %p", X);
	return 1;
}

static int ll_matrix_gc (lua_State *L) {
	struct ll_matrix  *X;

	X = luaL_checkudata(L, 1, LUALINEAR_MATRIX);
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

static int ll_vector (lua_State *L) {
	size_t  size;

	size = luaL_checkinteger(L, 1);
	luaL_argcheck(L, size >= 1 && size <= INT_MAX, 1, "bad dimension");
	ll_create_vector(L, size);
	return 1;
}

static int ll_matrix (lua_State *L) {
	size_t       rows, cols;
	CBLAS_ORDER  order;

	rows = luaL_checkinteger(L, 1);
	luaL_argcheck(L, rows >= 1 && rows <= INT_MAX, 1, "bad dimension");
	cols = luaL_checkinteger(L, 2);
	luaL_argcheck(L, cols >= 1 && cols <= INT_MAX, 2, "bad dimension");
	order = ll_checkorder(L, 3);
	ll_create_matrix(L, rows, cols, order);
	return 1;
}

static int ll_totable (lua_State *L) {
	size_t             i, j;
	const double      *value;
	struct ll_vector  *x;
	struct ll_matrix  *X;

	x = luaL_testudata(L, 1, LUALINEAR_VECTOR);
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
	X = luaL_testudata(L, 1, LUALINEAR_MATRIX);
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
	return argerror(L, 1);
}

static int ll_tolinear (lua_State *L) {
	double            *value;
	size_t             size, rows, cols, major, minor, i, j;
	CBLAS_ORDER        order;
	struct ll_vector  *x;
	struct ll_matrix  *X;

	luaL_checktype(L, 1, LUA_TTABLE);
	switch (ll_rawgeti(L, 1, 1)) {
	case LUA_TNUMBER:
		size = lua_rawlen(L, 1);
		if (size < 1 || size > INT_MAX) {
			return luaL_error(L, "bad size");
		}
		x = ll_create_vector(L, size);
		value = x->values;
		for (i = 0; i < size; i++) {
			if (ll_rawgeti(L, 1, i + 1) != LUA_TNUMBER) {
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
		order = ll_checkorder(L, 2);
		if (order == CblasRowMajor) {
			rows = major;
			cols = minor;
		} else {
			rows = minor;
			cols = major;
		}
		X = ll_create_matrix(L, rows, cols, order);
		for (i = 0; i < major; i++) {
			value = &X->values[i * X->ld];
			if (ll_rawgeti(L, 1, i + 1) != LUA_TTABLE
					 || lua_rawlen(L, -1) != minor) {
				return luaL_error(L, "bad value at index %d", i + 1);
			}
			for (j = 0; j < minor; j++) {
				if (ll_rawgeti(L, -1, j + 1) != LUA_TNUMBER) {
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

static int ll_type (lua_State *L) {
	if (luaL_testudata(L, 1, LUALINEAR_VECTOR) != NULL) {
		lua_pushliteral(L, "vector");
		return 1;
	}
	if (luaL_testudata(L, 1, LUALINEAR_MATRIX) != NULL) {
		lua_pushliteral(L, "matrix");
		return 1;
	}
	lua_pushnil(L);
	return 1;
}

static int ll_size (lua_State *L) {
	struct ll_vector  *x;
	struct ll_matrix  *X;

	x = luaL_testudata(L, 1, LUALINEAR_VECTOR);
	if (x != NULL) {
		lua_pushinteger(L, x->length);
		return 1;
	}
	X = luaL_testudata(L, 1, LUALINEAR_MATRIX);
	if (X != NULL) {
		lua_pushinteger(L, X->rows);
		lua_pushinteger(L, X->cols);
		lua_pushstring(L, LL_ORDERS[X->order == CblasRowMajor ? 0 : 1]);
		return 3;
	}
	return argerror(L, 1);
}

static int ll_tvector (lua_State *L) {
	size_t             index, length;
	struct ll_matrix  *X;

	X = luaL_checkudata(L, 1, LUALINEAR_MATRIX);
	index = luaL_checkinteger(L, 2);
	if (X->order == CblasRowMajor) {
		luaL_argcheck(L, index >= 1 && index <= X->cols, 2, "bad index");
		length = X->rows;
	} else {
		luaL_argcheck(L, index >= 1 && index <= X->rows, 2, "bad index");
		length = X->cols;
	}
	ll_push_vector(L, length, X->ld, X->data, &X->values[index - 1]);
	return 1;
}

static int ll_sub (lua_State *L) {
	struct ll_vector  *x;
	struct ll_matrix  *X;

	x = luaL_testudata(L, 1, LUALINEAR_VECTOR);
	if (x != NULL) {
		size_t  start, end;

		start = luaL_optinteger(L, 2, 1);
		luaL_argcheck(L, start >= 1 && start <= x->length, 2, "bad index");
		end = luaL_optinteger(L, 3, x->length);
		luaL_argcheck(L, end >= start && end <= x->length, 3, "bad index");
		ll_push_vector(L, end - start + 1, x->inc, x->data,
				&x->values[(start - 1) * x->inc]);
		return 1;
	}
	X = luaL_testudata(L, 1, LUALINEAR_MATRIX);
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
			ll_push_matrix(L, rowend - rowstart + 1, colend - colstart + 1, X->ld,
					X->order, X->data, &X->values[(rowstart - 1) * X->ld
					+ (colstart - 1)]);
		} else {
			ll_push_matrix(L, rowend - rowstart + 1, colend - colstart + 1, X->ld,
					X->order, X->data, &X->values[(colstart - 1) * X->ld
					+ (rowstart - 1)]);
		}
		return 1;
	}
	return argerror(L, 1);
}

static int ll_unwind (lua_State *L) {
	int                index;
	double            *s, *d, *last;
	size_t             i, j;
	struct ll_vector  *x;
	struct ll_matrix  *X;

	if (lua_gettop(L) == 0) {
		return luaL_error(L, "wrong number of arguments");
	}
	x = luaL_checkudata(L, lua_gettop(L), LUALINEAR_VECTOR);
	d = x->values;
	last = d + x->length * x->inc;
	index = 1;
	while (d < last) {
		X = luaL_checkudata(L, index, LUALINEAR_MATRIX);
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

static int ll_reshape (lua_State *L) {
	int                index;
	double            *d, *s, *last;
	size_t             i, j;
	struct ll_vector  *x;
	struct ll_matrix  *X;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR);
	s = x->values;
	last = x->values + x->length * x->inc;
	index = 2;
	while (s < last) {
		X = luaL_checkudata(L, index, LUALINEAR_MATRIX);
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
static int ll_ipairs (lua_State *L) {
	if (luaL_testudata(L, 1, LUALINEAR_VECTOR)) {
		return ll_vector_ipairs(L);
	}
	if (luaL_testudata(L, 1, LUALINEAR_MATRIX)) {
		return ll_matrix_ipairs(L);
	}
	return argerror(L, 1);
}
#endif


/*
 * elementary functions
 */

static int ll_elementary (lua_State *L, ll_elementary_function f, int hasalpha) {
	int                isnum;
	size_t             i;
	double             alpha, n;
	struct ll_vector  *x;
	struct ll_matrix  *X;

	alpha = hasalpha ? luaL_optnumber(L, 2, 1.0) : 0.0;
#if LUA_VERSION_NUM >= 502
	n = lua_tonumberx(L, 1, &isnum);
#else
	isnum = lua_isnumber(L, 1);
	n = lua_tonumber(L, 1);
#endif
	if (isnum) {
		f(1, alpha, &n, 1);
		lua_pushnumber(L, n);
		return 1;
	}
	x = luaL_testudata(L, 1, LUALINEAR_VECTOR);
	if (x != NULL) {
		f(x->length, alpha, x->values, x->inc);
		return 0;
	}
	X = luaL_testudata(L, 1, LUALINEAR_MATRIX);
	if (X != NULL) {
		if (X->order == CblasRowMajor) {
			if (X->cols == X->ld && X->rows * X->cols <= INT_MAX) {
				f(X->rows * X->cols, alpha, X->values, 1);
			} else {
				for (i = 0; i < X->rows; i++) {
					f(X->cols, alpha, &X->values[i * X->ld], 1);
				}
			}
		} else {
			if (X->rows == X->ld && X->cols * X->rows <= INT_MAX) {
				f(X->cols * X->rows, alpha, X->values, 1);
			} else {
				for (i = 0; i < X->cols; i++) {
					f(X->rows, alpha, &X->values[i * X->ld], 1);
				}
			}
		}
		return 0;
	}
	return argerror(L, 1);
}

static void ll_inc_handler (const int size, double alpha, double *x, const int incx) {
	int  i;

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

static int ll_inc (lua_State *L) {
	return ll_elementary(L, ll_inc_handler, 1);
}

static int ll_scal (lua_State *L) {
	return ll_elementary(L, cblas_dscal, 1);
}

static void ll_pow_handler (const int size, double alpha, double *x, const int incx) {
	int  i;

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

static int ll_pow (lua_State *L) {
	return ll_elementary(L, ll_pow_handler, 1);
}

static void ll_exp_handler (const int size, double alpha, double *x, const int incx) {
	int  i;

	(void)alpha;
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

static int ll_exp (lua_State *L) {
	return ll_elementary(L, ll_exp_handler, 0);
}

static void ll_log_handler (const int size, double alpha, double *x, const int incx) {
	int  i;

	(void)alpha;
	if (incx == 2) {
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

static int ll_log (lua_State *L) {
	return ll_elementary(L, ll_log_handler, 0);
}

static void ll_sgn_handler (const int size, double alpha, double *x, const int incx) {
	int  i;

	(void)alpha;
	for (i = 0; i < size; i++) {
		if (*x > 0) {
			*x = 1;
		} else if (*x < 0) {
			*x = -1;
		}
		x += incx;
	}
}

static int ll_sgn (lua_State *L) {
	return ll_elementary(L, ll_sgn_handler, 0);
}

static void ll_abs_handler (const int size, double alpha, double *x, const int incx) {
	int  i;

	(void)alpha;
	for (i = 0; i < size; i++) {
		*x = fabs(*x);
		x += incx;
	}
}

static int ll_abs (lua_State *L) {
	return ll_elementary(L, ll_abs_handler, 0);
}

static void ll_logistic_handler (const int size, double alpha, double *x, const int incx) {
	int  i;

	(void)alpha;
	for (i = 0; i < size; i++) {
		*x = 1.0 / (1.0 + exp(-*x));
		x += incx;
	}
}

static int ll_logistic (lua_State *L) {
	return ll_elementary(L, ll_logistic_handler, 0);
}

static void ll_tanh_handler (const int size, double alpha, double *x, const int incx) {
	int  i;

	(void)alpha;
	for (i = 0; i < size; i++) {
		*x = tanh(*x);
		x += incx;
	}
}

static int ll_tanh (lua_State *L) {
	return ll_elementary(L, ll_tanh_handler, 0);
}

static void ll_apply_handler (const int size, double alpha, double *x, const int incx) {
	int  i;

	(void)alpha;
	for (i = 0; i < size; i++) {
		lua_pushvalue(LL_TL, -1);
		lua_pushnumber(LL_TL, *x);
		lua_call(LL_TL, 1, 1);
		*x = lua_tonumber(LL_TL, -1);
		x += incx;
		lua_pop(LL_TL, 1);
	}
}

static int ll_apply (lua_State *L) {
	luaL_checktype(L, 2, LUA_TFUNCTION);
	lua_settop(L, 2);
	LL_TL = L;
	return ll_elementary(L, ll_apply_handler, 0);
}

static void ll_set_handler (const int size, double alpha, double *x, const int incx) {
	int  i;

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

static int ll_set (lua_State *L) {
	return ll_elementary(L, ll_set_handler, 1);
}

static void ll_uniform_handler (const int size, double alpha, double *x, const int incx) {
	int  i;

	(void)alpha;
	for (i = 0; i < size; i++) {
		*x = random() / (RAND_MAX + 1.0);
		x += incx;
	}
}

static int ll_uniform (lua_State *L) {
	return ll_elementary(L, ll_uniform_handler, 0);
}

static void ll_normal_handler (const int size, double alpha, double *x, const int incx) {
	int     i;
	double  u1, u2, r, s, c;

	(void)alpha;

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

static int ll_normal (lua_State *L) {
	return ll_elementary(L, ll_normal_handler, 0);
}


/*
 * unary vector functions
 */

static int ll_unary (lua_State *L, ll_unary_function f, int hasddof) {
	size_t             i, ddof;
	struct ll_vector  *x, *y;
	struct ll_matrix  *X;

	ddof = 0;
	x = luaL_testudata(L, 1, LUALINEAR_VECTOR);
	if (x != NULL) {
		/* vector */
		if (hasddof) {
			ddof = luaL_optinteger(L, 2, 0);
			luaL_argcheck(L, ddof < x->length, 2, "bad ddof");
		}
		lua_pushnumber(L, f(x->length, x->values, x->inc, ddof));
		return 1;
	}
	X = luaL_testudata(L, 1, LUALINEAR_MATRIX);
	if (X != NULL) {
		/* matrix-vector */
		y = luaL_checkudata(L, 2, LUALINEAR_VECTOR);
		if (ll_checkorder(L, 3) == CblasRowMajor) {
			luaL_argcheck(L, y->length == X->rows, 2, "dimension mismatch");
			if (hasddof) {
				ddof = luaL_optinteger(L, 4, 0);
				luaL_argcheck(L, ddof < X->rows, 4, "bad ddof");
			}
			if (X->order == CblasRowMajor) {
				for (i = 0; i < X->rows; i++) {
					y->values[i * y->inc] = f(X->cols, &X->values[i * X->ld],
							1, ddof);
				}
			} else {
				for (i = 0; i < X->rows; i++) {
					y->values[i * y->inc] = f(X->cols, &X->values[i], X->ld,
							ddof);
				}
			}
		} else {
			luaL_argcheck(L, y->length == X->cols, 2, "dimension mismatch");
			if (hasddof) {
				ddof = luaL_optinteger(L, 4, 0);
				luaL_argcheck(L, ddof < X->cols, 4, "bad ddof");
			}
			if (X->order == CblasColMajor) {
				for (i = 0; i < X->cols; i++) {
					y->values[i * y->inc] = f(X->rows, &X->values[i * X->ld],
							1, ddof);
				}
			} else {
				for (i = 0; i < X->cols; i++) {
					y->values[i * y->inc] = f(X->rows, &X->values[i], X->ld,
							ddof);
				}
			}
		}
		return 0;
	}
	return argerror(L, 1);
}

static double ll_sum_handler (int size, const double *x, const int incx, const int ddof) {
	int     i;
	double  sum;

	(void)ddof;
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

static int ll_sum (lua_State *L) {
	return ll_unary(L, ll_sum_handler, 0);
}

static double ll_mean_handler (int size, const double *x, const int incx, const int ddof) {
	int     i;
	double  sum;

	(void)ddof;
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

static int ll_mean (lua_State *L) {
	return ll_unary(L, ll_mean_handler, 0);
}

static double ll_var_handler (int size, const double *x, const int incx, const int ddof) {
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
	return sum / (size - ddof);
}

static int ll_var (lua_State *L) {
	return ll_unary(L, ll_var_handler, 1);
}

static double ll_std_handler (int size, const double *x, const int incx, const int ddof) {
	return sqrt(ll_var_handler(size, x, incx, ddof));
}

static int ll_std (lua_State *L) {
	return ll_unary(L, ll_std_handler, 1);
}

static double ll_nrm2_handler (int size, const double *x, const int incx, const int ddof) {
	(void)ddof;
	return cblas_dnrm2(size, x, incx);
}

static int ll_nrm2 (lua_State *L) {
	return ll_unary(L, ll_nrm2_handler, 0);
}

static double ll_asum_handler (int size, const double *x, const int incx, const int ddof) {
	(void)ddof;
	return cblas_dasum(size, x, incx);
}

static int ll_asum (lua_State *L) {
	return ll_unary(L, ll_asum_handler, 0);
}


/*
 * binary vector functions
 */

static int ll_binary (lua_State *L, ll_binary_function f, int hasalpha, int hasbeta) {
	size_t             i;
	double             alpha, beta;
	struct ll_vector  *x, *y;
	struct ll_matrix  *X, *Y;

	x = luaL_testudata(L, 1, LUALINEAR_VECTOR);
	if (x != NULL) {
		y = luaL_testudata(L, 2, LUALINEAR_VECTOR);
		if (y != NULL) {
			/* vector-vector */
			luaL_argcheck(L, y->length == x->length, 2, "dimension mismatch");
			alpha = hasalpha ? luaL_optnumber(L, 3, 1.0) : 0.0;
			beta = hasbeta ? luaL_optnumber(L, 4, 0.0) : 0.0;
			f(x->length, alpha, x->values, x->inc, beta, y->values, y->inc);
			return 0;
		}
		Y = luaL_testudata(L, 2, LUALINEAR_MATRIX);
		if (Y != NULL) {
			/* vector-matrix */
			alpha = hasalpha ? luaL_optnumber(L, 4, 1.0) : 0.0;
			beta = hasbeta ? luaL_optnumber(L, 5, 0.0) : 0.0;
			if (ll_checkorder(L, 3) == CblasRowMajor) {
				luaL_argcheck(L, x->length == Y->cols, 1, "dimension mismatch");
				if (Y->order == CblasRowMajor) {
					for (i = 0; i < Y->rows; i++) {
						f(x->length, alpha, x->values, x->inc, beta,
								&Y->values[i * Y->ld], 1);
					}
				} else {
					for (i = 0; i < Y->rows; i++) {
						f(x->length, alpha, x->values, x->inc, beta,
								&Y->values[i], Y->ld);
					}
				}
			} else {
				luaL_argcheck(L, x->length == Y->rows, 1, "dimension mismatch");
				if (Y->order == CblasColMajor) {
					for (i = 0; i < Y->cols; i++) {
						f(x->length, alpha, x->values, x->inc, beta,
								&Y->values[i * Y->ld], 1);
					}
				} else {
					for (i = 0; i < Y->cols; i++) {
						f(x->length, alpha, x->values, x->inc, beta,
								&Y->values[i], Y->ld);
					}
				}
			}
			return 0;
		}
		return argerror(L, 2);
	}
	X = luaL_testudata(L, 1, LUALINEAR_MATRIX);
	if (X != NULL) {
		/* matrix-matrix */
		Y = luaL_checkudata(L, 2, LUALINEAR_MATRIX);
		luaL_argcheck(L, X->order == Y->order, 2, "order mismatch");
		luaL_argcheck(L, X->rows == Y->rows && X->cols == Y->cols, 2, "dimension mismatch");
		alpha = hasalpha ? luaL_optnumber(L, 3, 1.0) : 0.0;
		beta = hasbeta ? luaL_optnumber(L, 4, 0.0) : 0.0;
		if (X->order == CblasRowMajor) {
			if (X->ld == X->cols && Y->ld == Y->cols && X->rows * X->cols <= INT_MAX) {
				f(X->rows * X->cols, alpha, X->values, 1, beta, Y->values, 1);
			} else {
				for (i = 0; i < X->rows; i++) {
					f(X->cols, alpha, &X->values[i * X->ld], 1, beta,
							&Y->values[i * Y->ld], 1);
				}
			}
		} else {
			if (X->ld == X->rows && Y->ld == Y->rows && X->cols * X->rows <= INT_MAX) {
				f(X->cols * X->rows, alpha, X->values, 1, beta, Y->values, 1);
			} else {
				for (i = 0; i < X->cols; i++) {
					f(X->rows, alpha, &X->values[i * X->ld], 1, beta,
							&Y->values[i * Y->ld], 1);
				}
			}
		}
		return 0;
	}
	return argerror(L, 1);
}

static void ll_axpy_handler (const int size, const double alpha, double *x, int incx,
		const double beta, double *y, int incy) {
	(void)beta;
	cblas_daxpy(size, alpha, x, incx, y, incy);
}

/* deprecated */
static int ll_axpy (lua_State *L) {
	return ll_binary(L, ll_axpy_handler, 1, 0);
}

static void ll_axpby_handler (const int size, const double alpha, double *x, int incx,
		const double beta, double *y, int incy) {
#if LUALINEAR_USE_AXPBY
	cblas_daxpby(size, alpha, x, incx, beta, y, incy);
#else
	if (beta != 1.0) {
		cblas_dscal(size, beta, y, incy);
	}
	cblas_daxpy(size, alpha, x, incx, y, incy);
#endif
}

static int ll_axpby (lua_State *L) {
	return ll_binary(L, ll_axpby_handler, 1, 1);
}

static void ll_mul_handler (const int size, const double alpha, double *x, int incx,
		const double beta, double *y, int incy) {
	int  i;

	(void)beta;
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

static int ll_mul (lua_State *L) {
	return ll_binary(L, ll_mul_handler, 1, 0);
}

static void ll_swap_handler (const int size, const double alpha, double *x, int incx,
		const double beta, double *y, int incy) {
	(void)alpha;
	(void)beta;
	cblas_dswap(size, x, incx, y, incy);
}

static int ll_swap (lua_State *L) {
	return ll_binary(L, ll_swap_handler, 0, 0);
}

static void ll_copy_handler (const int size, const double alpha, double *x, int incx,
		const double beta, double *y, int incy) {
	(void)alpha;
	(void)beta;
	cblas_dcopy(size, x, incx, y, incy);
}

static int ll_copy (lua_State *L) {
	return ll_binary(L, ll_copy_handler, 0, 0);
}


/*
 * program functions
 */

static int ll_dot (lua_State *L) {
	struct ll_vector  *x, *y;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR);
	y = luaL_checkudata(L, 2, LUALINEAR_VECTOR);
	luaL_argcheck(L, y->length == x->length, 2, "dimension mismatch");
	lua_pushnumber(L, cblas_ddot(x->length, x->values, x->inc, y->values, y->inc));
	return 1;
}

static int ll_ger (lua_State *L) {
	double             alpha;
	struct ll_vector  *x, *y;
	struct ll_matrix  *A;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR);
	y = luaL_checkudata(L, 2, LUALINEAR_VECTOR);
	A = luaL_checkudata(L, 3, LUALINEAR_MATRIX);
	luaL_argcheck(L, x->length == A->rows, 1, "dimension mismatch");
	luaL_argcheck(L, y->length == A->cols, 2, "dimension mismatch");
	alpha = luaL_optnumber(L, 4, 1.0);
	cblas_dger(A->order, A->rows, A->cols, alpha, x->values, x->inc, y->values, y->inc,
			A->values, A->ld);
	return 0;
}

static int ll_gemv (lua_State *L) {
	size_t              m, n;
	double              alpha, beta;
	CBLAS_TRANSPOSE     ta;
	struct ll_matrix   *A;
	struct ll_vector   *x, *y;

	A = luaL_checkudata(L, 1, LUALINEAR_MATRIX);
	x = luaL_checkudata(L, 2, LUALINEAR_VECTOR);
	y = luaL_checkudata(L, 3, LUALINEAR_VECTOR);
	ta = ll_checktranspose(L, 4);
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

static int ll_gemm (lua_State *L) {
	size_t               m, n, ka, kb;
	double               alpha, beta;
	CBLAS_TRANSPOSE      ta, tb;
	struct ll_matrix    *A, *B, *C;

	A = luaL_checkudata(L, 1, LUALINEAR_MATRIX);
	B = luaL_checkudata(L, 2, LUALINEAR_MATRIX);
	luaL_argcheck(L, B->order == A->order, 2, "order mismatch");
	C = luaL_checkudata(L, 3, LUALINEAR_MATRIX);
	luaL_argcheck(L, C->order == A->order, 3, "order mismatch");
	ta = ll_checktranspose(L, 4);
	tb = ll_checktranspose(L, 5);
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

static int ll_gesv (lua_State *L) {
	int               *ipiv, result;
	struct ll_matrix  *A, *B;

	A = luaL_checkudata(L, 1, LUALINEAR_MATRIX);
	luaL_argcheck(L, A->rows == A->cols, 1, "not square");
	B = luaL_checkudata(L, 2, LUALINEAR_MATRIX);
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

static int ll_gels (lua_State *L) {
	int                result;
	char               ta;
	struct ll_matrix  *A, *B;

	A = luaL_checkudata(L, 1, LUALINEAR_MATRIX);
	B = luaL_checkudata(L, 2, LUALINEAR_MATRIX);
	luaL_argcheck(L, B->order == A->order, 2, "order mismatch");
	ta = ll_lapacktranspose(ll_checktranspose(L, 3));
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

static int ll_inv (lua_State *L) {
	int               *ipiv, result;
	struct ll_matrix  *A;

	A = luaL_checkudata(L, 1, LUALINEAR_MATRIX);
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

static int ll_det (lua_State *L) {
	int               *ipiv, result, neg;
	size_t             n, i;
	double            *copy, *d, *s, det;
	struct ll_matrix  *A;

	/* check and process arguments */
	A = luaL_checkudata(L, 1, LUALINEAR_MATRIX);
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

static int ll_cov (lua_State *L) {
	size_t             i, j, k, ddof;
	double            *means, *v, *vi, *vj, sum;
	struct ll_matrix  *A, *B;

	/* check and process arguments */
	A = luaL_checkudata(L, 1, LUALINEAR_MATRIX);
	B = luaL_checkudata(L, 2, LUALINEAR_MATRIX);
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

static int ll_corr (lua_State *L) {
	size_t             i, j, k;
	double            *means, *stds, *v, *vi, *vj, sum;
	struct ll_matrix  *A, *B;

	/* check and process arguments */
	A = luaL_checkudata(L, 1, LUALINEAR_MATRIX);
	B = luaL_checkudata(L, 2, LUALINEAR_MATRIX);
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
		{ "vector", ll_vector },
		{ "matrix", ll_matrix },
		{ "totable", ll_totable },
		{ "tolinear", ll_tolinear },
		{ "type", ll_type },
		{ "size", ll_size },
		{ "tvector", ll_tvector },
		{ "sub", ll_sub },
		{ "unwind", ll_unwind },
		{ "reshape", ll_reshape },
#if LUA_VERSION_NUM < 502
		{ "ipairs", ipairs },
#endif

		/* elementary functions */
		{ "inc", ll_inc },
		{ "scal", ll_scal },
		{ "pow", ll_pow },
		{ "exp", ll_exp },
		{ "log", ll_log },
		{ "sgn", ll_sgn },
		{ "abs", ll_abs },
		{ "logistic", ll_logistic },
		{ "tanh", ll_tanh },
		{ "apply", ll_apply },
		{ "set", ll_set },
		{ "uniform", ll_uniform },
		{ "normal", ll_normal },

		/* unary vector functions */
		{ "sum", ll_sum },
		{ "mean", ll_mean },
		{ "var", ll_var },
		{ "std", ll_std },
		{ "nrm2", ll_nrm2 },
		{ "asum", ll_asum },

		/* binary vector functions */
		{ "axpy", ll_axpy },    /* deprecated */
		{ "axpby", ll_axpby },
		{ "mul", ll_mul },
		{ "swap", ll_swap },
		{ "copy", ll_copy },

		/* program functions */
		{ "dot", ll_dot },
		{ "ger", ll_ger },
		{ "gemv", ll_gemv },
		{ "gemm", ll_gemm },
		{ "gesv", ll_gesv },
		{ "gels", ll_gels },
		{ "inv", ll_inv },
		{ "det", ll_det },
		{ "cov", ll_cov },
		{ "corr", ll_corr },

		{ NULL, NULL }
	};

	/* register functions */
#if LUA_VERSION_NUM >= 502
	luaL_newlib(L, FUNCTIONS);
#else
	luaL_register(L, luaL_checkstring(L, 1), FUNCTIONS);
#endif

	/* vector metatable */
	luaL_newmetatable(L, LUALINEAR_VECTOR);
	lua_pushcfunction(L, ll_vector_len);
	lua_setfield(L, -2, "__len");
	lua_pushcfunction(L, ll_vector_index);
	lua_setfield(L, -2, "__index");
	lua_pushcfunction(L, ll_vector_newindex);
	lua_setfield(L, -2, "__newindex");
#if LUA_VERSION_NUM >= 502 && LUA_VERSION_NUM < 504
	lua_pushcfunction(L, ll_vector_ipairs);
	lua_setfield(L, -2, "__ipairs");
#endif
	lua_pushcfunction(L, ll_vector_tostring);
	lua_setfield(L, -2, "__tostring");
	lua_pushcfunction(L, ll_vector_gc);
	lua_setfield(L, -2, "__gc");
	lua_pop(L, 1);

	/* matrix metatable */
	luaL_newmetatable(L, LUALINEAR_MATRIX);
	lua_pushcfunction(L, ll_matrix_len);
	lua_setfield(L, -2, "__len");
	lua_pushcfunction(L, ll_matrix_index);
	lua_setfield(L, -2, "__index");
#if LUA_VERSION_NUM >= 502 && LUA_VERSION_NUM < 504
	lua_pushcfunction(L, ll_matrix_ipairs);
	lua_setfield(L, -2, "__ipairs");
#endif
	lua_pushcfunction(L, ll_matrix_tostring);
	lua_setfield(L, -2, "__tostring");
	lua_pushcfunction(L, ll_matrix_gc);
	lua_setfield(L, -2, "__gc");
	lua_pop(L, 1);

	return 1;
}
