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


typedef void (*elementary_function)(const int size, double alpha, double *x, const int incx);
typedef double (*vector_function)(const int size, const double *x, const int incx, const int ddof);
typedef void (*vector_matrix_function)(const int size, const double alpha, double *x,
		const int incx, const double beta, double *y, const int incy);


static inline CBLAS_ORDER checkorder(lua_State *L, int index);
static inline CBLAS_TRANSPOSE checktranspose(lua_State *L, int index);
static inline char lapacktranspose(CBLAS_TRANSPOSE transpose);
static int argerror(lua_State *L, int index);

struct vector *create_vector(lua_State *L, size_t size);
struct vector *wrap_vector(lua_State *L, size_t size, double *values);
static int vector(lua_State *L);
static int vector_len(lua_State *L);
static int vector_index(lua_State *L);
static int vector_newindex(lua_State *L);
static int vector_next(lua_State *L);
static int vector_ipairs(lua_State *L);
static int vector_tostring(lua_State *L);
static int vector_gc(lua_State *L);

struct matrix *create_matrix(lua_State *L, size_t rows, size_t cols, CBLAS_ORDER order);
struct matrix *wrap_matrix(lua_State *L, size_t rows, size_t cols, CBLAS_ORDER order,
		double *values);
static int matrix(lua_State *L);
static int matrix_len(lua_State *L);
static int matrix_index(lua_State *L);
static int matrix_next(lua_State *L);
static int matrix_ipairs(lua_State *L);
static int matrix_tostring(lua_State *L);
static int matrix_free(lua_State *L);

static int type(lua_State *L);
static int size(lua_State *L);
static int tvector(lua_State *L);
static int sub(lua_State *L);
static int unwind(lua_State *L);
static int reshape(lua_State *L);
static int totable(lua_State *L);
static int tolinear(lua_State *L);

static int elementary(lua_State *L, elementary_function f, int hasalpha);
static void _sgn(const int size, double alpha, double *x, const int incx);
static int sgn(lua_State *L);
static void _abs(const int size, double alpha, double *x, const int incx);
static int absx(lua_State *L);
static void _exp(const int size, double alpha, double *x, const int incx);
static int expx(lua_State *L);
static void _log(const int size, double alpha, double *x, const int incx);
static int logx(lua_State *L);
static void _logistic(const int size, double alpha, double *x, const int incx);
static int logistic(lua_State *L);
static void _tanh(const int size, double alpha, double *x, const int incx);
static int tanhx(lua_State *L);
static void _softplus(const int size, double alpha, double *x, const int incx);
static int softplus(lua_State *L);
static void _rectifier(const int size, double alpha, double *x, const int incx);
static int rectifier(lua_State *L);
static void _set(const int size, double alpha, double *x, const int incx);
static int set(lua_State *L);
static void _uniform(const int size, double alpha, double *x, const int incx);
static int uniform(lua_State *L);
static void _normal(const int size, double alpha, double *x, const int incx);
static int normal(lua_State *L);
static void _inc(const int size, double alpha, double *x, const int incx);
static int inc(lua_State *L);
static int scal(lua_State *L);
static void _pow(const int size, double alpha, double *x, const int incx);
static int powx(lua_State *L);
static void _apply(const int size, double alpha, double *x, const int incx);
static int apply(lua_State *L);

static int dot(lua_State *L);
static int _vector(lua_State *L, vector_function f, int hasddof);
static double _nrm2(int size, const double *values, const int inc, const int ddof);
static int nrm2(lua_State *L);
static double _asum(int size, const double *values, const int inc, const int ddof);
static int asum(lua_State *L);
static double _sum(int size, const double *values, const int inc, const int ddof);
static int sum(lua_State *L);
static double _mean(int size, const double *values, const int inc, const int ddof);
static int mean(lua_State *L);
static double _var(int size, const double *values, const int inc, const int ddof);
static int var(lua_State *L);
static double _std(int size, const double *values, const int inc, const int ddof);
static int std(lua_State *L);
static int iamax(lua_State *L);
static int iamin(lua_State *L);

static int vector_matrix(lua_State *L, vector_matrix_function s, int hasalpha, int hasbeta);
static void _swap(const int size, const double alpha, double *x, int incx, const double beta,
		double *y, int incy);
static int swap(lua_State *L);
static void _copy(const int size, const double alpha, double *x, int incx, const double beta,
		double *y, int incy);
static int copy(lua_State *L);
static void _axpy(const int size, const double alpha, double *x, int incx, const double beta,
		double *y, int incy);
static int axpy(lua_State *L);
static int axpby(lua_State *L);
static void _mul1(const int size, const double alpha, double *x, int incx, const double beta,
		double *y, int incy);
static void _mulm1(const int size, const double alpha, double *x, int incx, const double beta,
		double *y, int incy);
static void _mul(const int size, const double alpha, double *x, int incx, const double beta,
		 double *y, int incy);
static int mul(lua_State *L);

static int gemv(lua_State *L);
static int ger(lua_State *L);
static int gemm(lua_State *L);
static int gesv(lua_State *L);
static int gels(lua_State *L);
static int inv(lua_State *L);
static int det(lua_State *L);

static int cov(lua_State *L);
static int corr(lua_State *L);


static const char *const ORDERS[] = {"row", "col", NULL};
static const char *const TRANSPOSES[] = {"notrans", "trans", NULL};
static __thread lua_State  *TL;


/*
 * arguments
 */

static inline CBLAS_ORDER checkorder (lua_State *L, int index) {
	return luaL_checkoption(L, index, "row", ORDERS) == 0 ? CblasRowMajor : CblasColMajor;
}

static inline CBLAS_TRANSPOSE checktranspose (lua_State *L, int index) {
	return luaL_checkoption(L, index, "notrans", TRANSPOSES) == 0 ? CblasNoTrans : CblasTrans;
}

static inline char lapacktranspose (CBLAS_TRANSPOSE transpose) {
	return transpose == CblasNoTrans ? 'N' : 'T';
}

static int argerror (lua_State *L, int index) {
	return luaL_argerror(L, index, lua_pushfstring(L, "vector, or matrix expected, got %s",
			luaL_typename(L, index)));
}


/*
 * vector
 */

struct vector *create_vector (lua_State *L, size_t size) {
	struct vector  *vector;

	assert(size >= 1 && size <= INT_MAX);
	vector = lua_newuserdata(L, sizeof(struct vector));
	vector->length = size;
	vector->inc = 1;
	vector->values = NULL;
	vector->ref = LUA_NOREF;
	luaL_getmetatable(L, LUALINEAR_VECTOR_METATABLE);
	lua_setmetatable(L, -2);
	vector->values = calloc(size, sizeof(double));
	if (vector->values == NULL) {
		luaL_error(L, "cannot allocate values");
	}
        return vector;
}

struct vector *wrap_vector (lua_State *L, size_t size, double *values) {
	struct vector  *vector;

	assert(size >= 1 && size <= INT_MAX);
	vector = lua_newuserdata(L, sizeof(struct vector));
	vector->length = size;
	vector->inc = 1;
	vector->values = values;
	vector->ref = LUA_REFNIL;
	luaL_getmetatable(L, LUALINEAR_VECTOR_METATABLE);
	lua_setmetatable(L, -2);
	return vector;
}

static int vector (lua_State *L) {
	size_t  size;

	/* process arguments */
	size = luaL_checkinteger(L, 1);
	luaL_argcheck(L, size >= 1 && size <= INT_MAX, 1, "bad dimension");

	/* create */
	create_vector(L, size);
	return 1;
}

static int vector_len (lua_State *L) {
	struct vector  *x;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	lua_pushinteger(L, x->length);
	return 1;
}

static int vector_index (lua_State *L) {
	size_t          index;
	struct vector  *x;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	index = luaL_checkinteger(L, 2);
	luaL_argcheck(L, index >= 1 && index <= x->length, 2, "bad index");
	lua_pushnumber(L, x->values[(index - 1) * x->inc]);
	return 1;
}

static int vector_newindex (lua_State *L) {
	size_t          index;
	double          value;
	struct vector  *x;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	index = luaL_checkinteger(L, 2);
	luaL_argcheck(L, index >= 1 && index <= x->length, 2, "bad index");
	value = luaL_checknumber(L, 3);
	x->values[(index - 1) * x->inc] = value;
	return 0;
}

static int vector_next (lua_State *L) {
	size_t          index;
	struct vector  *x;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	index = luaL_checkinteger(L, 2);
	if (index < x->length) {
		lua_pushinteger(L, index + 1);
		lua_pushnumber(L, x->values[index]);
		return 2;
	}
	lua_pushnil(L);
	return 1;
}

static int vector_ipairs (lua_State *L) {
	luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	lua_pushcfunction(L, vector_next);
	lua_pushvalue(L, 1);
	lua_pushinteger(L, 0);
	return 3;
}

static int vector_tostring (lua_State *L) {
	struct vector  *x;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	lua_pushfstring(L, LUALINEAR_VECTOR_METATABLE ": %p", x);
	return 1;
}

static int vector_gc (lua_State *L) {
	struct vector  *x;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	if (x->ref == LUA_NOREF) {
		free(x->values);
	} else {
		luaL_unref(L, LUA_REGISTRYINDEX, x->ref);
	}
	return 0;
}


/*
 * matrix
 */

struct matrix *create_matrix (lua_State *L, size_t rows, size_t cols, CBLAS_ORDER order) {
	struct matrix  *matrix;

	assert(rows >= 1 && rows <= INT_MAX && cols >= 1 && cols <= INT_MAX);
	matrix = lua_newuserdata(L, sizeof(struct matrix));
	matrix->rows = rows;
	matrix->cols = cols;
	matrix->ld = order == CblasRowMajor ? cols : rows;
	matrix->order = order;
	matrix->values = NULL;
	matrix->ref = LUA_NOREF;
	luaL_getmetatable(L, LUALINEAR_MATRIX_METATABLE);
	lua_setmetatable(L, -2);
	matrix->values = calloc(rows * cols, sizeof(double));
	if (matrix->values == NULL) {
		luaL_error(L, "cannot allocate values");
	}
	return matrix;
}

struct matrix *wrap_matrix (lua_State *L, size_t rows, size_t cols, CBLAS_ORDER order,
		double *values) {
	struct matrix  *matrix;

	assert(rows >= 1 && rows <= INT_MAX && cols >= 1 && cols <= INT_MAX);
	matrix = (struct matrix *)lua_newuserdata(L, sizeof(struct matrix));
	matrix->rows = rows;
	matrix->cols = cols;
	matrix->ld = order == CblasRowMajor ? cols : rows;
	matrix->order = order;
	matrix->values = values;
	matrix->ref = LUA_REFNIL;
	luaL_getmetatable(L, LUALINEAR_MATRIX_METATABLE);
	lua_setmetatable(L, -2);
	return matrix;
}

static int matrix (lua_State *L) {
	size_t       rows, cols;
	CBLAS_ORDER  order;

	/* process arguments */
	rows = luaL_checkinteger(L, 1);
	luaL_argcheck(L, rows >= 1 && rows <= INT_MAX, 1, "bad dimension");
	cols = luaL_checkinteger(L, 2);
	luaL_argcheck(L, cols >= 1 && cols <= INT_MAX, 2, "bad dimension");
	order = checkorder(L, 3);

	/* create */
	create_matrix(L, rows, cols, order);
	return 1;
}

static int matrix_len (lua_State *L) {
	struct matrix  *X;

	X = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	if (X->order == CblasRowMajor) {
		lua_pushinteger(L, X->rows);
	} else {
		lua_pushinteger(L, X->cols);
	}
	return 1;
}

static int matrix_index (lua_State *L) {
	size_t          index, size;
	struct vector  *x;
	struct matrix  *X;

	/* process arguments */
	X = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	index = luaL_checkinteger(L, 2);
	luaL_argcheck(L, index >= 1, 2, "bad index");
	if (X->order == CblasRowMajor) {
		luaL_argcheck(L, index <= X->rows, 2, "bad index");
		size = X->cols;
	} else {
		luaL_argcheck(L, index <= X->cols, 2, "bad index");
		size = X->rows;
	}

	/* create vector */
	x = wrap_vector(L, size, &X->values[(index - 1) * X->ld]);
	lua_pushvalue(L, 1);
	x->ref = luaL_ref(L, LUA_REGISTRYINDEX);
	return 1;
}

static int matrix_next (lua_State *L) {
	size_t          index, majorsize, minorsize;
	struct vector  *x;
	struct matrix  *X;

	X = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
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
		x = wrap_vector(L, minorsize, &X->values[index * X->ld]);
		lua_pushvalue(L, 1);
		x->ref = luaL_ref(L, LUA_REGISTRYINDEX);
		return 2;
	}
	lua_pushnil(L);
	return 1;
}

static int matrix_ipairs (lua_State *L) {
	luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	lua_pushcfunction(L, matrix_next);
	lua_pushvalue(L, 1);
	lua_pushinteger(L, 0);
	return 3;
}

static int matrix_tostring (lua_State *L) {
	struct matrix  *X;

	X = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	lua_pushfstring(L, LUALINEAR_MATRIX_METATABLE ": %p", X);
	return 1;
}

static int matrix_free (lua_State *L) {
	struct matrix  *X;

	X = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	if (X->ref == LUA_NOREF) {
		free(X->values);
	} else {
		luaL_unref(L, LUA_REGISTRYINDEX, X->ref);
	}
	return 0;
}


/*
 * structural functions
 */

static int type (lua_State *L) {
	if (luaL_testudata(L, 1, LUALINEAR_VECTOR_METATABLE) != NULL) {
		lua_pushliteral(L, "vector");
		return 1;
	}
	if (luaL_testudata(L, 1, LUALINEAR_MATRIX_METATABLE) != NULL) {
		lua_pushliteral(L, "matrix");
		return 1;
	}
	lua_pushnil(L);
	return 1;
}

static int size (lua_State *L) {
	struct vector  *x;
	struct matrix  *X;

	x = luaL_testudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	if (x != NULL) {
		lua_pushinteger(L, x->length);
		return 1;
	}
	X = luaL_testudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	if (X != NULL) {
		lua_pushinteger(L, X->rows);
		lua_pushinteger(L, X->cols);
		lua_pushstring(L, ORDERS[X->order == CblasRowMajor ? 0 : 1]);
		return 3;
	}
	return argerror(L, 1);
}

static int tvector (lua_State *L) {
	size_t          index, size;
	struct vector  *x;
	struct matrix  *X;

	/* process arguments */
	X = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	index = luaL_checkinteger(L, 2);
	luaL_argcheck(L, index >= 1, 2, "bad index");
	if (X->order == CblasRowMajor) {
		luaL_argcheck(L, index <= X->cols, 2, "bad index");
		size = X->rows;
	} else {
		luaL_argcheck(L, index <= X->rows, 2, "bad index");
		size = X->cols;
	}

	/* create vector */
	x = wrap_vector(L, size, &X->values[index - 1]);
	x->inc = X->ld;
	lua_pushvalue(L, 1);
	x->ref = luaL_ref(L, LUA_REGISTRYINDEX);
	return 1;
}

static int sub (lua_State *L) {
	struct vector  *x, *s;
	struct matrix  *X, *S;

	x = luaL_testudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	if (x != NULL) {
		size_t  start, end;

		start = luaL_optinteger(L, 2, 1);
		luaL_argcheck(L, start >= 1 && start <= x->length, 2, "bad index");
		end = luaL_optinteger(L, 3, x->length);
		luaL_argcheck(L, end >= start && end <= x->length, 3, "bad index");
		s = wrap_vector(L, end - start + 1, &x->values[(start - 1) * x->inc]);
		s->inc = x->inc;
		lua_pushvalue(L, 1);
		s->ref = luaL_ref(L, LUA_REGISTRYINDEX);
		return 1;
	}
	X = luaL_testudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	if (X != NULL) {
		size_t  rowstart, rowend, colstart, colend;

		if (X->order == CblasRowMajor) {
			rowstart = luaL_optinteger(L, 2, 1);
			luaL_argcheck(L, rowstart >= 1 && rowstart <= X->rows, 2, "bad index");
			colstart = luaL_optinteger(L, 3, 1);
			luaL_argcheck(L, colstart >= 1 && colstart <= X->cols, 3, "bad index");
			rowend = luaL_optinteger(L, 4, X->rows);
			luaL_argcheck(L, rowend >= rowstart && rowend <= X->rows, 4, "bad index");
			colend = luaL_optinteger(L, 5, X->cols);
			luaL_argcheck(L, colend >= colstart && colend <= X->cols, 5, "bad index");
			S = wrap_matrix(L, rowend - rowstart + 1, colend - colstart + 1, X->order,
					&X->values[(rowstart - 1) * X->ld + colstart - 1]);
		} else {
			colstart = luaL_optinteger(L, 2, 1);
			luaL_argcheck(L, colstart >= 1 && colstart <= X->cols, 2, "bad index");
			rowstart = luaL_optinteger(L, 3, 1);
			luaL_argcheck(L, rowstart >= 1 && rowstart <= X->rows, 3, "bad index");
			colend = luaL_optinteger(L, 4, X->cols);
			luaL_argcheck(L, colend >= colstart && colend <= X->cols, 4, "bad index");
			rowend = luaL_optinteger(L, 5, X->rows);
			luaL_argcheck(L, rowend >= rowstart && rowend <= X->rows, 5, "bad index");
			S = wrap_matrix(L, rowend - rowstart + 1, colend - colstart + 1, X->order,
					&X->values[(colstart - 1) * X->ld + rowstart - 1]);
		}
		S->ld = X->ld;
		lua_pushvalue(L, 1);
		S->ref = luaL_ref(L, LUA_REGISTRYINDEX);
		return 1;
	}
	return argerror(L, 1);
}

static int unwind (lua_State *L) {
	int             index;
	size_t          base, i, j, k;
	struct vector  *x;
	struct matrix  *X;

	if (lua_gettop(L) == 0) {
		return luaL_error(L, "wrong number of arguments");
	}
	x = luaL_checkudata(L, lua_gettop(L), LUALINEAR_VECTOR_METATABLE);
	index = 1;
	i = 0;
	while (i < x->length) {
		X = luaL_checkudata(L, index, LUALINEAR_MATRIX_METATABLE);
		luaL_argcheck(L, X->rows * X->cols <= x->length - i, index, "matrix too large");
		if (X->order == CblasRowMajor) {
			for (j = 0; j < X->rows; j++) {
				base = j * X->ld;
				for (k = 0; k < X->cols; k++) {
					x->values[i * x->inc] = X->values[base + k];
					i++;
				}
			}
		} else {
			for (j = 0; j < X->cols; j++) {
				base = j * X->ld;
				for (k = 0; k < X->rows; k++) {
					x->values[i * x->inc] = X->values[base + k];
					i++;
				}
			}
		}
		index++;
	}
	return 0;
}

static int reshape (lua_State *L) {
	int             index;
	size_t          base, i, j, k;
	struct vector  *x;
	struct matrix  *X;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	index = 2;
	i = 0;
	while (i < x->length) {
		X = luaL_checkudata(L, index, LUALINEAR_MATRIX_METATABLE);
		luaL_argcheck(L, X->rows * X->cols <= x->length - i, index, "matrix too large");
		if (X->order == CblasRowMajor) {
			for (j = 0; j < X->rows; j++) {
				base = j * X->ld;
				for (k = 0; k < X->cols; k++) {
					X->values[base + k] = x->values[i * x->inc];
					i++;
				}
			}
		} else {
			for (j = 0; j < X->cols; j++) {
				base = j * X->ld;
				for (k = 0; k < X->rows; k++) {
					X->values[base + k] = x->values[i * x->inc];
					i++;
				}
			}
		}
		index++;
	}
	return 0;
}

static int totable (lua_State *L) {
	size_t          i, j;
	const double   *value;
	struct vector  *x;
	struct matrix  *X;

	x = luaL_testudata(L, 1, LUALINEAR_VECTOR_METATABLE);
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
	X = luaL_testudata(L, 1, LUALINEAR_MATRIX_METATABLE);
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

static int tolinear (lua_State *L) {
	int             isnum;
	double         *value;
	size_t          size, rows, cols, major, minor, i, j;
	CBLAS_ORDER     order;
	struct vector  *x;
	struct matrix  *X;

	luaL_checktype(L, 1, LUA_TTABLE);
	switch (lua_rawgeti(L, 1, 1)) {
	case LUA_TNUMBER:
		size = lua_rawlen(L, 1);
		if (size < 1 || size > INT_MAX) {
			return luaL_error(L, "bad size");
		}
		x = create_vector(L, size);
		value = x->values;
		for (i = 0; i < size; i++) {
			lua_rawgeti(L, 1, i + 1);
			*value++ = lua_tonumberx(L, -1, &isnum);
			if (!isnum) {
				return luaL_error(L, "bad value at index %d", i + 1);
			}
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
		order = checkorder(L, 2);
		if (order == CblasRowMajor) {
			rows = major;
			cols = minor;
		} else {
			rows = minor;
			cols = major;
		}
		X = create_matrix(L, rows, cols, order);
		for (i = 0; i < major; i++) {
			value = &X->values[i * X->ld];
			if (lua_rawgeti(L, 1, i + 1) != LUA_TTABLE) {
				return luaL_error(L, "bad value at index %d", i + 1);
			}
			for (j = 0; j < minor; j++) {
				lua_rawgeti(L, -1, j + 1);
				*value++ = lua_tonumberx(L, -1, &isnum);
				if (!isnum) {
					return luaL_error(L, "bad value at index (%d,%d)", i + 1,
							j + 1);
				}
				lua_pop(L, 1);
			}
			lua_pop(L, 1);
		}
		return 1;

	default:
		return luaL_argerror(L, 1, "bad table");
	}
}


/*
 * elementary functions
 */

static int elementary (lua_State *L, elementary_function f, int hasalpha) {
	size_t          i;
	double          alpha;
	struct vector  *x;
	struct matrix  *X;

	alpha = hasalpha ? luaL_optnumber(L, 2, 1.0) : 0.0;
	x = luaL_testudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	if (x != NULL) {
		f(x->length, alpha, x->values, x->inc);
		return 0;
	}
	X = luaL_testudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	if (X != NULL) {
		if (X->order == CblasRowMajor) {
			for (i = 0; i < X->rows; i++) {
				f(X->cols, alpha, &X->values[i * X->ld], 1);
			}
		} else {
			for (i = 0; i < X->cols; i++) {
				f(X->rows, alpha, &X->values[i * X->ld], 1);
			}
		}
		return 0;
	}
	return argerror(L, 1);
}

static void _sgn (const int size, double alpha, double *x, const int incx) {
	size_t  i;

	(void)alpha;
	for (i = 0; i < (size_t)size; i++) {
		if (*x > 0) {
			*x = 1;
		} else if (*x < 0) {
			*x = -1;
		}
		x += incx;
	}
}

static int sgn (lua_State *L) {
	return elementary(L, _sgn, 0);
}

static void _abs (const int size, double alpha, double *x, const int incx) {
	size_t  i;

	(void)alpha;
	for (i = 0; i < (size_t)size; i++) {
		*x = fabs(*x);
		x += incx;
	}
}

static int absx (lua_State *L) {
	return elementary(L, _abs, 0);
}

static void _exp (const int size, double alpha, double *x, const int incx) {
	size_t  i;

	(void)alpha;
	for (i = 0; i < (size_t)size; i++) {
		*x = exp(*x);
		x += incx;
	}
}

static int expx (lua_State *L) {
	return elementary(L, _exp, 0);
}

static void _log (const int size, double alpha, double *x, const int incx) {
	size_t  i;

	(void)alpha;
	for (i = 0; i < (size_t)size; i++) {
		*x = log(*x);
		x += incx;
	}
}

static int logx (lua_State *L) {
	return elementary(L, _log, 0);
}

static void _logistic (const int size, double alpha, double *x, const int incx) {
	size_t  i;

	(void)alpha;
	for (i = 0; i < (size_t)size; i++) {
		*x = 1.0 / (1.0 + exp(-*x));
		x += incx;
	}
}

static int logistic (lua_State *L) {
	return elementary(L, _logistic, 0);
}

static void _tanh (const int size, double alpha, double *x, const int incx) {
	size_t  i;

	(void)alpha;
	for (i = 0; i < (size_t)size; i++) {
		*x = tanh(*x);
		x += incx;
	}
}

static int tanhx (lua_State *L) {
	return elementary(L, _tanh, 0);
}

static void _softplus (const int size, double alpha, double *x, const int incx) {
	size_t  i;

	(void)alpha;
	for (i = 0; i < (size_t)size; i++) {
		*x = log(1.0 + exp(*x));
		x += incx;
	}
}

static int softplus (lua_State *L) {
	return elementary(L, _softplus, 0);
}

static void _rectifier (const int size, double alpha, double *x, const int incx) {
	size_t  i;

	(void)alpha;
	for (i = 0; i < (size_t)size; i++) {
		*x = *x > 0 ? *x : 0.0;
		x += incx;
	}
}

static int rectifier (lua_State *L) {
	return elementary(L, _rectifier, 0);
}

static void _set (const int size, double alpha, double *x, const int incx) {
	size_t  i;

	for (i = 0; i < (size_t)size; i++) {
		*x = alpha;
		x += incx;
	}

}

static int set (lua_State *L) {
	return elementary(L, _set, 1);
}

static void _uniform (const int size, double alpha, double *x, const int incx) {
	size_t  i;

	(void)alpha;
	for (i = 0; i < (size_t)size; i++) {
		*x = random() / (RAND_MAX + 1.0);
		x += incx;
	}
}

static int uniform (lua_State *L) {
	return elementary(L, _uniform, 0);
}

static void _normal (const int size, double alpha, double *x, const int incx) {
	size_t  i;
	double  u1, u2, r, s, c;

	(void)alpha;

	/* Box-Muller transform */
	for (i = 0; i < (size_t)size - 1; i += 2) {
		u1 = random() / (double)RAND_MAX;
		u2 = random() / (double)RAND_MAX;
		r = sqrt(-2.0 * log(u1));
		sincos(2 * M_PI * u2, &s, &c);
		*x = r * c;
		x += incx;
		*x = r * s;
		x += incx;
	}
	if (i < (size_t)size) {
		u1 = random() / (double)RAND_MAX;
		u2 = random() / (double)RAND_MAX;
		*x = sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);
	}
}

static int normal (lua_State *L) {
	return elementary(L, _normal, 0);
}

static void _inc (const int size, double alpha, double *x, const int incx) {
	size_t  i;

	for (i = 0; i < (size_t)size; i++) {
		*x += alpha;
		x += incx;
	}
}

static int inc (lua_State *L) {
	return elementary(L, _inc, 1);
}

static int scal (lua_State *L) {
	return elementary(L, cblas_dscal, 1);
}

static void _pow (const int size, double alpha, double *x, const int incx) {
	size_t  i;

	for (i = 0; i < (size_t)size; i++) {
		*x = pow(*x, alpha);
		x += incx;
	}
}

static int powx (lua_State *L) {
	return elementary(L, _pow, 1);
}

static void _apply (const int size, double alpha, double *x, const int incx) {
	size_t  i;

	(void)alpha;
	for (i = 0; i < (size_t)size; i++) {
		lua_pushvalue(TL, -1);
		lua_pushnumber(TL, *x);
		lua_call(TL, 1, 1);
		*x = lua_tonumber(TL, -1);
		x += incx;
		lua_pop(TL, 1);
	}
}

static int apply (lua_State *L) {
	luaL_checktype(L, 2, LUA_TFUNCTION);
	lua_settop(L, 2);
	TL = L;
	return elementary(L, _apply, 0);
}


/*
 * vector functions
 */

static int dot (lua_State *L) {
	struct vector  *x, *y;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	y = luaL_checkudata(L, 2, LUALINEAR_VECTOR_METATABLE);
	luaL_argcheck(L, y->length == x->length, 2, "dimension mismatch");
	lua_pushnumber(L, cblas_ddot(x->length, x->values, x->inc, y->values, y->inc));
	return 1;
}

static int _vector (lua_State *L, vector_function f, int hasddof) {
	size_t          i, ddof;
	struct vector  *x, *y;
	struct matrix  *X;

	ddof = 0;
	x = luaL_testudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	if (x != NULL) {
		/* vector */
		if (hasddof) {
			ddof = luaL_optinteger(L, 2, 0);
			luaL_argcheck(L, ddof < x->length, 2, "bad ddof");
		}
		lua_pushnumber(L, f(x->length, x->values, x->inc, ddof));
		return 1;
	}
	X = luaL_testudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	if (X != NULL) {
		/* matrix-vector */
		y = luaL_checkudata(L, 2, LUALINEAR_VECTOR_METATABLE);
		if (checkorder(L, 3) == CblasRowMajor) {
			luaL_argcheck(L, y->length == X->cols, 2, "dimension mismatch");
			if (hasddof) {
				ddof = luaL_optinteger(L, 3, 0);
				luaL_argcheck(L, ddof < X->rows, 3, "bad ddof");
			}
			if (X->order == CblasRowMajor) {
				for (i = 0; i < X->cols; i++) {
					y->values[i * y->inc] = f(X->rows, &X->values[i], X->ld,
							ddof);
				}
			} else {
				for (i = 0; i < X->cols; i++) {
					y->values[i * y->inc] = f(X->rows, &X->values[i * X->ld],
							1, ddof);
				}
			}
		} else {
			luaL_argcheck(L, y->length == X->rows, 2, "dimension mismatch");
			if (hasddof) {
				ddof = luaL_optinteger(L, 3, 0);
				luaL_argcheck(L, ddof < X->cols, 3, "bad ddof");
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
		}
		return 0;
	}
	return argerror(L, 1);
}

static double _nrm2 (int size, const double *x, const int incx, const int ddof) {
	(void)ddof;
	return cblas_dnrm2(size, x, incx);
}

static int nrm2 (lua_State *L) {
	return _vector(L, _nrm2, 0);
}

static double _asum (int size, const double *x, const int incx, const int ddof) {
	(void)ddof;
	return cblas_dasum(size, x, incx);
}

static int asum (lua_State *L) {
	return _vector(L, _asum, 0);
}

/* cblas_dsum does not work as expected */
static double _sum (int size, const double *x, const int incx, const int ddof) {
	size_t  i;
	double  sum;

	(void)ddof;
	sum = 0.0;
	for (i = 0; i < (size_t)size; i++) {
		sum += *x;
		x += incx;
	}
	return sum;
}

static int sum (lua_State *L) {
	return _vector(L, _sum, 0);
}

static double _mean (int size, const double *x, const int incx, const int ddof) {
	size_t  i;
	double  sum;

	(void)ddof;
	sum = 0.0;
	for (i = 0; i < (size_t)size; i++) {
		sum += *x;
		x += incx;
	}
	return sum / size;
}

static int mean (lua_State *L) {
	return _vector(L, _mean, 0);
}

static double _var (int size, const double *x, const int incx, const int ddof) {
	size_t  i;
	double  sum, mean;

	sum = 0.0;
	for (i = 0; i < (size_t)size; i++) {
		sum += x[i * incx];
	}
	mean = sum / size;
	sum = 0.0;
	for (i = 0; i < (size_t)size; i++) {
		sum += (*x - mean) * (*x - mean);
		x += incx;
	}
	return sum / (size - ddof);
}

static int var (lua_State *L) {
	return _vector(L, _var, 1);
}

static double _std (int size, const double *x, const int incx, const int ddof) {
	return sqrt(_var(size, x, incx, ddof));
}

static int std (lua_State *L) {
	return _vector(L, _std, 1);
}

static int iamax (lua_State *L) {
	struct vector  *x;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	lua_pushinteger(L, cblas_idamax(x->length, x->values, x->inc) + 1);
	return 1;
}

static int iamin (lua_State *L) {
	struct vector  *x;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	lua_pushinteger(L, cblas_idamin(x->length, x->values, x->inc) + 1);
	return 1;
}


/*
 * vector-matrix functions
 */

static int vector_matrix (lua_State *L, vector_matrix_function f, int hasalpha, int hasbeta) {
	int             index;
	size_t          i;
	double          alpha, beta;
	struct vector  *x, *y;
	struct matrix  *X, *Y;

	index = 3;
	alpha = hasalpha ? luaL_optnumber(L, index++, 1.0) : 0.0;
	beta = hasbeta ? luaL_optnumber(L, index++, 0.0) : 0.0;
	x = luaL_testudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	if (x != NULL) {
		y = luaL_testudata(L, 2, LUALINEAR_VECTOR_METATABLE);
		if (y != NULL) {
			/* vector-vector */
			luaL_argcheck(L, y->length == x->length, 2, "dimension mismatch");
			f(x->length, alpha, x->values, x->inc, beta, y->values, y->inc);
			return 0;
		}
		Y = luaL_testudata(L, 2, LUALINEAR_MATRIX_METATABLE);
		if (Y != NULL) {
			/* vector-matrix */
			if (checkorder(L, index) == CblasRowMajor) {
				luaL_argcheck(L, 1, x->length == Y->cols, "dimension mismatch");
				if (Y->order == CblasRowMajor) {
					for (i = 0; i < Y->rows; i++) {
						f(x->length, alpha, x->values, x->inc, beta,
								&Y->values[i * Y->ld], 1);
					}
				} else {
					for (i = 0;i < Y->rows; i++) {
						f(x->length, alpha, x->values, x->inc, beta,
								&Y->values[i], Y->ld);
					}
				}
			} else {
				luaL_argcheck(L, 1, x->length == Y->cols, "dimension mismatch");
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
	X = luaL_testudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	if (X != NULL) {
		/* matrix-matrix */
		Y = luaL_checkudata(L, 2, LUALINEAR_MATRIX_METATABLE);
		luaL_argcheck(L, X->order == Y->order, 2, "order mismatch");
		luaL_argcheck(L, X->rows == Y->rows && X->cols == Y->cols, 2, "dimension mismatch");
		if (X->order == CblasRowMajor) {
			for (i = 0; i < X->rows; i++) {
				f(X->cols, alpha, &X->values[i * X->ld], 1, beta,
						&Y->values[i * Y->ld], 1);
			}
		} else {
			for (i = 0; i < X->cols; i++) {
				f(X->rows, alpha, &X->values[i * X->ld], 1, beta,
						&Y->values[i * Y->ld], 1);
			}
		}
		return 0;
	}
	return argerror(L, 1);
}

static void _swap (const int size, const double alpha, double *x, int incx, const double beta,
		double *y, int incy) {
	(void)alpha;
	(void)beta;
	cblas_dswap(size, x, incx, y, incy);
}

static int swap (lua_State *L) {
	return vector_matrix(L, _swap, 0, 0);
}

static void _copy (const int size, const double alpha, double *x, int incx, const double beta,
		double *y, int incy) {
	(void)alpha;
	(void)beta;
	cblas_dcopy(size, x, incx, y, incy);
}

static int copy (lua_State *L) {
	return vector_matrix(L, _copy, 0, 0);
}

static void _axpy (const int size, const double alpha, double *x, int incx, const double beta,
		double *y, int incy) {
	(void)beta;
	cblas_daxpy(size, alpha, x, incx, y, incy);
}
	
static int axpy (lua_State *L) {
	return vector_matrix(L, _axpy, 1, 0);
}

static int axpby (lua_State *L) {
	return vector_matrix(L, (vector_matrix_function)cblas_daxpby, 1, 1);
}

static void _mul1 (const int size, const double alpha, double *x, int incx, const double beta,
		double *y, int incy) {
	int  i;

	(void)alpha;
	(void)beta;
	for (i = 0; i < size; i++) {
		*y *= *x;
		x += incx;
		y += incy;
	}
}

static void _mulm1 (const int size, const double alpha, double *x, int incx, const double beta,
		double *y, int incy) {
	int  i;

	(void)alpha;
	(void)beta;
	for (i = 0; i < size; i++) {
		*y /= *x;
		x += incx;
		y += incy;
	}
}

static void _mul (const int size, const double alpha, double *x, int incx, const double beta,
		double *y, int incy) {
	int  i;

	(void)beta;
	for (i = 0; i < size; i++) {
		*y *= pow(*x, alpha);
		x += incx;
		y += incy;
	}
}

static int mul (lua_State *L) {
	double  alpha;

	alpha = luaL_optnumber(L, 3, 1.0);
	if (alpha == 1.0) {
		return vector_matrix(L, _mul1, 1, 0);
	}
	if (alpha == -1.0) {
		return vector_matrix(L, _mulm1, 1, 0);
	}
	return vector_matrix(L, _mul, 1, 0);
}


/*
 * matrix functions
 */

static int gemv (lua_State *L) {
	size_t           m, n;
	double           alpha, beta;
	struct matrix   *A;
	struct vector   *x, *y;	
	CBLAS_TRANSPOSE  ta;

	/* check and process arguments */
	A = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	x = luaL_checkudata(L, 2, LUALINEAR_VECTOR_METATABLE);
	y = luaL_checkudata(L, 3, LUALINEAR_VECTOR_METATABLE);
	alpha = luaL_optnumber(L, 4, 1.0);
	beta = luaL_optnumber(L, 5, 0.0);
	ta = checktranspose(L, 6);
	m = ta == CblasNoTrans ? A->rows : A->cols;
	n = ta == CblasNoTrans ? A->cols : A->rows;
	luaL_argcheck(L, x->length == n, 2, "dimension mismatch");
	luaL_argcheck(L, y->length == m, 3, "dimension mismatch");

	/* invoke subprogram */
	cblas_dgemv(A->order, ta, A->rows, A->cols, alpha, A->values, A->ld, x->values, x->inc,
			beta, y->values, y->inc);
	return 0;
}

static int ger (lua_State *L) {
	double          alpha;
	struct vector  *x, *y;	
	struct matrix  *A;

	/* check and process arguments */
	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	y = luaL_checkudata(L, 2, LUALINEAR_VECTOR_METATABLE);
	A = luaL_checkudata(L, 3, LUALINEAR_MATRIX_METATABLE);
	alpha = luaL_optnumber(L, 4, 1.0);
	luaL_argcheck(L, x->length == A->rows, 1, "dimension mismatch");
	luaL_argcheck(L, y->length == A->cols, 2, "dimension mismatch");

	/* invoke subprogram */
	cblas_dger(A->order, A->rows, A->cols, alpha, x->values, x->inc, y->values, y->inc,
			A->values, A->ld);
	return 0;
}

static int gemm (lua_State *L) {
	size_t            m, n, ka, kb;
	double            alpha, beta;
	struct matrix    *A, *B, *C;
	CBLAS_TRANSPOSE   ta, tb;

	/* check and process arguments */
	A = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	B = luaL_checkudata(L, 2, LUALINEAR_MATRIX_METATABLE);
	luaL_argcheck(L, B->order == A->order, 2, "order mismatch");
	C = luaL_checkudata(L, 3, LUALINEAR_MATRIX_METATABLE);
	luaL_argcheck(L, C->order == A->order, 3, "order mismatch");
	alpha = luaL_optnumber(L, 4, 1.0);
	beta = luaL_optnumber(L, 5, 0.0);
	ta = checktranspose(L, 6);
	tb = checktranspose(L, 7);
	m = ta == CblasNoTrans ? A->rows : A->cols;
	n = tb == CblasNoTrans ? B->cols : B->rows;
	ka = ta == CblasNoTrans ? A->cols : A->rows;
	kb = tb == CblasNoTrans ? B->rows : B->cols;
	luaL_argcheck(L, ka == kb, 2, "dimension mismatch");

	/* invoke subprogramm */
	cblas_dgemm(A->order, ta, tb, m, n, ka, alpha, A->values, A->ld, B->values, B->ld, beta,
			C->values, C->ld);
	return 0;
}

/* invokes the GESV subprogram */
static int gesv (lua_State *L) {
	int            *ipiv, result;
	struct matrix  *A, *B;

	/* check and process arguments */
	A = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	luaL_argcheck(L, A->rows == A->cols, 1, "not square");
	B = luaL_checkudata(L, 2, LUALINEAR_MATRIX_METATABLE);
	luaL_argcheck(L, B->order == A->order, 2, "order mismatch");
	luaL_argcheck(L, B->rows == A->rows, 2, "dimension mismatch");

	/* invoke subprogramm */
	ipiv = calloc(A->rows, sizeof(lapack_int));
	if (ipiv == NULL) {
		return luaL_error(L, "cannot allocate indexes");
	}
	result = LAPACKE_dgesv(A->order, A->rows, B->cols, A->values, A->ld, ipiv, B->values, B->ld);
	free(ipiv);
	lua_pushinteger(L, result);
	return 1;
}

/* invokes the GELS subprogram */
static int gels (lua_State *L) {
	char            ta;
	struct matrix  *A, *B;

	/* check and process arguments */
	A = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	B = luaL_checkudata(L, 2, LUALINEAR_MATRIX_METATABLE);
	luaL_argcheck(L, B->order == A->order, 2, "order mismatch");
	ta = lapacktranspose(checktranspose(L, 3));
	luaL_argcheck(L, B->rows == (A->rows >= A->cols ? A->rows : A->cols), 2,
			"dimension mismatch");

	/* invoke subprogramm */
	lua_pushinteger(L, LAPACKE_dgels(A->order, ta, A->rows, A->cols, B->cols, A->values, A->ld,
			B->values, B->ld));
	return 1;
}

static int inv (lua_State *L) {
	int            *ipiv, result;
	struct matrix  *A;

	/* check and process arguments */
	A = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	luaL_argcheck(L, A->rows == A->cols, 1, "not square");

	/* invoke subprograms */
	ipiv = calloc(A->rows, sizeof(lapack_int));
	if (ipiv == NULL) {
		return luaL_error(L, "cannot allocate indexes");
	}
	result = LAPACKE_dgetrf(A->order, A->rows, A->cols, A->values, A->ld, ipiv);
	if (result != 0) {
		free(ipiv);
		lua_pushinteger(L, result);
		return 1;
	}
	result = LAPACKE_dgetri(A->order, A->rows, A->values, A->ld, ipiv);
	free(ipiv);
	lua_pushinteger(L, result);
	return 1;
}

static int det (lua_State *L) {
	size_t          n, i;
	int            *ipiv, result, neg;
	struct matrix  *A;
	double         *copy, *d, *s, det;

	/* check and process arguments */
	A = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
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
		lua_pushnumber(L, 0.0);
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


/*
 * statistical functions
 */

static int cov (lua_State *L) {
	size_t          i, j, k, ddof;
	double         *means, *v, *vi, *vj, sum;
	struct matrix  *A, *B;

	/* check and process arguments */
	A = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	B = luaL_checkudata(L, 2, LUALINEAR_MATRIX_METATABLE);
	luaL_argcheck(L, A->cols == B->rows, 2, "dimension mismatch");
	luaL_argcheck(L, B->rows == B->cols, 2, "not square");
	ddof = luaL_optinteger(L, 3, 0);
	luaL_argcheck(L, ddof < A->rows, 3, "bad ddof");

	/* calculate means */
	means = calloc(A->cols, sizeof(double));
	if (means == NULL) {
		return luaL_error(L, "cannot allocate values");
	}
	if (A->order == CblasRowMajor) {
		for (i = 0; i < A->cols; i++) {
			sum = 0.0;
			v = &A->values[i];
			for (j = 0; j < A->rows; j++) {
				sum += *v;
				v += A->ld;
			}
			means[i] = sum / A->rows;
		}
	} else {
		for (i = 0; i < A->cols; i++) {
			sum = 0.0;
			v = &A->values[i * A->ld];
			for (j = 0; j < A->rows; j++) {
				sum += *v;
				v++;
			}
			means[i] = sum / A->rows;
		}
	}

	/* calculate covariance */
	if (A->order == CblasRowMajor) {
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
	} else {
		for (i = 0; i < A->cols; i++) {
			for (j = i; j < A->cols; j++) {
				sum = 0.0;
				vi = &A->values[i * A->ld];
				vj = &A->values[j * A->ld];
				for (k = 0; k < A->rows; k++) {
					sum += (*vi - means[i]) * (*vj - means[j]);
					vi++;
					vj++;
				}
				B->values[i * B->ld + j] = B->values[j * B->ld + i] = sum
						/ (A->rows - ddof);
			}
		}
	}
	free(means);
	return 0;
}

static int corr (lua_State *L) {
	struct matrix  *A, *B;
	size_t          i, j, k;
	double         *means, *stds, *v, *vi, *vj, sum;

	/* check and process arguments */
	A = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	B = luaL_checkudata(L, 2, LUALINEAR_MATRIX_METATABLE);
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
	if (A->order == CblasRowMajor) {
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
	} else {
		for (i = 0; i < A->cols; i++) {
			sum = 0.0;
			v = &A->values[i * A->ld];
			for (j = 0; j < A->rows; j++) {
				sum += *v;
				v++;
			}
			means[i] = sum / A->rows;
			sum = 0.0;
			v = &A->values[i * A->ld];
			for (j = 0; j < A->rows; j++) {
				sum += (*v - means[i]) * (*v - means[i]);
				v++;
			}
			stds[i] = sqrt(sum);
		}
	}

	/* calculate correlation */
	if (A->order == CblasRowMajor) {
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
	} else {
		for (i = 0; i < A->cols; i++) {
			for (j = i; j < A->cols; j++) {
				sum = 0.0;
				vi = &A->values[i * A->ld];
				vj = &A->values[j * A->ld];
				for (k = 0; k < A->rows; k++) {
					sum += (*vi - means[i]) * (*vj - means[j]);
					vi++;
					vj++;
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
		{ "vector", vector },
		{ "matrix", matrix },
		{ "type", type },
		{ "size", size },
		{ "tvector", tvector },
		{ "sub", sub },
		{ "unwind", unwind },
		{ "reshape", reshape },
		{ "totable", totable },
		{ "tolinear", tolinear },

		/* elementary functions */
		{ "sgn", sgn },
		{ "abs", absx },
		{ "exp", expx },
		{ "log", logx },
		{ "logistic", logistic },
		{ "tanh", tanhx },
		{ "softplus", softplus },
		{ "rectifier", rectifier },
		{ "apply", apply },
		{ "set", set },
		{ "uniform", uniform },
		{ "normal", normal },
		{ "inc", inc },
		{ "pow", powx },
		{ "scal", scal },

		/* vector functions */
		{ "dot", dot },
		{ "nrm2", nrm2 },
		{ "asum", asum },
		{ "sum", sum },
		{ "mean", mean },
		{ "var", var },
		{ "std", std },
		{ "iamax", iamax },
		{ "iamin", iamin },

		/* vector-matrix functions */
		{ "swap", swap },
		{ "copy", copy },
		{ "axpy", axpy },
		{ "axpby", axpby },
		{ "mul", mul },

		/* matrix functions */
		{ "gemv", gemv },
		{ "ger", ger },
		{ "gemm", gemm },
		{ "gesv", gesv },
		{ "gels", gels },
		{ "inv", inv },
		{ "det", det },

		/* statistical functions */
		{ "cov", cov },
		{ "corr", corr },

		{ NULL, NULL }
	};

	/* register functions */
	#if LUA_VERSION_NUM >= 502
	luaL_newlib(L, FUNCTIONS);
	#else
	luaL_register(L, luaL_checkstring(L, 1), FUNCTIONS);
	#endif

	/* vector metatable */
	luaL_newmetatable(L, LUALINEAR_VECTOR_METATABLE);
	lua_pushcfunction(L, vector_len);
	lua_setfield(L, -2, "__len");
	lua_pushcfunction(L, vector_index);
	lua_setfield(L, -2, "__index");
	lua_pushcfunction(L, vector_newindex);
	lua_setfield(L, -2, "__newindex");
	lua_pushcfunction(L, vector_ipairs);
	lua_setfield(L, -2, "__ipairs");
	lua_pushcfunction(L, vector_tostring);
	lua_setfield(L, -2, "__tostring");
	lua_pushcfunction(L, vector_gc);
	lua_setfield(L, -2, "__gc");
	lua_pop(L, 1);

	/* matrix metatable */
	luaL_newmetatable(L, LUALINEAR_MATRIX_METATABLE);
	lua_pushcfunction(L, matrix_len);
	lua_setfield(L, -2, "__len");
	lua_pushcfunction(L, matrix_index);
	lua_setfield(L, -2, "__index");
	lua_pushcfunction(L, matrix_ipairs);
	lua_setfield(L, -2, "__ipairs");
	lua_pushcfunction(L, matrix_tostring);
	lua_setfield(L, -2, "__tostring");
	lua_pushcfunction(L, matrix_free);
	lua_setfield(L, -2, "__gc");
	lua_pop(L, 1);

	return 1;
}
