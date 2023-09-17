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
typedef double (*unary_function)(const int size, const double *x, const int incx, const int ddof);
typedef void (*binary_function)(const int size, const double alpha, double *x,
		const int incx, const double beta, double *y, const int incy);


static inline CBLAS_ORDER checkorder(lua_State *L, int index);
static inline CBLAS_TRANSPOSE checktranspose(lua_State *L, int index);
static inline char lapacktranspose(CBLAS_TRANSPOSE transpose);
static int argerror(lua_State *L, int index);

static struct vector *create_vector(lua_State *L, size_t length);
static void push_vector(lua_State *L, size_t length, size_t inc, struct data *data, double *values);
static int vector(lua_State *L);
static int vector_len(lua_State *L);
static int vector_index(lua_State *L);
static int vector_newindex(lua_State *L);
static int vector_next(lua_State *L);
static int vector_ipairs(lua_State *L);
static int vector_tostring(lua_State *L);
static int vector_gc(lua_State *L);

static struct matrix *create_matrix(lua_State *L, size_t rows, size_t cols, CBLAS_ORDER order);
static void push_matrix(lua_State *L, size_t rows, size_t cols, size_t ld, CBLAS_ORDER order,
		struct data *data, double *values);
static int matrix(lua_State *L);
static int matrix_len(lua_State *L);
static int matrix_index(lua_State *L);
static int matrix_next(lua_State *L);
static int matrix_ipairs(lua_State *L);
static int matrix_tostring(lua_State *L);
static int matrix_gc(lua_State *L);

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

static int unary(lua_State *L, unary_function f, int hasddof);
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

static int binary(lua_State *L, binary_function s, int hasalpha, int hasbeta);
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
static void _mul(const int size, const double alpha, double *x, int incx, const double beta,
		 double *y, int incy);
static int mul(lua_State *L);

static int dot(lua_State *L);
static int ger(lua_State *L);
static int gemv(lua_State *L);
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

static struct vector *create_vector (lua_State *L, size_t length) {
	struct vector  *vector;

	assert(length >= 1 && length <= INT_MAX);
	vector = lua_newuserdata(L, sizeof(struct vector));
	vector->length = length;
	vector->inc = 1;
	vector->data = NULL;
	luaL_getmetatable(L, LUALINEAR_VECTOR_METATABLE);
	lua_setmetatable(L, -2);
	vector->data = calloc(1, sizeof(struct data) + length * sizeof(double));
	if (vector->data == NULL) {
		luaL_error(L, "cannot allocate data");
	}
	vector->data->refs = 1;
	vector->values = (double *)((char *)vector->data + sizeof(struct data));
	return vector;
}

static void push_vector (lua_State *L, size_t length, size_t inc, struct data *data,
		double *values) {
	struct vector  *vector;

	assert(length >= 1 && length <= INT_MAX);
	vector = lua_newuserdata(L, sizeof(struct vector));
	vector->length = length;
	vector->inc = inc;
	vector->data = NULL;
	luaL_getmetatable(L, LUALINEAR_VECTOR_METATABLE);
	lua_setmetatable(L, -2);
	vector->data = data;
	data->refs++;
	vector->values = values;
}

static int vector (lua_State *L) {
	size_t  size;

	size = luaL_checkinteger(L, 1);
	luaL_argcheck(L, size >= 1 && size <= INT_MAX, 1, "bad dimension");
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
		lua_pushnumber(L, x->values[index * x->inc]);
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

static struct matrix *create_matrix (lua_State *L, size_t rows, size_t cols, CBLAS_ORDER order) {
	struct matrix  *matrix;

	assert(rows >= 1 && rows <= INT_MAX && cols >= 1 && cols <= INT_MAX);
	matrix = lua_newuserdata(L, sizeof(struct matrix));
	matrix->rows = rows;
	matrix->cols = cols;
	matrix->ld = order == CblasRowMajor ? cols : rows;
	matrix->order = order;
	matrix->data = NULL;
	luaL_getmetatable(L, LUALINEAR_MATRIX_METATABLE);
	lua_setmetatable(L, -2);
	matrix->data = calloc(1, sizeof(struct data) + rows * cols * sizeof(double));
	if (matrix->data == NULL) {
		luaL_error(L, "cannot allocate data");
	}
	matrix->data->refs = 1;
	matrix->values = (double *)((char *)matrix->data + sizeof(struct data));
	return matrix;
}

static void push_matrix (lua_State *L, size_t rows, size_t cols, size_t ld, CBLAS_ORDER order,
		struct data *data, double *values) {
	struct matrix  *matrix;

	assert(rows >= 1 && rows <= INT_MAX && cols >= 1 && cols <= INT_MAX);
	matrix = (struct matrix *)lua_newuserdata(L, sizeof(struct matrix));
	matrix->rows = rows;
	matrix->cols = cols;
	matrix->ld = ld;
	matrix->order = order;
	matrix->data = NULL;
	luaL_getmetatable(L, LUALINEAR_MATRIX_METATABLE);
	lua_setmetatable(L, -2);
	matrix->data = data;
	data->refs++;
	matrix->values = values;
}

static int matrix (lua_State *L) {
	size_t       rows, cols;
	CBLAS_ORDER  order;

	rows = luaL_checkinteger(L, 1);
	luaL_argcheck(L, rows >= 1 && rows <= INT_MAX, 1, "bad dimension");
	cols = luaL_checkinteger(L, 2);
	luaL_argcheck(L, cols >= 1 && cols <= INT_MAX, 2, "bad dimension");
	order = checkorder(L, 3);
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
	size_t          index, length;
	struct matrix  *X;

	X = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	index = luaL_checkinteger(L, 2);
	if (X->order == CblasRowMajor) {
		luaL_argcheck(L, index >= 1 && index <= X->rows, 2, "bad index");
		length = X->cols;
	} else {
		luaL_argcheck(L, index >= 1 && index <= X->cols, 2, "bad index");
		length = X->rows;
	}
	push_vector(L, length, 1, X->data, &X->values[(index - 1) * X->ld]);
	return 1;
}

static int matrix_next (lua_State *L) {
	size_t          index, majorsize, minorsize;
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
		push_vector(L, minorsize, 1, X->data, &X->values[index * X->ld]);
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

static int matrix_gc (lua_State *L) {
	struct matrix  *X;

	X = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
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
	size_t          index, length;
	struct matrix  *X;

	X = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	index = luaL_checkinteger(L, 2);
	if (X->order == CblasRowMajor) {
		luaL_argcheck(L, index >= 1 && index <= X->cols, 2, "bad index");
		length = X->rows;
	} else {
		luaL_argcheck(L, index >= 1 && index <= X->rows, 2, "bad index");
		length = X->cols;
	}
	push_vector(L, length, X->ld, X->data, &X->values[index - 1]);
	return 1;
}

static int sub (lua_State *L) {
	struct vector  *x;
	struct matrix  *X;

	x = luaL_testudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	if (x != NULL) {
		size_t  start, end;

		start = luaL_optinteger(L, 2, 1);
		luaL_argcheck(L, start >= 1 && start <= x->length, 2, "bad index");
		end = luaL_optinteger(L, 3, x->length);
		luaL_argcheck(L, end >= start && end <= x->length, 3, "bad index");
		push_vector(L, end - start + 1, x->inc, x->data, &x->values[(start - 1) * x->inc]);
		return 1;
	}
	X = luaL_testudata(L, 1, LUALINEAR_MATRIX_METATABLE);
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
			push_matrix(L, rowend - rowstart + 1, colend - colstart + 1, X->ld,
					X->order, X->data, &X->values[(rowstart - 1) * X->ld
					+ (colstart - 1)]);
		} else {
			push_matrix(L, rowend - rowstart + 1, colend - colstart + 1, X->ld,
					X->order, X->data, &X->values[(colstart - 1) * X->ld
					+ (rowstart - 1)]);
		}
		return 1;
	}
	return argerror(L, 1);
}

static int unwind (lua_State *L) {
	int             index;
	double         *s, *d, *last;
	size_t          i, j;
	struct vector  *x;
	struct matrix  *X;

	if (lua_gettop(L) == 0) {
		return luaL_error(L, "wrong number of arguments");
	}
	x = luaL_checkudata(L, lua_gettop(L), LUALINEAR_VECTOR_METATABLE);
	d = x->values;
	last = d + x->length;
	index = 1;
	while (d < last) {
		X = luaL_checkudata(L, index, LUALINEAR_MATRIX_METATABLE);
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

static int reshape (lua_State *L) {
	int             index;
	double         *d, *s, *last;
	size_t          i, j;
	struct vector  *x;
	struct matrix  *X;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	s = x->values;
	last = x->values + x->length;
	index = 2;
	while (s < last) {
		X = luaL_checkudata(L, index, LUALINEAR_MATRIX_METATABLE);
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
	int             isnum;
	size_t          i;
	double          alpha, n;
	struct vector  *x;
	struct matrix  *X;

	alpha = hasalpha ? luaL_optnumber(L, 2, 1.0) : 0.0;
	n = lua_tonumberx(L, 1, &isnum);
	if (isnum) {
		f(1, alpha, &n, 1);
		lua_pushnumber(L, n);
		return 1;
	}
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

	if (alpha == -1.0) {
		for (i = 0; i < (size_t)size; i++) {
			*x = 1 / *x;
			x += incx;
		}
	} else if (alpha == 0.0) {
		for (i = 0; i < (size_t)size; i++) {
			*x = 1.0;
			x += incx;
		}
	} else if (alpha != 1.0) {
		for (i = 0; i < (size_t)size; i++) {
			*x = pow(*x, alpha);
			x += incx;
		}
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
 * unary vector functions
 */

static int unary (lua_State *L, unary_function f, int hasddof) {
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

static double _nrm2 (int size, const double *x, const int incx, const int ddof) {
	(void)ddof;
	return cblas_dnrm2(size, x, incx);
}

static int nrm2 (lua_State *L) {
	return unary(L, _nrm2, 0);
}

static double _asum (int size, const double *x, const int incx, const int ddof) {
	(void)ddof;
	return cblas_dasum(size, x, incx);
}

static int asum (lua_State *L) {
	return unary(L, _asum, 0);
}

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
	return unary(L, _sum, 0);
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
	return unary(L, _mean, 0);
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
	return unary(L, _var, 1);
}

static double _std (int size, const double *x, const int incx, const int ddof) {
	return sqrt(_var(size, x, incx, ddof));
}

static int std (lua_State *L) {
	return unary(L, _std, 1);
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
 * binary vector functions
 */

static int binary (lua_State *L, binary_function f, int hasalpha, int hasbeta) {
	size_t          i;
	double          alpha, beta;
	struct vector  *x, *y;
	struct matrix  *X, *Y;

	x = luaL_testudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	if (x != NULL) {
		y = luaL_testudata(L, 2, LUALINEAR_VECTOR_METATABLE);
		if (y != NULL) {
			/* vector-vector */
			luaL_argcheck(L, y->length == x->length, 2, "dimension mismatch");
			alpha = hasalpha ? luaL_optnumber(L, 3, 1.0) : 0.0;
			beta = hasbeta ? luaL_optnumber(L, 4, 0.0) : 0.0;
			f(x->length, alpha, x->values, x->inc, beta, y->values, y->inc);
			return 0;
		}
		Y = luaL_testudata(L, 2, LUALINEAR_MATRIX_METATABLE);
		if (Y != NULL) {
			/* vector-matrix */
			alpha = hasalpha ? luaL_optnumber(L, 4, 1.0) : 0.0;
			beta = hasbeta ? luaL_optnumber(L, 5, 0.0) : 0.0;
			if (checkorder(L, 3) == CblasRowMajor) {
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
		alpha = hasalpha ? luaL_optnumber(L, 3, 1.0) : 0.0;
		beta = hasbeta ? luaL_optnumber(L, 4, 0.0) : 0.0;
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
	return binary(L, _swap, 0, 0);
}

static void _copy (const int size, const double alpha, double *x, int incx, const double beta,
		double *y, int incy) {
	(void)alpha;
	(void)beta;
	cblas_dcopy(size, x, incx, y, incy);
}

static int copy (lua_State *L) {
	return binary(L, _copy, 0, 0);
}

static void _axpy (const int size, const double alpha, double *x, int incx, const double beta,
		double *y, int incy) {
	(void)beta;
	cblas_daxpy(size, alpha, x, incx, y, incy);
}
	
static int axpy (lua_State *L) {
	return binary(L, _axpy, 1, 0);
}

static int axpby (lua_State *L) {
	return binary(L, (binary_function)cblas_daxpby, 1, 1);
}

static void _mul (const int size, const double alpha, double *x, int incx, const double beta,
		double *y, int incy) {
	int  i;

	(void)beta;
	if (alpha == 1.0) {
		for (i = 0; i < size; i++) {
			*y *= *x;
			x += incx;
			y += incy;
		}
	} else if (alpha == -1.0) {
		for (i = 0; i < size; i++) {
			*y /= *x;
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

static int mul (lua_State *L) {
	return binary(L, _mul, 1, 0);
}


/*
 * program functions
 */

static int dot (lua_State *L) {
	struct vector  *x, *y;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	y = luaL_checkudata(L, 2, LUALINEAR_VECTOR_METATABLE);
	luaL_argcheck(L, y->length == x->length, 2, "dimension mismatch");
	lua_pushnumber(L, cblas_ddot(x->length, x->values, x->inc, y->values, y->inc));
	return 1;
}

static int ger (lua_State *L) {
	double          alpha;
	struct vector  *x, *y;
	struct matrix  *A;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	y = luaL_checkudata(L, 2, LUALINEAR_VECTOR_METATABLE);
	A = luaL_checkudata(L, 3, LUALINEAR_MATRIX_METATABLE);
	luaL_argcheck(L, x->length == A->rows, 1, "dimension mismatch");
	luaL_argcheck(L, y->length == A->cols, 2, "dimension mismatch");
	alpha = luaL_optnumber(L, 4, 1.0);
	cblas_dger(A->order, A->rows, A->cols, alpha, x->values, x->inc, y->values, y->inc,
			A->values, A->ld);
	return 0;
}

static int gemv (lua_State *L) {
	size_t           m, n;
	double           alpha, beta;
	struct matrix   *A;
	struct vector   *x, *y;	
	CBLAS_TRANSPOSE  ta;

	A = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	x = luaL_checkudata(L, 2, LUALINEAR_VECTOR_METATABLE);
	y = luaL_checkudata(L, 3, LUALINEAR_VECTOR_METATABLE);
	ta = checktranspose(L, 4);
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

static int gemm (lua_State *L) {
	size_t            m, n, ka, kb;
	double            alpha, beta;
	struct matrix    *A, *B, *C;
	CBLAS_TRANSPOSE   ta, tb;

	A = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	B = luaL_checkudata(L, 2, LUALINEAR_MATRIX_METATABLE);
	luaL_argcheck(L, B->order == A->order, 2, "order mismatch");
	C = luaL_checkudata(L, 3, LUALINEAR_MATRIX_METATABLE);
	luaL_argcheck(L, C->order == A->order, 3, "order mismatch");
	ta = checktranspose(L, 4);
	tb = checktranspose(L, 5);
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

static int gesv (lua_State *L) {
	int            *ipiv, result;
	struct matrix  *A, *B;

	A = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	luaL_argcheck(L, A->rows == A->cols, 1, "not square");
	B = luaL_checkudata(L, 2, LUALINEAR_MATRIX_METATABLE);
	luaL_argcheck(L, B->order == A->order, 2, "order mismatch");
	luaL_argcheck(L, B->rows == A->rows, 2, "dimension mismatch");
	ipiv = calloc(A->rows, sizeof(lapack_int));
	if (ipiv == NULL) {
		return luaL_error(L, "cannot allocate indexes");
	}
	result = LAPACKE_dgesv(A->order, A->rows, B->cols, A->values, A->ld, ipiv, B->values, B->ld);
	free(ipiv);
	lua_pushinteger(L, result);
	return 1;
}

static int gels (lua_State *L) {
	char            ta;
	struct matrix  *A, *B;

	A = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	B = luaL_checkudata(L, 2, LUALINEAR_MATRIX_METATABLE);
	luaL_argcheck(L, B->order == A->order, 2, "order mismatch");
	ta = lapacktranspose(checktranspose(L, 3));
	luaL_argcheck(L, B->rows == (A->rows >= A->cols ? A->rows : A->cols), 2,
			"dimension mismatch");
	lua_pushinteger(L, LAPACKE_dgels(A->order, ta, A->rows, A->cols, B->cols, A->values, A->ld,
			B->values, B->ld));
	return 1;
}

static int inv (lua_State *L) {
	int            *ipiv, result;
	struct matrix  *A;

	A = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	luaL_argcheck(L, A->rows == A->cols, 1, "not square");
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

	/* calculate covariances */
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

	/* calculate Pearson correlations */
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

		/* unary vector functions */
		{ "nrm2", nrm2 },
		{ "asum", asum },
		{ "sum", sum },
		{ "mean", mean },
		{ "var", var },
		{ "std", std },
		{ "iamax", iamax },
		{ "iamin", iamin },

		/* binary vector functions */
		{ "swap", swap },
		{ "copy", copy },
		{ "axpy", axpy },
		{ "axpby", axpby },
		{ "mul", mul },

		/* program functions */
		{ "dot", dot },
		{ "ger", ger },
		{ "gemv", gemv },
		{ "gemm", gemm },
		{ "gesv", gesv },
		{ "gels", gels },
		{ "inv", inv },
		{ "det", det },
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
	lua_pushcfunction(L, matrix_gc);
	lua_setfield(L, -2, "__gc");
	lua_pop(L, 1);

	return 1;
}
