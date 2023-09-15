#include "linear.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <lauxlib.h>
#include <cblas.h>
#include <lapacke.h>


typedef double (*v_function)(int n, const double *x, int incx);
typedef void (*vm_function)(size_t, double *, int, double *, int, double);
typedef double (*unary_function)(double);

static inline CBLAS_ORDER checkorder(lua_State *L, int index);
static inline CBLAS_TRANSPOSE checktranspose(lua_State *L, int index);
static inline char lapacktranspose(CBLAS_TRANSPOSE transpose);
static int intvalue(lua_State *L, const char *key, int dfl);
static int optionvalue(lua_State *L, const char *key, const char *dfl, const char *const options[]);
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

static int dot(lua_State *L);
static int v(lua_State *L, v_function f);
static int nrm2(lua_State *L);
static int asum(lua_State *L);
static double _sum(int size, const double *values, int inc);
static int sum(lua_State *L);
static int iamax(lua_State *L);
static int iamin(lua_State *L);
static int imax(lua_State *L);
static int imin(lua_State *L);

static int xy(lua_State *L, vm_function s, int hasy, int hasalpha);
static void _swap(size_t size, double *x, int incx, double *y, int incy, double alpha);
static int swap(lua_State *L);
static void _copy(size_t size, double *x, int incx, double *y, int incy, double alpha);
static int copy(lua_State *L);
static void _axpy(size_t size, double *x, int incx, double *y, int incy, double alpha);
static int axpy(lua_State *L);
static void _scal(size_t size, double *x, int incx, double *y, int incy, double alpha);
static int scal(lua_State *L);
static void _set(size_t size, double *x, int incx, double *y, int incy, double alpha);
static int set(lua_State *L);
static void _uniform(size_t size, double *x, int incx, double *y, int incy, double alpha);
static int uniform(lua_State *L);
static void _normal(size_t size, double *x, int incx, double *y, int incy, double alpha);
static int normal(lua_State *L);
static void _inc(size_t size, double *x, int incx, double *y, int incy, double alpha);
static int inc(lua_State *L);
static void _mul1(size_t size, double *x, int incx, double *y, int incy, double alpha);
static void _mulm1(size_t size, double *x, int incx, double *y, int incy, double alpha);
static void _mul(size_t size, double *x, int incx, double *y, int incy, double alpha);
static int mul(lua_State *L);
static void _pow(size_t size, double *x, int incx, double *y, int incy, double alpha);
static int powx(lua_State *L);

static int apply(lua_State *L, unary_function apply, int parallel);
static double _sign(double x);
static int sign(lua_State *L);
static int absx(lua_State *L);
static int expx(lua_State *L);
static int logx(lua_State *L);
static double _logistic(double z);
static int logistic(lua_State *L);
static int tanhx(lua_State *L);
static double _softplus(double x);
static int softplus(lua_State *L);
static double _rectifier(double x);
static int rectifier(lua_State *L);
static double _apply(double x);
static int applyx(lua_State *L);

static int gemv(lua_State *L);
static int ger(lua_State *L);
static int gemm(lua_State *L);
static int gesv(lua_State *L);
static int gels(lua_State *L);
static int inv(lua_State *L);
static int det(lua_State *L);

static int cov(lua_State *L);
static int corr(lua_State *L);


static const char *const ORDERS[] = {"row", "col", "rowmajor", "colmajor", NULL};
static const char *const TRANSPOSES[] = {"notrans", "trans", NULL};
static const char *const TYPES[] = {"vector", "matrix", NULL};


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

static int intvalue (lua_State *L, const char *key, int dfl) {
	int  result, isint;

	lua_getfield(L, -1, key);
	if (!lua_isnil(L, -1)) {
		result = lua_tointegerx(L, -1, &isint);
		if (!isint) {
			luaL_error(L, "bad field " LUA_QS, key);
		}
	} else {
		if (dfl < 0) {
			luaL_error(L, "missing field " LUA_QS, key);
		}
		result = dfl;
	}
	lua_pop(L, 1);
	return result;
}

static int optionvalue (lua_State *L, const char *key, const char *dfl,
		const char *const options[]) {
	const char  *str;
	int          i;
	
	lua_getfield(L, -1, key);
	if (!lua_isnil(L, -1)) {
		str = lua_tostring(L, -1);
		if (str == NULL) {
			luaL_error(L, "bad field " LUA_QS, key);
		}
	} else {
		if (dfl == NULL) {
			luaL_error(L, "missing field " LUA_QS, key);
		}
		str = dfl;
	}
	lua_pop(L, 1);
	for (i = 0; options[i] != NULL; i++) {
		if (strcmp(options[i], str) == 0) {
			return i;
		}
	}
	return luaL_error(L, "bad option " LUA_QS " in field " LUA_QS, str, key);
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
	vector->size = size;
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
	vector->size = size;
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
	lua_pushinteger(L, x->size);
	return 1;
}

static int vector_index (lua_State *L) {
	size_t          index;
	struct vector  *x;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	index = luaL_checkinteger(L, 2);
	luaL_argcheck(L, index >= 1 && index <= x->size, 2, "bad index");
	lua_pushnumber(L, x->values[(index - 1) * x->inc]);
	return 1;
}

static int vector_newindex (lua_State *L) {
	size_t          index;
	double          value;
	struct vector  *x;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	index = luaL_checkinteger(L, 2);
	luaL_argcheck(L, index >= 1 && index <= x->size, 2, "bad index");
	value = luaL_checknumber(L, 3);
	x->values[(index - 1) * x->inc] = value;
	return 0;
}

static int vector_next (lua_State *L) {
	size_t          index;
	struct vector  *x;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	index = luaL_checkinteger(L, 2);
	if (index < x->size) {
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
	lua_pushfstring(L, "vector: %p", x);
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
	lua_pushfstring(L, "matrix: %p", X);
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
		lua_pushinteger(L, x->size);
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
		luaL_argcheck(L, start >= 1 && start <= x->size, 2, "bad index");
		end = luaL_optinteger(L, 3, x->size);
		luaL_argcheck(L, end >= start && end <= x->size, 3, "bad index");
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
	while (i < x->size) {
		X = luaL_checkudata(L, index, LUALINEAR_MATRIX_METATABLE);
		luaL_argcheck(L, X->rows * X->cols <= x->size - i, index, "matrix too large");
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
	while (i < x->size) {
		X = luaL_checkudata(L, index, LUALINEAR_MATRIX_METATABLE);
		luaL_argcheck(L, X->rows * X->cols <= x->size - i, index, "matrix too large");
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
		lua_createtable(L, 0, 3);
		lua_pushliteral(L, "vector");
		lua_setfield(L, -2, "type");
		lua_pushinteger(L, x->size);
		lua_setfield(L, -2, "length");
		lua_createtable(L, x->size, 0);
		value = x->values;
		for (i = 0; i < x->size; i++) {
			lua_pushnumber(L, *value);
			lua_rawseti(L, -2, i + 1);
			value += x->inc;
		}
		lua_setfield(L, -2, "values");
		return 1;
	}
	X = luaL_testudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	if (X != NULL) {
		lua_createtable(L, 0, 5);
		lua_pushliteral(L, "matrix");
		lua_setfield(L, -2, "type");
		lua_pushinteger(L, X->rows);
		lua_setfield(L, -2, "rows");
		lua_pushinteger(L, X->cols);
		lua_setfield(L, -2, "cols");
		if (X->order == CblasRowMajor) {
			lua_pushstring(L, ORDERS[0]);
			lua_setfield(L, -2, "order");
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
			lua_setfield(L, -2, "values");
		} else {
			lua_pushstring(L, ORDERS[1]);
			lua_setfield(L, -2, "order");
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
			lua_setfield(L, -2, "values");
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

	/* check arguments */
	luaL_checktype(L, 1, LUA_TTABLE);
	lua_settop(L, 1);

	/* handle types */
	if (optionvalue(L, "type", NULL, TYPES) == 0) {
		size = intvalue(L, "length", -1);
		if (size < 1) {
			return luaL_error(L, "bad field " LUA_QS, "length");
		}
		x = create_vector(L, size);
		lua_getfield(L, 1, "values");
		if (lua_type(L, -1) != LUA_TTABLE) {
			return luaL_error(L, "bad field " LUA_QS, "values");
		}
		value = x->values;
		for (i = 0; i < size; i++) {
			lua_rawgeti(L, -1, i + 1);
			*value++ = lua_tonumberx(L, -1, &isnum);
			if (!isnum) {
				return luaL_error(L, "bad value at index %d", i + 1);
			}
			lua_pop(L, 1);
		}
		lua_pop(L, 1);
	} else {
		rows = intvalue(L, "rows", -1);
		if (rows < 1) {
			return luaL_error(L, "bad field " LUA_QS, "rows");
		}
		cols = intvalue(L, "cols", -1);
		if (cols < 1) {
			return luaL_error(L, "bad field " LUA_QS, "cols");
		}
		order = optionvalue(L, "order", NULL, ORDERS);
		if (order >=2) {
			order -=2 ;
		}
		if (order == 0) {
			order = CblasRowMajor;
			major = rows;
			minor = cols;
		} else {
			order = CblasColMajor;
			major = cols;
			minor = rows;
		}
		X = create_matrix(L, rows, cols, order);
		lua_getfield(L, 1, "values");
		if (lua_type(L, -1) != LUA_TTABLE) {
			return luaL_error(L, "bad field " LUA_QS, "values");
		}
		for (i = 0; i < major; i++) {
			value = &X->values[i * X->ld];
			lua_rawgeti(L, -1, i + 1);
			if (lua_type(L, -1) != LUA_TTABLE) {
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
		lua_pop(L, 1);
	}
	return 1;
}


/*
 * vector functions
 */

static int dot (lua_State *L) {
	double          dot;
	struct vector  *x, *y;

	/* check and process arguments */
	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	y = luaL_checkudata(L, 2, LUALINEAR_VECTOR_METATABLE);
	luaL_argcheck(L, y->size == x->size, 2, "dimension mismatch");

	/* invoke subprogram */
	dot = cblas_ddot(x->size, x->values, x->inc, y->values, y->inc);
	lua_pushnumber(L, dot);
	return 1;
}

static int v (lua_State *L, v_function f) {
	size_t          i;
	struct vector  *x, *y;
	struct matrix  *X;

	/* check and process arguments */
	x = luaL_testudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	if (x != NULL) {
		lua_pushnumber(L, f(x->size, x->values, x->inc));
		return 1;
	}
	X = luaL_testudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	if (X != NULL) {
		y = luaL_checkudata(L, 2, LUALINEAR_VECTOR_METATABLE);
		if (checktranspose(L, 3) == CblasNoTrans) {
			if (X->order == CblasRowMajor) {
				luaL_argcheck(L, y->size == X->rows, 2, "dimension mismatch");
				for (i = 0; i < X->rows; i++) {
					y->values[i * y->inc] = f(X->cols, &X->values[i * X->ld],
							1);
				}
			} else {
				luaL_argcheck(L, y->size == X->cols, 2, "dimension mismatch");
				for (i = 0; i < X->cols; i++) {
					y->values[i * y->inc] = f(X->rows, &X->values[i * X->ld],
							1);
				}
			}
		} else {
			if (X->order == CblasRowMajor) {
				luaL_argcheck(L, y->size == X->cols, 2, "dimension mismatch");
				for (i = 0; i < X->cols; i++) {
					y->values[i * y->inc] = f(X->rows, &X->values[i], X->ld);
				}
			} else {
				luaL_argcheck(L, y->size == X->rows, 2, "dimension mismatch");
				for (i = 0; i < X->rows; i++) {
					y->values[i * y->inc] = f(X->cols, &X->values[i], X->ld);
				}
			}
		}
		return 0;
	}
	return argerror(L, 1);
}

static int nrm2 (lua_State *L) {
	return v(L, cblas_dnrm2);
}

static int asum (lua_State *L) {
	return v(L, cblas_dasum);
}

/* cblas_dsum does not work as expected */
static double _sum (int size, const double *values, int inc) {
	size_t  i;
	double  sum;

	sum = 0.0;
	#pragma omp parallel for private(i) schedule(auto) if(size >= LUALINEAR_OMP_MINSIZE) \
			reduction(+:sum)
	for (i = 0; i < (size_t)size; i++) {
		sum += values[i * inc];
	}
	return sum;
}

static int sum (lua_State *L) {
	return v(L, _sum);
}

static int iamax (lua_State *L) {
	struct vector  *x;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	lua_pushinteger(L, cblas_idamax(x->size, x->values, x->inc) + 1);
	return 1;
}

static int iamin (lua_State *L) {
	struct vector  *x;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	lua_pushinteger(L, cblas_idamin(x->size, x->values, x->inc) + 1);
	return 1;
}

static int imax (lua_State *L) {
	struct vector  *x;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	lua_pushinteger(L, cblas_idmax(x->size, x->values, x->inc) + 1);
	return 1;
}

static int imin (lua_State *L) {
	struct vector  *x;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	lua_pushinteger(L, cblas_idmin(x->size, x->values, x->inc) + 1);
	return 1;
}


/*
 * vector-matrix functions
 */

static int xy (lua_State *L, vm_function s, int hasy, int hasalpha) {
	int             index;
	size_t          i;
	double          alpha;
	struct vector  *x, *y;
	struct matrix  *X, *Y;

	/* check and process arguments */
	index = 2;
	x = luaL_testudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	if (x != NULL) {
		if (hasy) {
			y = luaL_testudata(L, 2, LUALINEAR_VECTOR_METATABLE);
			Y = luaL_testudata(L, 2, LUALINEAR_MATRIX_METATABLE);
			if (y == NULL && Y == NULL) {
				return argerror(L, 2);
			}
			index++;
		} else {
			y = x;
			Y = NULL;
		}
		if (hasalpha) {
			alpha = luaL_optnumber(L, index, 1.0);
			index++;
		} else {
			alpha = 0.0;
		}
		if (y != NULL) {
			/* invoke subprogram on vector-vector */
			luaL_argcheck(L, y->size == x->size, 2, "dimension mismatch");
			s(x->size, x->values, x->inc, y->values, y->inc, alpha);
			return 0;
		}

		/* invoke subprogram on vector-matrix */
		if (checktranspose(L, index) == CblasNoTrans) {
			if (Y->order == CblasRowMajor) {
				luaL_argcheck(L, 1, x->size == Y->cols, "dimension mismatch");
				for (i = 0; i < Y->rows; i++) {
					s(x->size, x->values, x->inc, &Y->values[i * Y->ld], 1,
							alpha);
				}
			} else {
				luaL_argcheck(L, 1, x->size == Y->rows, "dimension mismatch");
				for (i = 0; i < Y->cols; i++) {
					s(x->size, x->values, x->inc, &Y->values[i * Y->ld], 1,
							alpha);
				}
			}
		} else {
			if (Y->order == CblasRowMajor) {
				luaL_argcheck(L, 1, x->size == Y->rows, "dimension mismatch");
				for (i = 0; i < Y->rows; i++) {
					s(x->size, x->values, x->inc, &Y->values[i], Y->ld, alpha);
				}
			} else {
				luaL_argcheck(L, 1, x->size == Y->cols, "dimension mismatch");
				for (i = 0; i < Y->cols; i++) {
					s(x->size, x->values, x->inc, &Y->values[i], Y->ld, alpha);
				}
			}
		}
		return 0;
	}
	X = luaL_testudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	if (X != NULL) {
		if (hasy) {
			Y = luaL_checkudata(L, 2, LUALINEAR_MATRIX_METATABLE);
			luaL_argcheck(L, X->order == Y->order, 2, "order mismatch");
			luaL_argcheck(L, X->rows == Y->rows && X->cols == Y->cols, 2,
					"dimension mismatch");
			index++;
		} else {
			Y = X;
		}
		if (hasalpha) {
			alpha = luaL_optnumber(L, index, 1.0);
			index++;
		} else {
			alpha = 0.0;
		}

		/* invoke subprogram on matrix-matrix */
		if (X->order == CblasRowMajor) {
			for (i = 0; i < X->rows; i++) {
				s(X->cols, &X->values[i * X->ld], 1, &Y->values[i * Y->ld], 1,
						alpha);
			}
		} else {
			for (i = 0; i < X->cols; i++) {
				s(X->rows, &X->values[i * X->ld], 1, &Y->values[i * Y->ld], 1,
						alpha);
			}
		}
		return 0;
	}
	return argerror(L, 1);
}

static void _swap (size_t size, double *x, int incx, double *y, int incy, double alpha) {
	(void)alpha;
	cblas_dswap(size, x, incx, y, incy);
}

static int swap (lua_State *L) {
	return xy(L, _swap, 1, 0);
}

static void _copy (size_t size, double *x, int incx, double *y, int incy, double alpha) {
	(void)alpha;
	cblas_dcopy(size, x, incx, y, incy);
}

static int copy (lua_State *L) {
	return xy(L, _copy, 1, 0);
}

static void _axpy (size_t size, double *x, int incx, double *y, int incy, double alpha) {
	cblas_daxpy(size, alpha, x, incx, y, incy);
}

static int axpy (lua_State *L) {
	return xy(L, _axpy, 1, 1);
}

static void _scal (size_t size, double *x, int incx, double *y, int incy, double alpha) {
	(void)y;
	(void)incy;
	cblas_dscal(size, alpha, x, incx);
}

static int scal (lua_State *L) {
	return xy(L, _scal, 0, 1);
}

static void _set (size_t size, double *x, int incx, double *y, int incy, double alpha) {
	size_t  i;

	(void)y;
	(void)incy;
	#pragma omp parallel for private(i) schedule(auto) if(size >= LUALINEAR_OMP_MINSIZE)
	for (i = 0; i < size; i++) {
		x[i * incx] = alpha;
	}
}

static int set (lua_State *L) {
	return xy(L, _set, 0, 1);
}

static void _uniform (size_t size, double *x, int incx, double *y, int incy, double alpha) {
	size_t  i;

	(void)y;
	(void)incy;
	(void)alpha;
	for (i = 0; i < size; i++) {
		*x = random() / (RAND_MAX + 1.0);
		x += incx;
	}
}

static int uniform (lua_State *L) {
	return xy(L, _uniform, 0, 0);
}

static void _normal (size_t size, double *x, int incx, double *y, int incy, double alpha) {
	size_t  i;
	double  u1, u2, r, s, c;

	(void)y;
	(void)incy;
	(void)alpha;
	for (i = 0; i < size - 1; i += 2) {
		do {
			u1 = random() / (double)RAND_MAX;
			u2 = random() / (double)RAND_MAX;
		} while (u1 <= -DBL_MAX);
		r = sqrt(-2.0 * log(u1));
		sincos(2 * M_PI * u2, &s, &c);
		*x = r * c;
		x += incx;
		*x = r * s;
		x += incx;
	}
	if (i < size) {
		do {
			u1 = random() / (double)RAND_MAX;
			u2 = random() / (double)RAND_MAX;
		} while (u1 <= -DBL_MAX);
		*x = sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);
		x += incx;
	}
}

static int normal (lua_State *L) {
	return xy(L, _normal, 0, 0);
}

static void _inc (size_t size, double *x, int incx, double *y, int incy, double alpha) {
	size_t  i;

	(void)y;
	(void)incy;
	#pragma omp parallel for private(i) schedule(auto) if(size >= LUALINEAR_OMP_MINSIZE)
	for (i = 0; i < size; i++) {
		x[i * incx] += alpha;
	}
}

static int inc (lua_State *L) {
	return xy(L, _inc, 0, 1);
}

static void _mul1 (size_t size, double *x, int incx, double *y, int incy, double alpha) {
	size_t  i;

	(void)alpha;
	#pragma omp parallel for private(i) schedule(auto) if(size >= LUALINEAR_OMP_MINSIZE)
	for (i = 0; i < size; i++) {
		y[i * incy] *= x[i * incx];
	}
}

static void _mulm1 (size_t size, double *x, int incx, double *y, int incy, double alpha) {
	size_t  i;

	(void)alpha;
	#pragma omp parallel for private(i) schedule(auto) if(size >= LUALINEAR_OMP_MINSIZE)
	for (i = 0; i < size; i++) {
		y[i * incy] /= x[i * incx];
	}
}

static void _mul (size_t size, double *x, int incx, double *y, int incy, double alpha) {
	size_t  i;

	#pragma omp parallel for private(i) schedule(auto) if(size >= LUALINEAR_OMP_MINSIZE)
	for (i = 0; i < size; i++) {
		y[i * incy] *= pow(x[i * incx], alpha);
	}
}

static int mul (lua_State *L) {
	double  alpha;

	alpha = luaL_optnumber(L, 3, 1.0);
	if (alpha == 1.0) {
		return xy(L, _mul1, 1, 1);
	}
	if (alpha == -1.0) {
		return xy(L, _mulm1, 1, 1);
	}
	return xy(L, _mul, 1, 1);
}

static void _pow (size_t size, double *x, int incx, double *y, int incy, double alpha) {
	size_t  i;

	(void)y;
	(void)incy;
	#pragma omp parallel for private(i) schedule(auto) if(size >= LUALINEAR_OMP_MINSIZE)
	for (i = 0; i < size; i++) {
		x[i * incx] = pow(x[i * incx], alpha);
	}
}

static int powx (lua_State *L) {
	return xy(L, _pow, 0, 1);
}


/*
 * unary functions
 */

static int apply (lua_State *L, unary_function apply, int parallel) {
	size_t          base, i, j;
	struct vector  *x;
	struct matrix  *X;

	if (lua_type(L, 1) == LUA_TNUMBER) {
		lua_pushnumber(L, apply(lua_tonumber(L, 1)));
		return 1;
	}
	x = luaL_testudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	if (x != NULL) {
		#pragma omp parallel for private(i) schedule(auto) \
				if(parallel && x->size >= LUALINEAR_OMP_MINSIZE)
		for (i = 0; i < x->size; i++) {
			x->values[i * x->inc] = apply(x->values[i * x->inc]);
		}
		return 0;
	}
	X = luaL_testudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	if (X != NULL) {
		if (X->order == CblasRowMajor) {
			for (i = 0; i < X->rows; i++) {
				base = i * X->ld;
				#pragma omp parallel for private(j) schedule(auto) \
						if(parallel && X->cols >= LUALINEAR_OMP_MINSIZE)
				for (j = 0; j < X->cols; j++) {
					X->values[base + j] = apply(X->values[base + j]);
				}
			}
		} else {
			for (i = 0; i < X->cols; i++) {
				base = i * X->ld;
				#pragma omp parallel for private(j) schedule(auto) \
						if(parallel && X->rows >= LUALINEAR_OMP_MINSIZE)
				for (j = 0; j < X->rows; j++) {
					X->values[base + j] = apply(X->values[base + j]);
				}
			}
		}
		return 0;
	}
	return luaL_argerror(L, 1, lua_pushfstring(L, "number, vector, or matrix expected, got %s",
			luaL_typename(L, 1)));
}

static double _sign (double x) {
	if (x > 0) {
		return 1;
	}
	if (x < 0) {
		return -1;
	}
	return x;
}

static int sign (lua_State *L) {
	return apply(L, _sign, 1);
}

static int absx (lua_State *L) {
	return apply(L, fabs, 1);
}

static int expx (lua_State *L) {
	return apply(L, exp, 1);
}

static int logx (lua_State *L) {
	return apply(L, log, 1);
}

static double _logistic (double z) {
	return 1.0 / (1.0 + exp(-z));
}

static int logistic (lua_State *L) {
	return apply(L, _logistic, 1);
}

static int tanhx (lua_State *L) {
	return apply(L, tanh, 1);
}

static double _softplus (double x) {
	return log(1 + exp(x));
}

static int softplus (lua_State *L) {
	return apply(L, _softplus, 1);
}

static double _rectifier (double x) {
	return x > 0.0 ? x : 0.0;
}

static int rectifier (lua_State *L) {
	return apply(L, _rectifier, 1);
}

static __thread lua_State *TL;

static double _apply (double x) {
	double result;

	lua_pushvalue(TL, -1);
	lua_pushnumber(TL, x);
	lua_call(TL, 1, 1);
	result = lua_tonumber(TL, -1);
	lua_pop(TL, 1);
	return result;
}

static int applyx (lua_State *L) {
	luaL_checktype(L, 2, LUA_TFUNCTION);
	lua_settop(L, 2);
	TL = L;
	return apply(L, _apply, 0);
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
	luaL_argcheck(L, x->size == n, 2, "dimension mismatch");
	luaL_argcheck(L, y->size == m, 3, "dimension mismatch");

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
	luaL_argcheck(L, x->size == A->rows, 1, "dimension mismatch");
	luaL_argcheck(L, y->size == A->cols, 2, "dimension mismatch");

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
		#pragma omp parallel for private(i, j, sum, v) schedule(auto) \
				if(A->rows * A->cols >= LUALINEAR_OMP_MINSIZE)
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
		#pragma omp parallel for private(i, j, sum, v) schedule(auto) \
				if(A->rows * A->cols >= LUALINEAR_OMP_MINSIZE)
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
			#pragma omp parallel for private(j, k, sum, vi, vj) schedule(auto) \
					if(A->rows * (A->cols - i) >= LUALINEAR_OMP_MINSIZE)
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
			#pragma omp parallel for private(j, k, sum, vi, vj) schedule(auto) \
					if(A->rows * (A->cols - i) >= LUALINEAR_OMP_MINSIZE)
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
		#pragma omp parallel for private(i, j, sum, v) schedule(auto) \
				if(A->rows * A->cols >= LUALINEAR_OMP_MINSIZE)
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
		#pragma omp parallel for private(i, j, sum, v) schedule(auto) \
				if(A->rows * A->cols >= LUALINEAR_OMP_MINSIZE)
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
			#pragma omp parallel for private(j, k, sum, vi, vj) schedule(auto) \
					if(A->rows * (A->cols - i) >= LUALINEAR_OMP_MINSIZE)
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
			#pragma omp parallel for private(j, k, sum, vi, vj) schedule(auto) \
					if(A->rows * (A->cols - i) >= LUALINEAR_OMP_MINSIZE)
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
		{ "dot", dot },
		{ "nrm2", nrm2 },
		{ "asum", asum },
		{ "sum", sum },
		{ "iamax", iamax },
		{ "iamin", iamin },
		{ "imax", imax },
		{ "imin", imin },
		{ "swap", swap },
		{ "copy", copy },
		{ "axpy", axpy },
		{ "scal", scal },
		{ "set", set },
		{ "uniform", uniform },
		{ "normal", normal },
		{ "inc", inc },
		{ "mul", mul },
		{ "pow", powx },
		{ "sign", sign },
		{ "abs", absx },
		{ "exp", expx },
		{ "log", logx },
		{ "logistic", logistic },
		{ "tanh", tanhx },
		{ "softplus", softplus },
		{ "rectifier", rectifier },
		{ "apply", applyx },
		{ "gemv", gemv },
		{ "ger", ger },
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
	lua_pushcfunction(L, matrix_free);
	lua_setfield(L, -2, "__gc");
	lua_pop(L, 1);

	return 1;
}
