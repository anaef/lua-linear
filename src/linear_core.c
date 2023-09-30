/*
 * Lua Linear core
 *
 * Copyright (C) 2017-2023 Andre Naef
 */


#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <lauxlib.h>
#include "linear_core.h"
#include "linear_elementary.h"
#include "linear_unary.h"
#include "linear_binary.h"
#include "linear_program.h"


/* compatibility */
#if LUA_VERSION_NUM < 502
#define lua_rawlen  lua_objlen
void *luaL_testudata(lua_State *L, int index, const char *name);
#endif

/* vector */
static void linear_push_vector(lua_State *L, size_t length, size_t inc, linear_data_t *data,
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
		linear_data_t *data, double *values);
static int linear_matrix_len(lua_State *L);
static int linear_matrix_index(lua_State *L);
#if LUA_VERSION_NUM < 504
static int linear_matrix_next(lua_State *L);
static int linear_matrix_ipairs(lua_State *L);
#endif
static int linear_matrix_tostring(lua_State *L);
static int linear_matrix_gc(lua_State *L);

/* random */
static void linear_seedrandomstate(uint64_t *s, uint64_t seed);
static uint64_t *linear_randomstate(lua_State *L);

/* core functions */
static int linear_vector(lua_State *L);
static int linear_matrix(lua_State *L);
static int linear_totable(lua_State *L);
static int linear_tolinear(lua_State *L);
static int linear_tovector(lua_State *L);
static int linear_type(lua_State *L);
static int linear_size(lua_State *L);
static int linear_tvector(lua_State *L);
static int linear_sub(lua_State *L);
static int linear_unwind(lua_State *L);
static int linear_reshape(lua_State *L);
static int linear_randomseed(lua_State *L);
#if LUA_VERSION_NUM < 502
static int linear_ipairs(lua_State *L);
#endif


static const char *const LINEAR_ORDERS[] = {"row", "col", NULL};


/*
 * arguments
 */

CBLAS_ORDER linear_checkorder (lua_State *L, int index) {
	return luaL_checkoption(L, index, "row", LINEAR_ORDERS) == 0 ? CblasRowMajor
			: CblasColMajor;
}

int linear_checkargs (lua_State *L, int index, size_t size, linear_param_t *params,
		linear_arg_u *args) {
	while (params->type) {
		switch (params->type) {
		case 'n':
			args->n = luaL_optnumber(L, index, params->def.n);
			index++;
			break;

		case 'i':
			args->i = luaL_optinteger(L, index, params->def.i);
			index++;
			break;

		case 'e':
			args->e = luaL_checkoption(L, index, params->def.e[0], params->def.e);
			index++;
			break;

		case 'd':
			args->d = luaL_optinteger(L, index, params->def.d);
			luaL_argcheck(L, args->d < size, index, "bad ddof");
			index++;
			break;

		case 'L':
			args->L = L;
			break;

		case 'r':
			args->r = linear_randomstate(L);
			break;

		default:
			return luaL_error(L, "bad param type");
		}
		params++;
		args++;
	}
	return 0;
}

int linear_argerror (lua_State *L, int index, int numok) {
	const char  *fmt;

	fmt = !numok ? "vector, or matrix expected, got %s"
			: "number, vector, or matrix expected, got %s";
	return luaL_argerror(L, index, lua_pushfstring(L, fmt, luaL_typename(L, index)));
}


/*
 * compatibility
 */

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

linear_vector_t *linear_create_vector (lua_State *L, size_t length) {
	linear_vector_t  *vector;

	assert(length >= 1 && length <= INT_MAX);
	vector = lua_newuserdata(L, sizeof(linear_vector_t));
	vector->length = length;
	vector->inc = 1;
	vector->data = NULL;
	luaL_getmetatable(L, LINEAR_VECTOR);
	lua_setmetatable(L, -2);
	vector->data = calloc(1, sizeof(linear_data_t) + length * sizeof(double));
	if (vector->data == NULL) {
		luaL_error(L, "cannot allocate data");
	}
	vector->data->refs = 1;
	vector->values = (double *)((char *)vector->data + sizeof(linear_data_t));
	return vector;
}

static void linear_push_vector (lua_State *L, size_t length, size_t inc, linear_data_t *data,
		double *values) {
	linear_vector_t  *vector;

	assert(length >= 1 && length <= INT_MAX);
	vector = lua_newuserdata(L, sizeof(linear_vector_t));
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
	linear_vector_t  *x;

	x = luaL_checkudata(L, 1, LINEAR_VECTOR);
	lua_pushinteger(L, x->length);
	return 1;
}

static int linear_vector_index (lua_State *L) {
	size_t             index;
	linear_vector_t  *x;

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
	size_t            index;
	double            value;
	linear_vector_t  *x;

	x = luaL_checkudata(L, 1, LINEAR_VECTOR);
	index = luaL_checkinteger(L, 2);
	luaL_argcheck(L, index >= 1 && index <= x->length, 2, "bad index");
	value = luaL_checknumber(L, 3);
	x->values[(index - 1) * x->inc] = value;
	return 0;
}

#if LUA_VERSION_NUM < 504
static int linear_vector_next (lua_State *L) {
	size_t            index;
	linear_vector_t  *x;

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
	linear_vector_t  *x;

	x = luaL_checkudata(L, 1, LINEAR_VECTOR);
	lua_pushfstring(L, LINEAR_VECTOR ": %p", x);
	return 1;
}

static int linear_vector_gc (lua_State *L) {
	linear_vector_t  *x;

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

linear_matrix_t *linear_create_matrix (lua_State *L, size_t rows, size_t cols,
		CBLAS_ORDER order) {
	linear_matrix_t  *matrix;

	assert(rows >= 1 && rows <= INT_MAX && cols >= 1 && cols <= INT_MAX);
	matrix = lua_newuserdata(L, sizeof(linear_matrix_t));
	matrix->rows = rows;
	matrix->cols = cols;
	matrix->ld = order == CblasRowMajor ? cols : rows;
	matrix->order = order;
	matrix->data = NULL;
	luaL_getmetatable(L, LINEAR_MATRIX);
	lua_setmetatable(L, -2);
	matrix->data = calloc(1, sizeof(linear_data_t) + rows * cols * sizeof(double));
	if (matrix->data == NULL) {
		luaL_error(L, "cannot allocate data");
	}
	matrix->data->refs = 1;
	matrix->values = (double *)((char *)matrix->data + sizeof(linear_data_t));
	return matrix;
}

static void linear_push_matrix (lua_State *L, size_t rows, size_t cols, size_t ld,
		CBLAS_ORDER order, linear_data_t *data, double *values) {
	linear_matrix_t  *matrix;

	assert(rows >= 1 && rows <= INT_MAX && cols >= 1 && cols <= INT_MAX);
	matrix = (linear_matrix_t *)lua_newuserdata(L, sizeof(linear_matrix_t));
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
	linear_matrix_t  *X;

	X = luaL_checkudata(L, 1, LINEAR_MATRIX);
	if (X->order == CblasRowMajor) {
		lua_pushinteger(L, X->rows);
	} else {
		lua_pushinteger(L, X->cols);
	}
	return 1;
}

static int linear_matrix_index (lua_State *L) {
	size_t            index;
	linear_matrix_t  *X;

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
	size_t            index, majorsize, minorsize;
	linear_matrix_t  *X;

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
	linear_matrix_t  *X;

	X = luaL_checkudata(L, 1, LINEAR_MATRIX);
	lua_pushfstring(L, LINEAR_MATRIX ": %p", X);
	return 1;
}

static int linear_matrix_gc (lua_State *L) {
	linear_matrix_t  *X;

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
 * random
 */

static void linear_seedrandomstate (uint64_t *r, uint64_t seed) {
	int       i;
	uint64_t  z;

	/* source: SplitMix64; https://prng.di.unimi.it/ */
	for (i = 0; i < 4; i++) {
		z = (seed += 0x9e3779b97f4a7c15);
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
		r[i] = z ^ (z >> 31);
	}
}

static uint64_t *linear_randomstate (lua_State *L) {
	uint64_t  *r;

	lua_getfield(L, LUA_REGISTRYINDEX, LINEAR_RANDOM);
	r = (void *)lua_topointer(L, -1);
	lua_pop(L, 1);
	return r;
}

double linear_random (uint64_t *r) {
	uint64_t  result, t;

	/* source: xoshiro256+; https://prng.di.unimi.it/ */
	result = r[0] + r[3];
	t = r[1] << 17;
	r[2] ^= r[0];
	r[3] ^= r[1];
	r[1] ^= r[2];
	r[0] ^= r[3];
	r[2] ^= t;
	r[3] = (r[3] << 45) | (r[3] >> (64 - 45));
	return (result >> (64 - DBL_MANT_DIG)) * (1.0 / ((uint64_t)1 << DBL_MANT_DIG));  /* [0,1) */
}


/*
 * comparison
 */

int linear_comparison_handler (const void *a, const void *b) {
	double  da, db;

	da = *(const double *)a;
	db = *(const double *)b;
	return da < db ? -1 : (da > db ? 1 : 0);
}


/*
 * core functions
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
	size_t            i, j;
	const double     *value;
	linear_vector_t  *x;
	linear_matrix_t  *X;

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
	double           *value;
	size_t            size, rows, cols, major, minor, i, j;
	CBLAS_ORDER       order;
	linear_vector_t  *x;
	linear_matrix_t  *X;

	luaL_checktype(L, 1, LUA_TTABLE);
	switch (linear_rawgeti(L, 1, 1)) {
	case LUA_TNUMBER:
		size = lua_rawlen(L, 1);
		if (size < 1 || size > INT_MAX) {
			return luaL_error(L, "bad dimension");
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
			return luaL_error(L, "bad dimension");
		}
		minor = lua_rawlen(L, -1);
		if (minor < 1 || minor > INT_MAX) {
			return luaL_error(L, "bad dimension");
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

static int linear_tovector (lua_State *L) {
	double           *value;
	size_t            size, i;
	const char       *key;
	linear_data_t    *data;
	linear_vector_t  *x;

	luaL_checktype(L, 1, LUA_TTABLE);
	size = lua_rawlen(L, 1);
	luaL_argcheck(L, size >= 1 && size <= INT_MAX, 1, "bad dimension");
	x = linear_create_vector(L, size);
	value = x->values;
	switch (lua_type(L, 2)) {
	case LUA_TSTRING:
		key = lua_tostring(L, 2);
		for (i = 0; i < size; i++) {
			if (linear_rawgeti(L, 1, i + 1) != LUA_TTABLE) {
				return luaL_error(L, "bad value at index %d", i + 1);
			}
			switch (linear_getfield(L, -1, key)) {
			case LUA_TNUMBER:
				*value++ = lua_tonumber(L, -1);
				break;

			case LUA_TNIL:
				break;

			default:
				return luaL_error(L, "bad value at index %d", i + 1);
			}
			lua_pop(L, 2);
		}
		break;

	case LUA_TFUNCTION:
		for (i = 0; i < size; i++) {
			lua_pushvalue(L, 2);
			lua_rawgeti(L, 1, i + 1);
			lua_call(L, 1, 1);
			switch (lua_type(L, -1)) {
			case LUA_TNUMBER:
				*value++ = lua_tonumber(L, -1);
				break;

			case LUA_TNIL:
				break;

			default:
				return luaL_error(L, "bad value at index %d", i + 1);
			}
			lua_pop(L, 1);
		}
		break;

	default:
		return luaL_argerror(L, 2, "bad selector");
	}
	x->length = value - x->values;
	if (x->length == 0) {
		return luaL_error(L, "bad dimension");
	}
	if (x->length < size / 2) {
		data = realloc(x->data, sizeof(linear_data_t) + x->length * sizeof(double));
		if (data) {
			x->data = data;
			x->values = (double *)((char *)x->data + sizeof(linear_data_t));
		}
	}
	return 1;
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
	linear_vector_t  *x;
	linear_matrix_t  *X;

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
	size_t            index, length;
	linear_matrix_t  *X;

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
	linear_vector_t  *x;
	linear_matrix_t  *X;

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
	int               index;
	double           *s, *d, *last;
	size_t            i, j;
	linear_vector_t  *x;
	linear_matrix_t  *X;

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
	int               index;
	double           *d, *s, *last;
	size_t            i, j;
	linear_vector_t  *x;
	linear_matrix_t  *X;

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

static int linear_randomseed (lua_State *L) {
	uint64_t  seed;

	seed = luaL_checkinteger(L, 1);
	linear_seedrandomstate(linear_randomstate(L), seed);
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
	return linear_argerror(L, 1, 0);
}
#endif


/*
 * library
 */

int luaopen_linear (lua_State *L) {
	static const luaL_Reg FUNCTIONS[] = {
		{"vector", linear_vector},
		{"matrix", linear_matrix},
		{"totable", linear_totable},
		{"tolinear", linear_tolinear},
		{"tovector", linear_tovector},
		{"type", linear_type},
		{"size", linear_size},
		{"tvector", linear_tvector},
		{"sub", linear_sub},
		{"unwind", linear_unwind},
		{"reshape", linear_reshape},
		{"randomseed", linear_randomseed},
#if LUA_VERSION_NUM < 502
		{ "ipairs", linear_ipairs },
#endif
		{ NULL, NULL }
	};
	uint64_t  *r;

	/* register functions */
#if LUA_VERSION_NUM >= 502
	luaL_newlib(L, FUNCTIONS);
#else
	luaL_register(L, luaL_checkstring(L, 1), FUNCTIONS);
#endif
	linear_open_elementary(L);
	linear_open_unary(L);
	linear_open_binary(L);
	linear_open_program(L);

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

	/* random state */
	r = lua_newuserdata(L, 4 * sizeof(uint64_t));
	linear_seedrandomstate(r, (uint64_t)time(NULL) ^ (uintptr_t)L);
	lua_setfield(L, LUA_REGISTRYINDEX, LINEAR_RANDOM);

	return 1;
}
