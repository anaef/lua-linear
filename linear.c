#include "linear.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <lauxlib.h>
#include <cblas.h>
#include <lapacke.h>

/* matrix orders */
static const char * const ORDERS[] = { "row", "col", NULL };

/* checks an order */
static CBLAS_ORDER checkorder (lua_State *L, int index) {

	switch (luaL_checkoption(L, index, "row", ORDERS)) {
	case 0:
		return CblasRowMajor;

	case 1:
		return CblasColMajor;
	}

	/* not reached */
	assert(0);
	return (CBLAS_ORDER)0;
}

/* checks a transpose */
static CBLAS_TRANSPOSE checktranspose (lua_State *L, int index) {
	static const char * const TRANSPOSES[] = { "notrans", "trans", NULL };

	switch (luaL_checkoption(L, index, "notrans", TRANSPOSES)) {
	case 0:
		return CblasNoTrans;

	case 1:
		return CblasTrans;
	}

	/* not reached */
	assert(0);
	return (CBLAS_TRANSPOSE)0;
}

/* translates a transpose for LAPACK */
static char lapacktranspose (CBLAS_TRANSPOSE transpose) {
	switch (transpose) {
	case CblasNoTrans:
		return 'N';

	case CblasTrans:
		return 'T';

	default:
		/* not reached */
		assert(0);
		return '\0';
	}
}

/* returns an int value from a table */
static int intvalue (lua_State *L, const char *key, int dfl) {
	int result, isinteger;

	lua_getfield(L, -1, key);
	if (!lua_isnil(L, -1)) {
		result = lua_tointegerx(L, -1, &isinteger);
		if (!isinteger) {
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

/* returns an option value from a table */
static int optionvalue (lua_State *L, const char *key, const char *dfl,
		const char *options[]) {
	const char *str;
	int i;
	
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
	luaL_error(L, "bad option " LUA_QS " in field " LUA_QS, str, key);
	return 0; /* not reached */
}

/* raises a linear argument error */
static int argerror (lua_State *L, int index) {
	return luaL_argerror(L, index, lua_pushfstring(L, "vector, or matrix "
			"expected, got %s", luaL_typename(L, index)));
}

/* pushes a new vector onto the stack */
static struct vector *newvector (lua_State *L, int size) {
	return lualinear_newvector(L, size);
}

/* pushes an existing vector onto the stack */
static struct vector *wrapvector (lua_State *L, int size, float *values) {
	return lualinear_wrapvector(L, size, values);
}

/* creates a new vector */
static int vector (lua_State *L) {
	int size;

	/* process arguments */
	size = luaL_checkinteger(L, 1);
	luaL_argcheck(L, size >= 1, 1, "bad dimension");

	/* create */
	newvector(L, size);
	return 1;
}

/* vector length implementation */
static int vector_len (lua_State *L) {
	struct vector *x;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	lua_pushinteger(L, x->size);
	return 1;
}

/* vector index implementation */
static int vector_index (lua_State *L) {
	struct vector *x;
	int index;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	index = luaL_checkinteger(L, 2);
	luaL_argcheck(L, index >= 1 && index <= x->size, 2, "bad index");
	lua_pushnumber(L, x->values[(size_t)(index - 1) * x->inc]);
	return 1;
}

/* matrix vector newindex implementation */
static int vector_newindex (lua_State *L) {
	struct vector *x;
	int index;
	float value;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	index = luaL_checkinteger(L, 2);
	luaL_argcheck(L, index >= 1 && index <= x->size, 2, "bad index");
	value = luaL_checknumber(L, 3);
	x->values[(size_t)(index - 1) * x->inc] = value;
	return 0;
}

/* vector next function */
static int vector_next (lua_State *L) {
	struct vector *x;
	int index;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	index = luaL_checkinteger(L, 2);
	if (index >= 0 && index < x->size) {
		lua_pushinteger(L, index + 1);
		lua_pushnumber(L, x->values[(size_t)index]);
		return 2;
	}
	lua_pushnil(L);
	return 1;
}

/* vector ipairs function */
static int vector_ipairs (lua_State *L) {
	luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	lua_pushcfunction(L, vector_next);
	lua_pushvalue(L, 1);
	lua_pushinteger(L, 0);
	return 3;
}

/* returns the string representation of a vector */
static int vector_tostring (lua_State *L) {
	struct vector *x;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	lua_pushfstring(L, "vector: %p", x);
	return 1;
}

/* frees a vector */
static int vector_free (lua_State *L) {
	struct vector *x;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	if (x->ref == LUA_NOREF) {
		free(x->values);
	} else {
		luaL_unref(L, LUA_REGISTRYINDEX, x->ref);
	}
	return 0;
}

/* pushes a new matrix onto the stack */
static struct matrix *newmatrix (lua_State *L, int rows, int cols,
		CBLAS_ORDER order) {
	return lualinear_newmatrix(L, rows, cols, order);
}

/* pushes an existing matrix onto the stack */
static struct matrix *wrapmatrix (lua_State *L, int rows, int cols,
		CBLAS_ORDER order, float *values) {
	return lualinear_wrapmatrix(L, rows, cols, order, values);
}

/* creates a new matrix */
static int matrix (lua_State *L) {
	int rows, cols;
	CBLAS_ORDER order;

	/* process arguments */
	rows = luaL_checkinteger(L, 1);
	luaL_argcheck(L, rows >= 1, 1, "bad dimension");
	cols = luaL_checkinteger(L, 2);
	luaL_argcheck(L, cols >= 1, 2, "bad dimension");
	order = checkorder(L, 3);

	/* create */
	newmatrix(L, rows, cols, order);
	return 1;
}

/* returns the length of a matrix */
static int matrix_len (lua_State *L) {
	struct matrix *X;

	X = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	switch (X->order) {
	case CblasRowMajor:
		lua_pushinteger(L, X->rows);
		break;

	case CblasColMajor:
		lua_pushinteger(L, X->cols);
		break;
	}
	return 1;
}

/* matrix index implementation */
static int matrix_index (lua_State *L) {
	struct matrix *X;
	int index, size;
	struct vector *x;

	/* process arguments */
	X = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	index = luaL_checkinteger(L, 2);
	luaL_argcheck(L, index >= 1, 2, "bad index");
	switch (X->order) {
	case CblasRowMajor:
		luaL_argcheck(L, index <= X->rows, 2, "bad index");
		size = X->cols;
		break;

	case CblasColMajor:
		luaL_argcheck(L, index <= X->cols, 2, "bad index");
		size = X->rows;
		break;

	default:
		/* not reached */
		size = -1; 
		assert(0);
	}

	/* create vector */
	x = wrapvector(L, size, &X->values[(size_t)(index - 1) * X->ld]);
	lua_pushvalue(L, 1);
	x->ref = luaL_ref(L, LUA_REGISTRYINDEX);
	return 1;
}

/* matrix next function */
static int matrix_next (lua_State *L) {
	struct matrix *X;
	int index, majorsize, minorsize;
	struct vector *x;

	X = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	index = luaL_checkinteger(L, 2);
	switch (X->order) {
	case CblasRowMajor:
		majorsize = X->rows;
		minorsize = X->cols;
		break;

	case CblasColMajor:
		majorsize = X->cols;
		minorsize = X->rows;
		break;

	default:
		/* not reached */
		assert(0);
		return 0;
	}
	if (index >= 0 && index < majorsize) {
		lua_pushinteger(L, index + 1);
		x = wrapvector(L, minorsize, &X->values[(size_t)index * X->ld]);
		lua_pushvalue(L, 1);
		x->ref = luaL_ref(L, LUA_REGISTRYINDEX);
		return 2;
	}
	lua_pushnil(L);
	return 1;
}

/* matrix ipairs function */
static int matrix_ipairs (lua_State *L) {
	luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	lua_pushcfunction(L, matrix_next);
	lua_pushvalue(L, 1);
	lua_pushinteger(L, 0);
	return 3;
}

/* returns the string representation of a matrix */
static int matrix_tostring (lua_State *L) {
	struct matrix *X;

	X = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	lua_pushfstring(L, "matrix: %p", X);
	return 1;
}

/* frees a matrix */
static int matrix_free (lua_State *L) {
	struct matrix *X;

	X = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	if (X->ref == LUA_NOREF) {
		free(X->values);
	} else {
		luaL_unref(L, LUA_REGISTRYINDEX, X->ref);
	}
	return 0;
}

/* returns the type of a linear object */
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

/* returns the size of a linear object */
static int size (lua_State *L) {
	struct vector *x;
	struct matrix *X;

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

/* transposed vector */
static int tvector (lua_State *L) {
	struct matrix *X;
	int index, size;
	struct vector *x;

	/* process arguments */
	X = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	index = luaL_checkinteger(L, 2);
	luaL_argcheck(L, index >= 1, 2, "bad index");
	switch (X->order) {
	case CblasRowMajor:
		luaL_argcheck(L, index <= X->cols, 2, "bad index");
		size = X->rows;
		break;

	case CblasColMajor:
		luaL_argcheck(L, index <= X->rows, 2, "bad index");
		size = X->cols;
		break;

	default:
		/* not reached */
		size = -1; 
		assert(0);
	}

	/* create vector */
	x = wrapvector(L, size, &X->values[index - 1]);
	x->inc = X->ld;
	lua_pushvalue(L, 1);
	x->ref = luaL_ref(L, LUA_REGISTRYINDEX);
	return 1;
}

/* subvector or submatrix */
static int sub (lua_State *L) {
	struct vector *x, *s;
	struct matrix *X, *S;

	/* process arguments */
	x = luaL_testudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	if (x != NULL) {
		int start, end;

		start = luaL_optint(L, 2, 1);
		luaL_argcheck(L, start >= 1 && start <= x->size, 2,
				"bad index");
		end = luaL_optint(L, 3, x->size);
		luaL_argcheck(L, end >= start && end <= x->size, 3,
				"bad index");
		s = wrapvector(L, end - start + 1, &x->values[
				(size_t)(start - 1) * x->inc]);
		s->inc = x->inc;
		lua_pushvalue(L, 1);
		s->ref = luaL_ref(L, LUA_REGISTRYINDEX);
		return 1;
	}
	X = luaL_testudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	if (X != NULL) {
		int rowstart, rowend, colstart, colend;

		switch (X->order){
		case CblasRowMajor:
			rowstart = luaL_optint(L, 2, 1);
			luaL_argcheck(L, rowstart >= 1 && rowstart <= X->rows,
					2, "bad index");
			colstart = luaL_optint(L, 3, 1);
			luaL_argcheck(L, colstart >= 1 && colstart <= X->cols,
					3, "bad index");
			rowend = luaL_optint(L, 4, X->rows);
			luaL_argcheck(L, rowend >= rowstart && rowend
					<= X->rows, 4, "bad index");
			colend = luaL_optint(L, 5, X->cols);
			luaL_argcheck(L, colend >= colstart && colend
					<= X->cols, 5, "bad index");
			S = wrapmatrix(L, rowend - rowstart + 1, colend
					- colstart + 1, X->order, &X->values[
					(size_t)(rowstart - 1) * X->ld
					+ colstart - 1]);
			break;
		
		case CblasColMajor:
			colstart = luaL_optint(L, 2, 1);
			luaL_argcheck(L, colstart >= 1 && colstart <= X->cols,
					2, "bad index");
			rowstart = luaL_optint(L, 3, 1);
			luaL_argcheck(L, rowstart >= 1 && rowstart <= X->rows,
					3, "bad index");
			colend = luaL_optint(L, 4, X->cols);
			luaL_argcheck(L, colend >= colstart && colend
					<= X->cols, 4, "bad index");
			rowend = luaL_optint(L, 5, X->rows);
			luaL_argcheck(L, rowend >= rowstart && rowend
					<= X->rows, 5, "bad index");
			S = wrapmatrix(L, rowend - rowstart + 1, colend
					- colstart + 1, X->order, &X->values[
					(size_t)(colstart - 1) * X->ld
					+ rowstart - 1]);
			break;

		default:
			/* not reached */
			assert(0);
			return 0;
		}
		S->ld = X->ld;
		lua_pushvalue(L, 1);
		S->ref = luaL_ref(L, LUA_REGISTRYINDEX);
		return 1;
	}
	return argerror(L, 1);
}

/* unwinds matrices into a vector */
static int unwind (lua_State *L) {
	struct vector *x;
	int index, i, j, k;
	size_t base;
	struct matrix *X;

	if (lua_gettop(L) == 0) {
		return luaL_error(L, "wrong number of arguments");
	}
	x = luaL_checkudata(L, lua_gettop(L), LUALINEAR_VECTOR_METATABLE);
	index = 1;
	i = 0;
	while (i < x->size) {
		X = luaL_checkudata(L, index, LUALINEAR_MATRIX_METATABLE);
		luaL_argcheck(L, X->rows * X->cols <= x->size - i, index,
				"matrix too large");
		switch (X->order) {
		case CblasRowMajor:
			for (j = 0; j < X->rows; j++) {
				base = (size_t)j * X->ld;
				for (k = 0; k < X->cols; k++) {
					x->values[(size_t)i * x->inc]
							= X->values[base + k];
					i++;
				}
			}
			break;

		case CblasColMajor:
			for (j = 0; j < X->cols; j++) {
				base = (size_t)j * X->ld;
				for (k = 0; k < X->rows; k++) {
					x->values[(size_t)i * x->inc]
							= X->values[base + k];
					i++;
				}
			}
			break;
		}
		index++;
	}
	return 0;
}

/* reshapes a vector into matrices */
static int reshape (lua_State *L) {
	struct vector *x;
	int index, i, j, k;
	size_t base;
	struct matrix *X;

	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	index = 2;
	i = 0;
	while (i < x->size) {
		X = luaL_checkudata(L, index, LUALINEAR_MATRIX_METATABLE);
		luaL_argcheck(L, X->rows * X->cols <= x->size - i, index,
				"matrix too large");
		switch (X->order) {
		case CblasRowMajor:
			for (j = 0; j < X->rows; j++) {
				base = (size_t)j * X->ld;
				for (k = 0; k < X->cols; k++) {
					X->values[base + k] = x->values[
							(size_t)i * x->inc];
					i++;
				}
			}
			break;

		case CblasColMajor:
			for (j = 0; j < X->cols; j++) {
				base = (size_t)j * X->ld;
				for (k = 0; k < X->rows; k++) {
					X->values[base + k] = x->values[
							(size_t)i * x->inc];
					i++;
				}
			}
			break;
		}
		index++;
	}
	return 0;
}

/* converts a vector or matrix to a table */
static int totable (lua_State *L) {
	struct vector *x;
	struct matrix *X;
	int i, j;
	const float *value;

	/* check and process arguments */
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
		switch (X->order) {
		case CblasRowMajor:
			lua_pushliteral(L, "rowmajor");
			lua_setfield(L, -2, "order");
			lua_createtable(L, X->rows, 0);
			for (i = 0; i < X->rows; i++) {
				lua_createtable(L, X->cols, 0);
				value = &X->values[(size_t)i * X->ld];
				for (j = 0; j < X->cols; j++) {
					lua_pushnumber(L, *value++);
					lua_rawseti(L, -2, j + 1);
				}
				lua_rawseti(L, -2, i + 1);
			}
			lua_setfield(L, -2, "values");
			break;

		case CblasColMajor:
			lua_pushliteral(L, "colmajor");
			lua_setfield(L, -2, "order");
			lua_createtable(L, X->cols, 0);
			for (i = 0; i < X->cols; i++) {
				lua_createtable(L, X->rows, 0);
				value = &X->values[(size_t)i * X->ld];
				for (j = 0; j < X->rows; j++) {
					lua_pushnumber(L, *value++);
					lua_rawseti(L, -2, j + 1);
				}
				lua_rawseti(L, -2, i + 1);
			}
			lua_setfield(L, -2, "values");
			break;
		}
		return 1;
	}
	return argerror(L, 1);
}

/* converts a table to a vector or matrix */
static int tolinear (lua_State *L) {
	static const char *types[] = { "vector", "matrix", NULL };
	static const char *orders[] = { "rowmajor", "colmajor", NULL };
	struct vector *x;
	struct matrix *X;
	int size, rows, cols, major, minor;
	CBLAS_ORDER order;
	int i, j;
	int isnum;
	float *value;

	/* check arguments */
	luaL_checktype(L, 1, LUA_TTABLE);
	lua_settop(L, 1);

	/* handle types */
	switch (optionvalue(L, "type", NULL, types)) {
	case 0: /* vector */
		size = intvalue(L, "length", -1);
		if (size < 1) {
			return luaL_error(L, "bad field " LUA_QS, "length");
		}
		x = newvector(L, size);
		lua_getfield(L, 1, "values");
		if (lua_type(L, -1) != LUA_TTABLE) {
			return luaL_error(L, "bad field " LUA_QS, "values");
		}
		value = x->values;
		for (i = 0; i < size; i++) {
			lua_rawgeti(L, -1, i + 1);
			*value++ = lua_tonumberx(L, -1, &isnum);
			if (!isnum) {
				return luaL_error(L, "bad value at index %d",
						i + 1);
			}
			lua_pop(L, 1);
		}
		lua_pop(L, 1);
		return 1;

	case 1: /* matrix */
		rows = intvalue(L, "rows", -1);
		if (rows < 1) {
			return luaL_error(L, "bad field " LUA_QS, "rows");
		}
		cols = intvalue(L, "cols", -1);
		if (cols < 1) {
			return luaL_error(L, "bad field " LUA_QS, "cols");
		}
		switch (optionvalue(L, "order", NULL, orders)) {
		case 0:
			order = CblasRowMajor;
			major = rows;
			minor = cols;
			break;

		case 1:
			order = CblasColMajor;
			major = cols;
			minor = rows;
			break;

		default:
			/* not reched */
			assert(0);
			return 0;
		}
		X = newmatrix(L, rows, cols, order);
		lua_getfield(L, 1, "values");
		if (lua_type(L, -1) != LUA_TTABLE) {
			return luaL_error(L, "bad field " LUA_QS, "values");
		}
		for (i = 0; i < major; i++) {
			value = &X->values[i * X->ld];
			lua_rawgeti(L, -1, i + 1);
			if (lua_type(L, -1) != LUA_TTABLE) {
				return luaL_error(L, "bad value at index %d",
						i + 1);
			}
			for (j = 0; j < minor; j++) {
				lua_rawgeti(L, -1, j + 1);
				*value++ = lua_tonumberx(L, -1, &isnum);
				if (!isnum) {
					return luaL_error(L, "bad value at "
							"index (%d,%d)", i + 1,
							j + 1);
				}
				lua_pop(L, 1);
			}
			lua_pop(L, 1);
		}
		lua_pop(L, 1);
		return 1;
	}

	/* not reached */
	assert(0);
	return 0;
}

/* invokes the DOT subprogram (x' y) */
static int dot (lua_State *L) {
	struct vector *x, *y;
	float dot;

	/* check and process arguments */
	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	y = luaL_checkudata(L, 2, LUALINEAR_VECTOR_METATABLE);
	luaL_argcheck(L, y->size == x->size, 2, "dimension mismatch");

	/* invoke subprogram */
	dot = cblas_sdot(x->size, x->values, x->inc, y->values, y->inc);
	lua_pushnumber(L, dot);
	return 1;
}

/* invokes the NRM2 subprogram (||x||_2) */
static int nrm2 (lua_State *L) {
	struct vector *x;
	float nrm2;

	/* check and process arguments */
	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);

	/* invoke subprogram */
	nrm2 = cblas_snrm2(x->size, x->values, x->inc);
	lua_pushnumber(L, nrm2);
	return 1;
}

/* invokes the ASUM subprogram (sigma |x|) */
static int asum (lua_State *L) {
	struct vector *x;
	float asum;

	/* check and process arguments */
	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);

	/* invoke subprogram */
	asum = cblas_sasum(x->size, x->values, x->inc);
	lua_pushnumber(L, asum);
	return 1;
}

/* invokes the IAMAX subprogram (argmax |x|) */
static int iamax (lua_State *L) {
	struct vector *x;
	int iamax;

	/* check and process arguments */
	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);

	/* invoke subprogram */
	iamax = cblas_isamax(x->size, x->values, x->inc);
	lua_pushinteger(L, iamax + 1);
	return 1;
}

/* sum implementation */
static float _sum (const float *values, int size, int inc) {
	float sum;
	int i;

	sum = 0.0;
	#pragma omp parallel for private(i) schedule(auto) \
			 if(size >= LUALINEAR_OMP_MINSIZE) reduction(+:sum)
	for (i = 0; i < size; i++) {
		sum += values[(size_t)i * inc];
	}
	return sum;
}

/* sum implementation (sigma x_i) */
static int sum (lua_State *L) {
	struct vector *x, *y;
	struct matrix *X;
	int i;

	/* check and process arguments */
	x = luaL_testudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	if (x != NULL) {
		lua_pushnumber(L, _sum(x->values, x->size, x->inc));
		return 1;
	}
	X = luaL_testudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	if (X != NULL) {
		y = luaL_checkudata(L, 2, LUALINEAR_VECTOR_METATABLE);
		switch (checktranspose(L, 3)) {
		case CblasNoTrans:
			switch (X->order) {
			case CblasRowMajor:
				luaL_argcheck(L, y->size == X->rows, 2,
						"dimension mismatch");
				for (i = 0; i < X->rows; i++) {
					y->values[(size_t)i * y->inc] = _sum(
							&X->values[(size_t)i
							* X->ld], X->cols, 1);
				}
				break;

			case CblasColMajor:
				luaL_argcheck(L, y->size == X->cols, 2,
						"dimension mismatch");
				for (i = 0; i < X->cols; i++) {
					y->values[(size_t)i * y->inc] = _sum(
							&X->values[(size_t)i
							* X->ld], X->rows, 1);
				}
				break;
			}
			break;

		case CblasTrans:
			switch (X->order) {
			case CblasRowMajor:
				luaL_argcheck(L, y->size == X->cols, 2,
						"dimension mismatch");
				for (i = 0; i < X->cols; i++) {
					y->values[(size_t)i * y->inc] = _sum(
							&X->values[(size_t)i],
							X->rows, X->ld);
				}
				break;

			case CblasColMajor:
				luaL_argcheck(L, y->size == X->rows, 2,
						"dimension mismatch");
				for (i = 0; i < X->rows; i++) {
					y->values[(size_t)i * y->inc] = _sum(
							&X->values[(size_t)i],
							X->cols, X->ld);
				}
				break;
			}
			break;

		default:
			/* not reached */
			assert(0);
			break;
		}
		return 0;
	}
	return argerror(L, 1);
}

/* xy function */
typedef void(*xyfunction)(int, float *, int, float *, int, float);

/* invokes an (x,y) subproram */
static int xy (lua_State *L, xyfunction s, int hasy, int hasalpha) {
	int index, i;
	float alpha;
	struct vector *x, *y;
	struct matrix *X, *Y;

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
			luaL_argcheck(L, y->size == x->size, 2,
					"dimension mismatch");
			s(x->size, x->values, x->inc, y->values, y->inc, alpha);
			return 0;
		}

		/* invoke subprogram on vector-matrix */
		switch (checktranspose(L, index)) {
		case CblasNoTrans:
			switch (Y->order) {
			case CblasRowMajor:
				luaL_argcheck(L, 1, x->size == Y->cols,
						"dimension mismatch");
				for (i = 0; i < Y->rows; i++) {
					s(x->size, x->values, x->inc,
							&Y->values[(size_t)i
							* Y->ld], 1, alpha);
				}
				break;

			case CblasColMajor:
				luaL_argcheck(L, 1, x->size == Y->rows,
						"dimension mismatch");
				for (i = 0; i < Y->cols; i++) {
					s(x->size, x->values, x->inc,
							&Y->values[(size_t)i
							* Y->ld], 1, alpha);
				}
				break;
			}
			break;

		case CblasTrans:
			switch (Y->order) {
			case CblasRowMajor:
				luaL_argcheck(L, 1, x->size == Y->rows,
						"dimension mismatch");
				for (i = 0; i < Y->rows; i++) {
					s(x->size, x->values, x->inc,
							&Y->values[(size_t)i],
							Y->ld, alpha);
				}
				break;

			case CblasColMajor:
				luaL_argcheck(L, 1, x->size == Y->cols,
						"dimension mismatch");
				for (i = 0; i < Y->cols; i++) {
					s(x->size, x->values, x->inc,
							&Y->values[(size_t)i],
							Y->ld, alpha);
				}
				break;
			}
			break;

		default:
			/* not reached */
			assert(0);
		}
		return 0;
	}
	X = luaL_testudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	if (X != NULL) {
		if (hasy) {
			Y = luaL_checkudata(L, 2, LUALINEAR_MATRIX_METATABLE);
			luaL_argcheck(L, X->order == Y->order, 2,
					"order mismatch");
			luaL_argcheck(L, X->rows == Y->rows && X->cols
					== Y->cols, 2, "dimension mismatch");
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
		switch (X->order) {
		case CblasRowMajor:
			for (i = 0; i < X->rows; i++) {
				s(X->cols, &X->values[(size_t)i * X->ld], 1,
						&Y->values[(size_t)i * Y->ld],
						1, alpha);
			}
			break;

		case CblasColMajor:
			for (i = 0; i < X->cols; i++) {
				s(X->rows, &X->values[(size_t)i * X->ld], 1,
						&Y->values[(size_t)i * Y->ld],
						1, alpha);
			}
			break;
		}
		return 0;
	}
	return argerror(L, 1);
}

/* wraps the SWAP subprogram */
static void _swap (int size, float *x, int incx, float *y, int incy,
		float alpha) {
	(void)alpha;
	cblas_sswap(size, x, incx, y, incy);
}

/* invokes the SWAP subprogram (y <-> x) */
static int swap (lua_State *L) {
	return xy(L, _swap, 1, 0);
}

/* wraps the COPY subprogram */
static void _copy (int size, float *x, int incx, float *y, int incy,
		float alpha) {
	(void)alpha;
	cblas_scopy(size, x, incx, y, incy);
}

/* invokes the COPY subprogram (y <- x) */
static int copy (lua_State *L) {
	return xy(L, _copy, 1, 0);
}

/* wraps the AXPY subprogram */
static void _axpy (int size, float *x, int incx, float *y, int incy,
		float alpha) {
	cblas_saxpy(size, alpha, x, incx, y, incy);
}

/* invokes the AXPY subprogram (y <- alpha x + y) */
static int axpy (lua_State *L) {
	return xy(L, _axpy, 1, 1);
}

/* wraps the SCAL subprogram */
static void _scal (int size, float *x, int incx, float *y, int incy,
			float alpha) {
	(void)y;
	(void)incy;
	cblas_sscal(size, alpha, x, incx);
}

/* invokes the SCAL subprogram (x <- alpha x) */
static int scal (lua_State *L) {
	return xy(L, _scal, 0, 1);
}

/* set operation implementation */
static void _set (int size, float *x, int incx, float *y, int incy,
			float alpha) {
	int i;

	(void)y;
	(void)incy;
	#pragma omp parallel for private(i) schedule(auto) \
			if(size >= LUALINEAR_OMP_MINSIZE)
	for (i = 0; i < size; i++) {
		x[(size_t)i * incx] = alpha;
	}
}

/* performs a set operation (x <- alpha) */
static int set (lua_State *L) {
	return xy(L, _set, 0, 1);
}

/* uniform RNG implementation */
static void _uniform (int size, float *x, int incx, float *y, int incy,
		float alpha) {
	int i;

	(void)y;
	(void)incy;
	(void)alpha;
	for (i = 0; i < size; i++) {
		*x = (float)random() * (1.0 / ((float)RAND_MAX + 1.0));
		x += incx;
	}
}

/* performs a uniform operation (x <- uniform) */
static int uniform (lua_State *L) {
	return xy(L, _uniform, 0, 0);
}

/* normal RNG implementation */
static void _normal (int size, float *x, int incx, float *y, int incy,
		float alpha) {
	int i;
	float u1, u2, r, s, c;

	(void)y;
	(void)incy;
	(void)alpha;
	for (i = 0; i < size - 1; i += 2) {
		do {
			u1 = (float)random() * (1.0 / (float)RAND_MAX);
			u2 = (float)random() * (1.0 / (float)RAND_MAX);
		} while (u1 <= -DBL_MAX);
		r = sqrt(-2.0 * logf(u1));
		sincosf(2 * M_PI * u2, &s, &c);
		*x = r * c;
		x += incx;
		*x = r * s;
		x += incx;
	}
	if (i < size) {
		do {
			u1 = (float)random() * (1.0 / (float)RAND_MAX);
			u2 = (float)random() * (1.0 / (float)RAND_MAX);
		} while (u1 <= -DBL_MAX);
		*x = sqrtf(-2.0 * logf(u1)) * cosf(2 * M_PI * u2);
		x += incx;
	}
}

/* performs a normal operation (x <- normal) */
static int normal (lua_State *L) {
	return xy(L, _normal, 0, 0);
}

/* inc operation implementation */
static void _inc (int size, float *x, int incx, float *y, int incy,
			float alpha) {
	int i;

	(void)y;
	(void)incy;
	#pragma omp parallel for private(i) schedule(auto) \
			if(size >= LUALINEAR_OMP_MINSIZE)
	for (i = 0; i < size; i++) {
		x[(size_t)i * incx] += alpha;
	}
}

/* performs a inc operation (x <- x + alpha) */
static int inc (lua_State *L) {
	return xy(L, _inc, 0, 1);
}

/* element-wise multiplication implementation, alpha = 1 */
static void _mul1 (int size, float *x, int incx, float *y, int incy,
		float alpha) {
	int i;

	(void)alpha;
	#pragma omp parallel for private(i) schedule(auto) \
			if(size >= LUALINEAR_OMP_MINSIZE)
	for (i = 0; i < size; i++) {
		y[(size_t)i * incy] *= x[(size_t)i * incx];
	}
}

/* element-wise multiplication implementation, alpha = -1 */
static void _mulm1 (int size, float *x, int incx, float *y, int incy,
		float alpha) {
	int i;

	(void)alpha;
	#pragma omp parallel for private(i) schedule(auto) \
			if(size >= LUALINEAR_OMP_MINSIZE)
	for (i = 0; i < size; i++) {
		y[(size_t)i * incy] /= x[(size_t)i * incx];
	}
}

/* element-wise multiplication implementation, alpha = any */
static void _mul (int size, float *x, int incx, float *y, int incy,
		float alpha) {
	int i;

	#pragma omp parallel for private(i) schedule(auto) \
			if(size >= LUALINEAR_OMP_MINSIZE)
	for (i = 0; i < size; i++) {
		y[(size_t)i * incy] *= pow(x[(size_t)i * incx], alpha);
	}
}

/* performs element-wise multiplication (y <- x^alpha .* y) */
static int mul (lua_State *L) {
	float alpha;

	alpha = luaL_optnumber(L, 3, 1.0);
	if (alpha == 1.0) {
		return xy(L, _mul1, 1, 1);
	}
	if (alpha == -1.0) {
		return xy(L, _mulm1, 1, 1);
	}
	return xy(L, _mul, 1, 1);
}

/* power raising operation implementation */
static void _pow (int size, float *x, int incx, float *y, int incy,
			float alpha) {
	int i;

	(void)y;
	(void)incy;
	#pragma omp parallel for private(i) schedule(auto) \
			if(size >= LUALINEAR_OMP_MINSIZE)
	for (i = 0; i < size; i++) {
		x[(size_t)i * incx] = pow(x[(size_t)i * incx], alpha);
	}
}

/* performs element-wise power raising (x <- x^alpha) */
static int powx (lua_State *L) {
	return xy(L, _pow, 0, 1);
}

/* apply function */
typedef float(*applyfunction)(float);

/* applies a function to a value */
static int apply (lua_State *L, applyfunction apply, int parallel) {
	struct vector *x;
	struct matrix *X;
	int i, j;
	size_t base;

	if (lua_type(L, 1) == LUA_TNUMBER) {
		lua_pushnumber(L, apply(lua_tonumber(L, 1)));
		return 1;
	}
	x = luaL_testudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	if (x != NULL) {
		#pragma omp parallel for private(i) schedule(auto) \
				if(parallel && x->size >= LUALINEAR_OMP_MINSIZE)
		for (i = 0; i < x->size; i++) {
			x->values[(size_t)i * x->inc] =
					apply(x->values[(size_t)i
					* x->inc]);
		}
		return 0;
	}
	X = luaL_testudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	if (X != NULL) {
		switch (X->order) {
		case CblasRowMajor:
			for (i = 0; i < X->rows; i++) {
				base = (size_t)i * X->ld;
				#pragma omp parallel for private(j) \
						schedule(auto) \
						if(parallel && X->cols \
						>= LUALINEAR_OMP_MINSIZE)
				for (j = 0; j < X->cols; j++) {
					X->values[base + j] = apply(
							X->values[base
							+ j]);
				}
			}
			break;

		case CblasColMajor:
			for (i = 0; i < X->cols; i++) {
				base = (size_t)i * X->ld;
				#pragma omp parallel for private(j) \
						schedule(auto) \
						if(parallel && X->rows \
						>= LUALINEAR_OMP_MINSIZE)
				for (j = 0; j < X->rows; j++) {
					X->values[base + j] = apply(
							X->values[base
							+ j]);
				}
			}
			break;
		}
		return 0;
	}
	return luaL_argerror(L, 1, lua_pushfstring(L, "number, vector, or "
			"matrix expected, got %s", luaL_typename(L, 1)));

}

/* sign function implementation */
static float _sign (float x) {
	if (x > 0) {
		return 1;
	}
	if (x < 0) {
		return -1;
	}
	return x;
}

/* sign function */
static int sign (lua_State *L) {
	return apply(L, _sign, 1);
}

/* abs function implementation */
static float _abs (float x) {
	return abs(x);
}

/* abs function */
static int absx (lua_State *L) {
	return apply(L, _abs, 1);
}

/* exp function */
static int expx (lua_State *L) {
	return apply(L, expf, 1);
}

/* log function */
static int logx (lua_State *L) {
	return apply(L, logf, 1);
}

/* logistic function implementation */
static float _logistic (float z) {
	return 1.0 / (1.0 + expf(-z));
}

/* logistic function */
static int logistic (lua_State *L) {
	return apply(L, _logistic, 1);
}

/* tanh function */
static int tanhx (lua_State *L) {
	return apply(L, tanhf, 1);
}

/* softplus function implementation */
static float _softplus (float x) {
	return logf(1 + expf(x));
}

/* softplus function */
static int softplus (lua_State *L) {
	return apply(L, _softplus, 1);
}

/* rectifier function implementation */
static float _rectifier (float x) {
	return x > 0.0 ? x : 0.0;
}

/* rectifier function */
static int rectifier (lua_State *L) {
	return apply(L, _rectifier, 1);
}

/* current Lua state */
static __thread lua_State *TL;

/* apply function implementation */
static float _apply (float x) {
	float result;

	lua_pushvalue(TL, -1);
	lua_pushnumber(TL, x);
	lua_call(TL, 1, 1);
	result = lua_tonumber(TL, -1);
	lua_pop(TL, 1);
	return result;
}

/* apply function */
static int applyx (lua_State *L) {
	luaL_checktype(L, 2, LUA_TFUNCTION);
	lua_settop(L, 2);
	TL = L;
	return apply(L, _apply, 0);
}

/* invokes the GEMV subprogram (y <- alpha A x + b y) */
static int gemv (lua_State *L) {
	struct matrix *A;
	struct vector *x, *y;	
	float alpha, beta;
	CBLAS_TRANSPOSE ta;
	int m, n;

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
	cblas_sgemv(A->order, ta, A->rows, A->cols, alpha, A->values, A->ld,
			x->values, x->inc, beta, y->values, y->inc);
	return 0;
}

/* invokes the GER subprogram (A <- alpha x y' + A) */
static int ger (lua_State *L) {
	struct vector *x, *y;	
	struct matrix *A;
	float alpha;

	/* check and process arguments */
	x = luaL_checkudata(L, 1, LUALINEAR_VECTOR_METATABLE);
	y = luaL_checkudata(L, 2, LUALINEAR_VECTOR_METATABLE);
	A = luaL_checkudata(L, 3, LUALINEAR_MATRIX_METATABLE);
	alpha = luaL_optnumber(L, 4, 1.0);
	luaL_argcheck(L, x->size == A->rows, 1, "dimension mismatch");
	luaL_argcheck(L, y->size == A->cols, 2, "dimension mismatch");

	/* invoke subprogram */
	cblas_sger(A->order, A->rows, A->cols, alpha, x->values, x->inc,
			y->values, y->inc, A->values, A->ld);
	return 0;
}

/* invokes the GEMM subprogram (C <- alpha A B + beta C) */
static int gemm (lua_State *L) {
	struct matrix *A, *B, *C;
	float alpha, beta;
	CBLAS_TRANSPOSE ta, tb;
	int m, n, ka, kb;

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
	cblas_sgemm(A->order, ta, tb, m, n, ka, alpha, A->values, A->ld,
			B->values, B->ld, beta, C->values, C->ld);
	return 0;
}

/* invokes the GESV subprogram */
static int gesv (lua_State *L) {
	struct matrix *A, *B;
	int *ipiv, result;

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
	result = LAPACKE_sgesv(A->order, A->rows, B->cols, A->values, A->ld,
			ipiv, B->values, B->ld);
	free(ipiv);
	lua_pushinteger(L, result);
	return 1;
}

/* invokes the GELS subprogram */
static int gels (lua_State *L) {
	struct matrix *A, *B;
	char ta;

	/* check and process arguments */
	A = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	B = luaL_checkudata(L, 2, LUALINEAR_MATRIX_METATABLE);
	luaL_argcheck(L, B->order == A->order, 2, "order mismatch");
	ta = lapacktranspose(checktranspose(L, 3));
	luaL_argcheck(L, B->rows == (ta == 'N' ? A->rows : A->cols), 2,
			"dimension mismatch");

	/* invoke subprogramm */
	lua_pushinteger(L, LAPACKE_sgels(A->order, ta, A->rows, A->cols,
			B->cols, A->values, A->ld, B->values, B->ld));
	return 1;
}

/* calculates the inverse of a matrix */
static int inv (lua_State *L) {
	struct matrix *A;
	int *ipiv, result;

	/* check and process arguments */
	A = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	luaL_argcheck(L, A->rows == A->cols, 1, "not square");

	/* invoke subprograms */
	ipiv = calloc(A->rows, sizeof(lapack_int));
	if (ipiv == NULL) {
		return luaL_error(L, "cannot allocate indexes");
	}
	result = LAPACKE_sgetrf(A->order, A->rows, A->cols, A->values, A->ld,
			ipiv);
	if (result != 0) {
		free(ipiv);
		lua_pushinteger(L, result);
		return 1;
	}
	result = LAPACKE_sgetri(A->order, A->rows, A->values, A->ld, ipiv);
	free(ipiv);
	lua_pushinteger(L, result);
	return 1;
}

/* calculates the determinant of a matrix */
static int det (lua_State *L) {
	struct matrix *A;
	float *copy, *d, *s, det;
	int n, *ipiv, result, neg, i;

	/* check and process arguments */
	A = luaL_checkudata(L, 1, LUALINEAR_MATRIX_METATABLE);
	luaL_argcheck(L, A->rows == A->cols, 1, "not square");
	n = A->rows;

	/* copy matrix */
	copy = calloc((size_t)n * n, sizeof(float));
	if (copy == NULL) {
		return luaL_error(L, "cannot allocate values");
	}
	d = copy;
	s = A->values;
	for (i = 0; i < n; i++) {
		memcpy(d, s, (size_t)n * sizeof(float));
		d += n;
		s += A->ld;
	}

	/* invoke subprograms */
	ipiv = calloc(n, sizeof(lapack_int));
	if (ipiv == NULL) {
		free(copy);
		return luaL_error(L, "cannot allocate indexes");
	}
	result = LAPACKE_sgetrf(A->order, n, n, copy, n, ipiv);
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
		det *= copy[(size_t)i * n + i];
		if (ipiv[i] != i + 1) {
			neg = !neg;
		}
	}
	free(copy);
	free(ipiv);
	lua_pushnumber(L, neg ? -det : det);
	return 1;
}


/*
 * Exported functions.
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
		{ "iamax", iamax },
		{ "sum", sum },
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
	lua_pushcfunction(L, vector_free);
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
