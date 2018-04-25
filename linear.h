/* Provides the Lua Linear module. See LICENSE for license terms. */

#ifndef LUALINEAR_INCLUDED
#define LUALINEAR_INCLUDED

#include <assert.h>
#include <stdlib.h>
#include <lua.h>
#include <lauxlib.h>
#include <cblas.h>

/**
 * Meta tables.
 */
#define LUALINEAR_VECTOR_METATABLE "linear.vector"
#define LUALINEAR_MATRIX_METATABLE "linear.matrix"

/**
 * OpenMP.
 */
#define LUALINEAR_OMP_MINSIZE 2048

/**
 * Vector.
 */
struct vector {
        int size;
        int inc;
        float *values;
        int ref;
};

/**
 * Pushes a new vector onto the stack.
 */
inline struct vector *lualinear_newvector (lua_State *L, int size) {
	struct vector *vector;

	assert(size >= 1);
	vector = (struct vector *)lua_newuserdata(L, sizeof(struct vector));
	vector->size = size;
	vector->inc = 1;
	vector->values = NULL;
	vector->ref = LUA_NOREF;
	luaL_getmetatable(L, LUALINEAR_VECTOR_METATABLE);
	lua_setmetatable(L, -2);
	vector->values = (float *)calloc(size, sizeof(float));
	if (vector->values == NULL) {
		luaL_error(L, "cannot allocate values");
	}
        return vector;
}

/**
 * Pushes an existing vector onto the stack.
 */
inline struct vector *lualinear_wrapvector (lua_State *L, int size,
		float *values) {
	struct vector *vector;

	assert(size >= 1);
	vector = (struct vector *)lua_newuserdata(L, sizeof(struct vector));
	vector->size = size;
	vector->inc = 1;
	vector->values = values;
	vector->ref = LUA_REFNIL;
	luaL_getmetatable(L, LUALINEAR_VECTOR_METATABLE);
	lua_setmetatable(L, -2);
	return vector;
}

/**
 * Matrix.
 */
struct matrix {
        int rows;
        int cols;
        int ld;
        CBLAS_ORDER order;
        float *values;
        int ref;
};

/**
 * Pushes a new matrix onto the stack.
 */
inline struct matrix *lualinear_newmatrix (lua_State *L, int rows, int cols,
                CBLAS_ORDER order) {
	struct matrix *matrix;

	assert(rows >= 1 && cols >= 1);
	matrix = (struct matrix *)lua_newuserdata(L, sizeof(struct matrix));
	matrix->rows = rows;
	matrix->cols = cols;
	matrix->ld = order == CblasRowMajor ? cols : rows;
	matrix->order = order;
	matrix->values = NULL;
	matrix->ref = LUA_NOREF;
	luaL_getmetatable(L, LUALINEAR_MATRIX_METATABLE);
	lua_setmetatable(L, -2);
	matrix->values = (float *)calloc((size_t)rows * cols, sizeof(float));
	if (matrix->values == NULL) {
		luaL_error(L, "cannot allocate values");
	}
	return matrix;
}

/**
 * Pushes an existing matrix onto the stack-
 */
inline struct matrix *lualinear_wrapmatrix (lua_State *L, int rows, int cols,
		CBLAS_ORDER order, float *values) {
	struct matrix *matrix;

	assert(rows >= 1 && cols >= 1);
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

/**
 * Opens the Linear module in a Lua state.
 */
int luaopen_linear (lua_State *L);

#endif /* LUALINEAR_INCLUDED */
