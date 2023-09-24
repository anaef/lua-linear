/*
 * Lua Linear core
 *
 * Copyright (C) 2017-2023 Andre Naef
 */


#ifndef _LINEAR_CORE_INCLUDED
#define _LINEAR_CORE_INCLUDED


#include <stdint.h>
#include <lua.h>
#include <cblas.h>


#define LINEAR_VECTOR      "linear.vector"   /* vector metatable*/
#define LINEAR_MATRIX      "linear.matrix"   /* matrix metatable */
#define LINEAR_RANDOM      "linear.random"   /* random state */
#define LINEAR_PARAMS_MAX  5                 /* maximum number of extra parameters */


typedef struct linear_data_s {
	size_t   refs;    /* number of references */
} linear_data_t;

typedef struct linear_vector_s {
	size_t              length;  /* length*/
	size_t              inc;     /* increment to next value */
	linear_data_t  *data;    /* shared data */
	double              *values;  /* components */
} linear_vector_t;

typedef struct linear_matrix_s {
	size_t               rows;    /* number of rows */
	size_t               cols;    /* number of columns*/
	size_t               ld;      /* increment to next major vector */
	CBLAS_ORDER          order;   /* order */
	linear_data_t  *data;    /* shared data */
	double              *values;  /* elements */
} linear_matrix_t;

typedef struct linear_param_s {
	const char  *name;      /* name */
	char         type;      /* 'n' number, 'd' ddof, 'r' random state */
	union {
		double  defn;   /* default number */
		size_t  defd;   /* default ddof */
	};
} linear_param_t;

typedef union linear_arg {
	double     n;  /* number */
	size_t     d;  /* ddof */
	uint64_t  *r;  /* random state */
} linear_arg_u;


CBLAS_ORDER linear_checkorder(lua_State *L, int index);
int linear_checkargs(lua_State *L, linear_param_t *params, size_t size, int index,
	linear_arg_u *args);
int linear_argerror(lua_State *L, int index, int numok);
double linear_random(uint64_t *r);
linear_vector_t *linear_create_vector(lua_State *L, size_t length);
linear_matrix_t *linear_create_matrix(lua_State *L, size_t rows, size_t cols, CBLAS_ORDER order);
int luaopen_linear(lua_State *L);


#endif /* _LINEAR_CORE_INCLUDED */
