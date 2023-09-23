/*
 * Lua Linear
 *
 * Copyright (C) 2017-2023 Andre Naef
 */


#ifndef _LINEAR_LINEAR_INCLUDED
#define _LINEAR_LINEAR_INCLUDED


#include <assert.h>
#include <stdlib.h>
#include <lua.h>
#include <lauxlib.h>
#include <cblas.h>


#define LINEAR_VECTOR     "linear.vector"  /* vector metatable*/
#define LINEAR_MATRIX     "linear.matrix"  /* matrix metatable */
#define LINEAR_PARAM_MAX  5                /* maximum number of extra parameters */


struct linear_data {
	size_t   refs;    /* number of references */
};

struct linear_vector {
	size_t              length;  /* length*/
	size_t              inc;     /* increment to next value */
	struct linear_data  *data;    /* shared data */
	double              *values;  /* components */
};

struct linear_matrix {
	size_t               rows;    /* number of rows */
	size_t               cols;    /* number of columns*/
	size_t               ld;      /* increment to next major vector */
	CBLAS_ORDER          order;   /* order */
	struct linear_data  *data;    /* shared data */
	double              *values;  /* elements */
};

struct linear_param {
	const char  *name;      /* name */
	char         type;      /* 'n' number, 'd' delta degrees of freedom */
	union {
		double  defn;   /* default number */
		size_t  defd;   /* default delta degrees of freedom */
	};
};

union linear_arg {
	double  n;  /* number */
	size_t  d;  /* delta degrees of freedom */
};

typedef void (*linear_elementary_function)(int size, double *x, int incx, union linear_arg *args);
typedef double (*linear_unary_function)(int size, double *x, int incx, union linear_arg *args);
typedef void (*linear_binary_function)(int size, double *x, int incx, double *y, int incy,
		union linear_arg *args);


struct linear_vector *linear_create_vector(lua_State *L, size_t length);
struct linear_matrix *linear_create_matrix(lua_State *L, size_t rows, size_t cols, CBLAS_ORDER order);
int linear_elementary(lua_State *L, linear_elementary_function f, struct linear_param *params);
int linear_unary(lua_State *L, linear_unary_function f, struct linear_param *params);
int linear_binary(lua_State *L, linear_binary_function s, struct linear_param *params);
int luaopen_linear(lua_State *L);


#endif /* _LINEAR_LINEAR_INCLUDED */
