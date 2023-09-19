/*
 * Lua Linear
 *
 * Copyright (C) 2017-2023 Andre Naef
 */


#ifndef _LUALINEAR_INCLUDED
#define _LUALINEAR_INCLUDED


#include <assert.h>
#include <stdlib.h>
#include <lua.h>
#include <lauxlib.h>
#include <cblas.h>


#define LUALINEAR_VECTOR  "linear.vector"  /* vector metatable*/
#define LUALINEAR_MATRIX  "linear.matrix"  /* matrix metatable */


struct data {
	size_t   refs;    /* number of references */
};

struct vector {
	size_t        length;  /* length*/
	size_t        inc;     /* increment to next value */
	struct data  *data;    /* shared data */
	double       *values;  /* values */
};

struct matrix {
	size_t        rows;    /* number of rows */
	size_t        cols;    /* number of columns*/
	size_t        ld;      /* increment to next major vector */
	CBLAS_ORDER   order;   /* order */
	struct data  *data;    /* shared data */
	double       *values;  /* values */
};


int luaopen_linear(lua_State *L);


#endif /* _LUALINEAR_INCLUDED */
