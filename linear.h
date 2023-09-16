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


#define LUALINEAR_VECTOR_METATABLE  "linear.vector"
#define LUALINEAR_MATRIX_METATABLE  "linear.matrix"


struct vector {
        size_t   length;  /* length*/
        size_t   inc;     /* increment to next value*/
        double  *values;  /* values */
        int      ref;     /* Lua reference */
};

struct matrix {
        size_t        rows;    /* number of rows */
        size_t        cols;    /* number of columns*/
        size_t        ld;      /* increment to next major vector */
        CBLAS_ORDER   order;   /* order */
        double       *values;  /* values */
        int           ref;     /* Lua reference */
};


int luaopen_linear(lua_State *L);


#endif /* _LUALINEAR_INCLUDED */
