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
        size_t   size;
        size_t   inc;
        double  *values;
        int      ref;
};

struct matrix {
        size_t        rows;
        size_t        cols;
        size_t        ld;
        CBLAS_ORDER   order;
        double       *values;
        int           ref;
};


int luaopen_linear(lua_State *L);


#endif /* _LUALINEAR_INCLUDED */
