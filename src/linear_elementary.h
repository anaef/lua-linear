/*
 * Lua Linear elementary functions
 *
 * Copyright (C) 2017-2023 Andre Naef
 */


#ifndef _LINEAR_ELEMENTARY_INCLUDED
#define _LINEAR_ELEMENTARY_INCLUDED


#include <lua.h>


typedef void (*linear_elementary_function)(int size, double *x, int incx, union linear_arg *args);


int linear_elementary(lua_State *L, linear_elementary_function f, struct linear_param *params);
int linear_open_elementary(lua_State *L);


#endif /* _LINEAR_ELEMENTARY_INCLUDED */
