/*
 * Lua Linear unary vector functions
 *
 * Copyright (C) 2017-2023 Andre Naef
 */


#ifndef _LINEAR_UNARY_INCLUDED
#define _LINEAR_UNARY_INCLUDED


#include <lua.h>


typedef double (*linear_unary_function)(int size, double *x, int incx, union linear_arg *args);


int linear_unary(lua_State *L, linear_unary_function f, linear_param_t *params);
int linear_open_unary(lua_State *L);


#endif /* _LINEAR_UNARY_INCLUDED */
