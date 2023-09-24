/*
 * Lua Linear binary vector functions
 *
 * Copyright (C) 2017-2023 Andre Naef
 */


#ifndef _LINEAR_BINARY_INCLUDED
#define _LINEAR_BINARY_INCLUDED


#include <lua.h>


typedef void (*linear_binary_function)(int size, double *x, int incx, double *y, int incy,
		linear_arg_u *args);


int linear_binary(lua_State *L, linear_binary_function s, linear_param_t *params);
int linear_open_binary(lua_State *L);


#endif /* _LINEAR_BINARY_INCLUDED */
