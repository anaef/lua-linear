/*
 * Lua Linear elementary functions
 *
 * Copyright (C) 2017-2023 Andre Naef
 */


#ifndef _LINEAR_ELEMENTARY_INCLUDED
#define _LINEAR_ELEMENTARY_INCLUDED


#include <lua.h>


typedef void (*linear_elementary_function)(size_t size, double *x, size_t incx, linear_arg_u *args);


int linear_elementary(lua_State *L, linear_elementary_function f, linear_param_t *params);
int linear_open_elementary(lua_State *L);


#endif /* _LINEAR_ELEMENTARY_INCLUDED */
