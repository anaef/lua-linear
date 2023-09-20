# Lua Linear

## Introduction

Lua Linear provides comprehensive linear algebra support for the Lua programming language. Where
applicable, the BLAS and LAPACK implementations on the system are used.


## Build, Test, and Install

### Dependencies

You may need to install the following packages to build Lua Linear:

* libopenblas-dev
* liblapacke-dev


### Building and Installing with LuaRocks

To build and install with LuaRocks, run:

```
luarocks install lua-linear
```


### Building, Testing and Installing with Make

Lua Linear comes with a simple Makefile. Please adapt the Makefile to your environment, and then
run:

```
make
make test
make install
```


## Release Notes

Please see the [release notes](NEWS.md) document.


## Documentation

Please browse the [documentation](doc/) folder for the extensive documentation.


## Limitations

Lua Linear supports Lua 5.1, Lua 5.2, Lua 5.3, and Lua 5.4.

Lua Linear has been built and tested on Ubuntu Linux (64-bit).

Lua Linear uses the BLAS and LAPACK implementations on the system.


## License

Lua Linear is released under the MIT license. See LICENSE for license terms.
