# Lua Linear

## Introduction

Lua Linear provides comprehensive linear algebra support for the Lua programming language. Where
applicable, the BLAS and LAPACK implementations on the system are used.


## Build, Test, and Install

### Dependencies

You may need to install the following packages to compile Lua Linear:

* libopenblas-dev
* liblapacke-dev


### Building with Make

Lua Linear comes with a simple Makefile. Please adapt the Makefile to your environment, and then
run:

```
make
make test
make install
```


## Documentation

Please browse the [documentation](doc/) folder for the extensive documentation.


## Limitations

Lua Linear supports Lua 5.2, and Lua 5.3.

Lua Linear has been built and tested on Ubuntu Linux (64-bit).

Lua Linear uses the BLAS and LAPACK implementations on the system.


## License

Lua Linear is released under the MIT license. See LICENSE for license terms.
