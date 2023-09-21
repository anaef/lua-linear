# Lua Linear

## Introduction

Lua Linear provides comprehensive linear algebra support for the Lua programming language. Where
applicable, the BLAS and LAPACK implementations on the system are used.

Here are some quick examples:

```lua
local linear = require("linear")

-- Calculate an inner product
local x = linear.tolinear({ 1, 2, 3 })
local y = linear.tolinear({ 3, 2, 1 })
print("x^T y", linear.dot(x, y))  -- should be 10

-- Generate and analyze normally distributed random numbers with mean 10, std 5, i.e., N(10,5)
local x = linear.vector(100)
linear.normal(x)   -- N(0,1)
linear.scal(x, 5)  -- N(0,5)
linear.inc(x, 10)  -- N(10,5)
print("N(10, 5)", x[1], x[2], x[3], "...")
print("mean, std", linear.mean(x), linear.std(x, 1))  -- should approximate 10, 5

-- Solve a system of linear equations: x + 2y = 7, 2x - y = 9
local A = linear.tolinear({ { 1, 2 }, { 2, -1 } })
local B = linear.matrix(2, 1)
local b = linear.tvector(B, 1)  -- column vector
b[1], b[2] = 7, 9
linear.gesv(A, B)
print("solutions", b[1], b[2])  -- should be 5, 1

-- Calculate pair-wise Pearson correlation coefficients of 10k uniform random numbers
local A = linear.matrix(10000, 3)
linear.uniform(A)
local C = linear.matrix(3, 3)
linear.corr(A, C)
print("correlations", C[1][2], C[1][3], C[2][3])  -- should be close to zero
```

Output:

```
x^T y   10.0
N(10, 5)        7.6755368439453 11.817516987964 11.047801936107 ...
mean, std       9.6617642376046 4.8338730760376
solutions       5.0     1.0
correlations    -0.0026144290886127     0.0063412466674457      -0.004072920697946
```


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
