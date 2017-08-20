# Lua Linear README

## Introduction

Lua Linear provides linear algebra functions for the Lua programming language.


## Build, Test, and Install

Lua Linear comes with a simple Makefile. Please adapt the Makefile to your environment, and then run:

```
make <platform>
make test
make install
```

## Reference

### Types

#### `vector`

A vector of double values. Vectors have support for the Lua length operator `#`,
as well as index access to get and set the component values of the vector.
Indexing is 1-based.


#### `matrix`

A matrix of double values. Matrices have support for the Lua length operator
`#`, as well as index access to get vectors referencing the rows or columns
of the matrix. The order of the matrix determines whether row or column vectors
are returned. Indexing is 1-based.


### Functions

#### `linear.vector (size)`

Creates a new vector of the specified size.


#### `linear.matrix (rows, cols [, order])`

Creates a new matrix of the specified size. Order is one of `row`, `col` and
defaults to creating a matrix with row major order if omitted.


#### `linear.type (value)`

Returns `vector` if the value is a vector, `matrix` if the value is a matrix,
and `nil` otherwise.


#### `linear.size (vector|matrix)`

Returns the size of a vector, or two values for the number of rows and columns
of a matrix.


#### `linear.tvector (matrix, index)`

Returns a transposed vector of a matrix. This is a column vector in case of
a row major matrix, and a row vector in case of a column major matrix.


#### `linear.sub (vector [, start] [, end])`

Returns a sub vector referencing the underlying vector. Start and end are
inclusive. If start is omitted, it defaults to 1. If end is omitted, it
defaults to the size of the vector.


#### `linear.sub (matrix [, majorstart] [, minorstart] [, majorend] [, minorend]`

Returns a sub matrix referencing the underlying matrix. All bounding values are
inclusive. If a start value is omitted, it defaults to 1. If an end value is
omitted, it defaults to the number of rows or columns of the matrix. For a row
major matrix, majorstart and majorend refer to the rows of the matrix, and
minorstart and minorend refer to the columns of the matrix. For a column major
matrix, the meaning is swapped.


#### `linear.unwind (matrix {, matrix}, vector)`

Unwinds one or more matrices into a vector. The number of values of the matrices
and the size of the vector must match.


#### `linear.reshape (vector, matrix {, matrix})`

Reshapes a vector into one or more matrices. The size of the vector and the
number of values of the matrices must match.


#### `linear.totable (vector|matrix)`

Converts a vector or matrix into a Lua table. The Lua table uses only standard
Lua types, and can be converted back to a vector or matrix by passing it to
`linear.tolinear`.


#### `linear.tolinear (table)`

Converts a Lua table returned by `linear.totable` into a vector or matrix.


#### `linear.dot`

TODO


#### `linear.nrm2`

TODO


#### `linear.asum`

TODO


#### `linear.iamax`

TODO


#### `linear.sum`

TODO


#### `linear.swap`

TODO


#### `linear.copy`

TODO


#### `linear.axpy`

TODO


#### `linear.scal`

TODO


#### `linear.set`

TODO


#### `linear.rand`

TODO


#### `linear.inc`

TODO


#### `linear.mul`

TODO


#### `linear.sign`

TODO


#### `linear.abs`

TODO


#### `linear.logistic`

TODO


#### `linear.tanh`

TODO


#### `linear.softplus`

TODO


#### `linear.rectifier`

TODO


#### `linear.gemv`

TODO


#### `linear.ger`

TODO


#### `linear.gemm`

TODO


#### `linear.gesv`

TODO


#### `linear.gels`

TODO


#### `linear.inv`

TODO


#### `linear.det`

TODO


## Limitations

Lua Linear supports Lua 5.2.

Lua Linear has been built and tested on Ubuntu Linux (64-bit).

Lua Linear uses the BLAS and LAPACK implementations on the system.


## License

Lua TZ is released under the MIT license. See LICENSE for license terms.
