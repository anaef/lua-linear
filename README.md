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
as well as index access to get and set the elements of the vector. Indexing is
1-based.


#### `matrix`

A matrix of double values. Matrices have support for the Lua length operator
`#`, as well as index access to get vectors referencing the rows or columns
of the matrix. For a row major matrix, row vectors are returned, and for a
columm major matrix, column vectors are returned. Indexing is 1-based.


### Functions

#### `linear.vector (size)`

Creates a new vector of the specified size.


#### `linear.matrix (rows, cols [, order])`

Creates a new matrix of the specified size. Order is one of `row`, `col` and
defaults to creating a matrix with row major order if omitted or `nil`.


#### `linear.type (value)`

Returns `vector` if the value is a vector, `matrix` if the value is a matrix,
and `nil` otherwise.


#### `linear.size (vector|matrix)`

Returns the number of elements of a vector, or two values for the number of
rows and columns of a matrix.


#### `linear.tvector (matrix, index)`

Returns a transposed vector of a matrix. This is a column vector in case of
a row major matrix, and a row vector in case of a column major matrix.


#### `linear.sub (vector [, start] [, end])`

Returns a sub vector referencing the underlying vector. Start and end are
inclusive. If start is omitted or `nil`, it defaults to 1. If end is omitted or
`nil`, it defaults to the size of the vector.


#### `linear.sub (matrix [, majorstart [, minorstart [, majorend [, minorend]]]])`

Returns a sub matrix referencing the underlying matrix. All bounding values are
inclusive. If a start value is omitted or `nil`, it defaults to 1. If an end
value is omitted or `nil`, it defaults to the number of rows or columns of the
matrix. For a row major matrix, majorstart and majorend refer to the rows of
the matrix, and minorstart and minorend refer to the columns of the matrix. For
a column major matrix, the meaning is swapped.


#### `linear.unwind (matrix {, matrix}, vector)`

Unwinds one or more matrices into a vector. The number of elements of the
matrices and the vector must match.


#### `linear.reshape (vector, matrix {, matrix})`

Reshapes a vector into one or more matrices. The number of elements  of the
vector and the matrices must match.


#### `linear.totable (vector|matrix)`

Converts a vector or matrix into a Lua table. The Lua table uses only standard
Lua types, and can be converted back to a vector or matrix by passing it to
`linear.tolinear`.


#### `linear.tolinear (table)`

Converts a Lua table returned by `linear.totable` into a vector or matrix.


#### `linear.dot (vector x, vector y)`

Returns the dot product of two vectors, formally `x^T y`.


#### `linear.nrm2 (vector)`

Returns the Euclidean norm (also known as L2 norm) of a vector, formally
`||x||_2`.


#### `linear.asum (vector)`

Returns the absolute-value norm (also known as L1 norm) of a vector, formally
`||x||`.


#### `linear.iamax (vector)`

Returns the index of the first element of a vector having the largest absolute
value, formally `argmax |x_k|`.


#### `linear.sum (vector)`

Returns the sum of the elements of a vector, formally `sigma x_k`.


#### `linear.swap (vector|matrix x, vector|matrix y)`

Swaps the elements of two vectors or matrices, formally `x <-> y`.


#### `linear.copy (vector|matrix x, vector|matrix y)`

Copies the elements of a vector or matrix to another vector or matrix, formally
`y <- x`.


#### `linear.axpy (vector|matrix x, vector|matrix y [, alpha])`

Adds a scaled vector or matrix to another vector or matrix, formally
`y <- alpha x + y`. If omitted or `nil`, alpha defaults to `1.0`.


#### `linear.scal (vector|matrix x [, alpha])`

Scales a vecctor or matrix, formally `x <- alpha x`. If omitted or `nil`, alpha
defaults to `1.0`.


#### `linear.set (vector|matrix x [, alpha])`

Sets the elements of a vector or matrix to a constant, formally `x <- alpha`. If
omitted or `nil`, alpha defaults to `1.0`.


#### `linear.rand (vector|matrix x)`

Sets the elements of a vector or matrix to random values, formally
`x <- rand` where `0.0 <= rand < 1.0`.


#### `linear.inc (vector|matrix x [, alpha])`

Increments the elements of a vector or matrix, formally `x <- x + alpha`. If
omitted or `nil`, alpha defaults to `1.0`.


#### `linear.mul (vector|matrix x, vector|matrix y)`

Performs element-wise multiplication on two vectors or matrices, formally
`y <- x .* y`.


#### `linear.sign (number|vector|matrix x)`

Applies the sign function to a vector or matrix, or returns the sign of a
number, formally `x <- sgn(x)`. The sign is `1` if x > 0, `-1` if x < 0, and
`x` otherwise.


#### `linear.abs (number|vector|matrix x)`

Applies the absoute value function to a vector or matrix, or returns the
absolute value of a number, formally `x <- abs(x)`.


#### `linear.logistic (number|vector|matrix x)`

Applies the logistic function to a vector or matrix, or returns the logistic
function of a number, formally `x <- 1 / (1 + exp(-x))`.


#### `linear.tanh (number|vector|matrix x)`

Applies the hyperbolic tangent function to a vector or matrix, or returns the
hyperbolic tangent of a number, formally `x <- tanh(x)`.


#### `linear.softplus (number|vector|matrix x)`

Applies the softplus function to a vector or matrix, or returns the softplus
function of a number, formally `x <- log(1 + exp(x))`.


#### `linear.rectifier (number|vector|matrix x)`

Applies the rectifier function (also known as rectified linear unit, ReLU) to a
vector or matrix, or returns the recitifier function of a number, formally
`x <- max(0, x)`.


#### `linear.gemv (matrix A, vector x, vector y [, transpose A [, alpha [, beta]]])`

Performs a matrix-vector product and addition operation, formally
`y <- alpha A x + beta y`. Transpose is one of `notrans`, `trans`, and defaults
to `notrans` if omitted or `nil`. If set to `trans`, the operation is performed
on `A^T`. If omitted or `nil`, alpha default to `1.0`, and beta defaults to
`0.0`.


#### `linear.ger (vector x, vector y, matrix A [, alpha])`

Performs a vector-vector product and addition operation, formally
`A <- alpha x y^T + A`. If omitted or `nil`, alpha defaults to `1.0`.


#### `linear.gemm (matrix A, matrix B, matrix C [, transpose A [, transpose B [, alpha [, beta ]]]])`

Performs a matrix-matrix product and addition operation, formally
`C <- alpha A B + beta C`. Transpose arguments function as in the `linear.gemv`
function. If omitted or `nil`, alpha defaults to `1.0`, and beta defaults to
`0.0`.


#### `linear.gesv (matrix A, matrix B)`

Solves systems of linear equations, formally `A X = B`. On input, each column
of B represents the right-hand sides of a system. On output, the solutions X
are stored in B.


#### `linear.gels (matrix A, matrix B [, transpose A])`

Solves overdetermined or underdermined systems of linear equations, formally
`A X = B`. On input, each column of B represents the right-hand sides of a
system. On output, the solutions X are stored in B. In case of overdetermined
systems, the function solves the least squares problems `min ||b - A x||_2`,
and in case of underdetermined systems, the function finds minimun norm
solutions. The transpose argument functions as in the `linear.gemv` function.


#### `linear.inv (matrix A)`

Inverts a matrix, formally `A <- A^-1`. Returns 0 if the calculation was
successful, and a positive value if the matrix is singular and cannot be
inverted. In this case, the elements of the matrix are undefined.


#### `linear.det (matrix A)`

Returns the determinant of a matrix, formally `det(A)`.


## Limitations

Lua Linear supports Lua 5.2.

Lua Linear has been built and tested on Ubuntu Linux (64-bit).

Lua Linear uses the BLAS and LAPACK implementations on the system.


## License

Lua Linear is released under the MIT license. See LICENSE for license terms.