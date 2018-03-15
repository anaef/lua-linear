# Lua Linear README

## Introduction

Lua Linear provides comprehensive linear algebra support for the Lua
programming language. Where applicable, the BLAS and LAPACK implementations on
the system are used.


## Build, Test, and Install

Lua Linear comes with a simple Makefile. Please adapt the Makefile to your
environment, and then run:

```
make <platform>
make test
make install
```

## Reference

### Types

#### `vector`

A vector of float values. Vectors have support for the ipairs function, the
length operator `#`, as well as index access to get and set the elements of the
vector. Indexing is 1-based.


#### `matrix`

A matrix of float values. Matrices have support for the ipairs function, the
length operator `#`, as well as index access to get major order vectors
referencing the underlying matrix. For a row major matrix, row vectors are
returned, and for a columm major matrix, column vectors are returned. Indexing
is 1-based.


### Functions

#### `linear.vector (size)`

Creates a new vector of the specified size.


#### `linear.matrix (rows, cols [, order])`

Creates a new matrix of the specified size. Order is one of `row`, `col`, and
defaults to creating a matrix with row major order.


#### `linear.type (value)`

Returns `vector` if the value is a vector, `matrix` if the value is a matrix,
and `nil` otherwise.


#### `linear.size (vector|matrix)`

Returns the number of elements of a vector, or three values for the number of
rows and columns as well as the order of a matrix.


#### `linear.tvector (matrix, index)`

Returns a transposed (or minor order) vector referencing the underlying matrix.
This is a column vector in case of a row major matrix, and a row vector in case
of a column major matrix.


#### `linear.sub (vector [, start [, end]])`

Returns a sub vector referencing the underlying vector. Start and end are
inclusive. The argument start defaults to 1, and the argument end defaults to
the size of the vector.


#### `linear.sub (matrix [, majorstart [, minorstart [, majorend [, minorend]]]])`

Returns a sub matrix referencing the underlying matrix. All bounding values are
inclusive. Start values default to 1, and end values default to the number of
rows or columns of the matrix. For a row major matrix, majorstart and majorend
relate to the rows of the matrix, and minorstart and minorend relate to the
columns of the matrix. For a column major matrix, the relation is inverse.


#### `linear.unwind (matrix {, matrix}, vector)`

Unwinds one or more matrices into a vector. The number of elements of the
matrices and the vector must match.


#### `linear.reshape (vector, matrix {, matrix})`

Reshapes a vector into one or more matrices. The number of elements of the
vector and the matrices must match.


#### `linear.totable (vector|matrix)`

Converts a vector or matrix into a Lua table. The Lua table uses only standard
Lua types, and can be converted back to a vector or matrix by passing it to the
`linear.tolinear` function.


#### `linear.tolinear (table)`

Converts a Lua table returned by the `linear.totable` function into a vector or
matrix.


#### `linear.dot (vector x, vector y)`

Returns the dot product of two vectors, formally `x^T y`.


#### `linear.nrm2 (vector x)`

Returns the Euclidean norm (also known as L2 norm) of a vector, formally
`||x||_2`.


#### `linear.asum (vector x)`

Returns the absolute-value norm (also known as L1 norm) of a vector, formally
`||x||`.


#### `linear.iamax (vector x)`

Returns the index of the first element of a vector having the largest absolute
value, formally `argmax_i |x_i|`.


#### `linear.sum (vector|matrix x [, vector y [, transpose]])`

Returns the sum of the elements of a vector, formally `sigma_i x_i`, or sets a
vector to the sum of each major order vector of a matrix, formally
`y_i <- sigma_j x_i,j`. The argument transpose is one of `notrans`, `trans`, and
defaults to `notrans`. If set to `trans`, the vector is set to the sum of each
minor order vector of the matrix, formally `y_j <- sigma_i x_i_j`.


#### `linear.swap (vector|matrix x, vector|matrix y [, transpose])`

Swaps the elements of two vectors or matrices, formally `x <-> y`. The function
can be invoked with a vector and a matrix to swap the vector repeatedly with
the major order vectors of the matrix. The argument transpose is one of
`notrans`, `trans`, and defaults to `notrans`. If set to `trans`, the vector
is swapped with the minor order vectors of the matrix.


#### `linear.copy (vector|matrix x, vector|matrix y [, transpose])`

Copies the elements of a vector or matrix to another vector or matrix, formally
`y <- x`. The function can be invoked with a vector and a matrix to copy the
vector repeatedly to the major order vectors of the matrix. The argument
transpose is one of `notrans`, `trans`, and defaults to `notrans`. If set to
`trans`, the vector is copied to the minor order vectors of the matrix.


#### `linear.axpy (vector|matrix x, vector|matrix y [, alpha [, transpose]])`

Adds a scaled vector or matrix to another vector or matrix, formally
`y <- alpha x + y`. The function can be invoked with a vector and a matrix to
add the vector repatedly to the major order vectors of the matrix. The argument
alpha defaults to `1.0`. The argument transpose is one of `notrans`, `trans`,
and defaults to `notrans`. If set to `trans`, the vector is added to the
minor order vectors of the matrix.


#### `linear.scal (vector|matrix x [, alpha])`

Scales a vecctor or matrix, formally `x <- alpha x`. The argument alpha defaults
to `1.0`.


#### `linear.set (vector|matrix x [, alpha])`

Sets the elements of a vector or matrix to a constant, formally `x <- alpha`.
The argument alpha defaults to `1.0`.


#### `linear.uniform (vector|matrix x)`

Sets the elements of a vector or matrix to uniformly distributed random values,
formally `x <- uniform` where `uniform ~ 𝒰(0, 1 - ε)`.


#### `linear.normal (vector|matrix x)`

Sets the elements of a vector or matrix to normally distributed random values,
formally `x <- normal` where `normal ~ 𝒩(0, 1)`.


#### `linear.inc (vector|matrix x [, alpha])`

Increments the elements of a vector or matrix, formally `x <- x + alpha`. The
argument alpha defaults to `1.0`.


#### `linear.mul (vector|matrix x, vector|matrix y [, alpha [, transpose]])`

Performs element-wise power multiplication on two vectors or matrices, formally
`y <- x^alpha * y`.  The function can be invoked with a vector and a matrix to
multiply the vector repatedly with the major order vectors of the matrix. The
argument alpha defaults to `1.0`. The argument transpose is one of `notrans`,
`trans`, and defaults to `notrans`. If set to `trans`, the vector is multiplied
with the minor order vectors of the matrix.


#### `linear.pow (vector|matrix x, [, alpha])`

Raises the elements of a vector or matrix to a power, formally `x <- x^alpha`.
The argument alpha defaults to `1.0`.


#### `linear.sign (number|vector|matrix x)`

Applies the sign function to the elements of a vector or matrix, formally
`x <- sgn(x)`, or returns the sign of a number, formally `sgn(x)`. The sign is
`1` if x > 0, `-1` if x < 0, and `x` otherwise.


#### `linear.abs (number|vector|matrix x)`

Applies the absolute value function to the elements of a vector or matrix,
formally `x <- abs(x)`, or returns the absolute value of a number, formally
`abs(x)`.


#### `linear.exp (number|vector|matrix x)`

Applies the exponent function to the elements of a vector or matrix, formally
`x <- e^x`, or returns the exponent of a number, formally `e^x`.


#### `linear.log (number|vector|matrix x)`

Applies the natural logarithm function to the elements of a vector or matrix,
formally `x <- log(x)`, or returns the natural logarithm of a number, formally
`log(x)`.


#### `linear.logistic (number|vector|matrix x)`

Applies the logistic function to the elements of a vector or matrix, formally
`x <- 1 / (1 + e^-x)`, or returns the logistic function of a number, formally
`1 / (1 + e^-x)`.


#### `linear.tanh (number|vector|matrix x)`

Applies the hyperbolic tangent function to the elements of a vector or matrix,
formally `x <- tanh(x)`, or returns the hyperbolic tangent of a number, formally
`tanh(x)`.


#### `linear.softplus (number|vector|matrix x)`

Applies the softplus function to the elements of a vector or matrix, formally
`x <- log(1 + e^x)`, or returns the softplus function of a number, formally
`log(1 + e^x)`.


#### `linear.rectifier (number|vector|matrix x)`

Applies the rectifier function (also known as rectified linear unit, ReLU) to
the elements of a vector or matrix, formally `x <- max(0, x)`, or returns the
recitifier function of a number, formally `max(0, x)`.


#### `linear.apply (number|vector|matrix x, function f)`

Applies the specified function to the elements of a vector or matrix, formally
`x <- f(x)`, or returns the function value of a number, formally `f(x)`.


#### `linear.gemv (matrix A, vector x, vector y [, alpha [, beta [, transpose A]]])`

Performs a matrix-vector product and addition operation, formally
`y <- alpha A x + beta y`. The arguments alpha and beta default to `1.0` and
`0.0` respectively. The argument transpose is one of `notrans`, `trans`,
and defaults to `notrans`. If set to `trans`, the operation is performed on
`A^T` instead of `A`.


#### `linear.ger (vector x, vector y, matrix A [, alpha])`

Performs a vector-vector product and addition operation, formally
`A <- alpha x y^T + A`. The argument alpha defaults to `1.0`.


#### `linear.gemm (matrix A, matrix B, matrix C [, alpha [, beta [, transpose A [, transpose B]]]])`

Performs a matrix-matrix product and addition operation, formally
`C <- alpha A B + beta C`. The arguments alpha and beta default to `1.0` and
`0.0` respectively. The transpose arguments are one of `notrans`, `trans`, and
default to `notrans`. If set to `trans`, the operations is performed on `A^T`
and/or `B^T` respectively.


#### `linear.gesv (matrix A, matrix B)`

Solves systems of linear equations, formally `A X = B`. On input, each column
of B represents the right-hand sides of a system. On output, the solutions X
are stored in B.


#### `linear.gels (matrix A, matrix B [, transpose A])`

Solves overdetermined or underdetermined systems of linear equations, formally
`A X = B`. On input, each column of B represents the right-hand sides of a
system. On output, the solutions X are stored in B. In case of overdetermined
systems, the function solves the least squares problems `min ||b - A x||_2`,
and in case of underdetermined systems, the function finds minimum L2 norm
solutions. The argument transpose is one of `notrans`, `trans`, and defaults
to `notrans`. If set to `trans`, the operation is performed on `A^T`.


#### `linear.inv (matrix A)`

Inverts a matrix, formally `A <- A^-1`. Returns 0 if the calculation was
successful, and a positive value if the matrix is singular and cannot be
inverted. In this case, the elements of the matrix are undefined when the
function returns.


#### `linear.det (matrix A)`

Returns the determinant of a matrix, formally `det(A)`.


## Limitations

Lua Linear supports Lua 5.2.

Lua Linear has been built and tested on Ubuntu Linux (64-bit).

Lua Linear uses the BLAS and LAPACK implementations on the system.

Lua Linear uses the OpenMP implementation on the system.


## License

Lua Linear is released under the MIT license. See LICENSE for license terms.
