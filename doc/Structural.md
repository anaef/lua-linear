# Structural Functions

This section describes the structural functions of Lua Linear.


## `linear.vector (length)`

Creates a new vector of the specified length.


## `linear.matrix (rows, cols [, order])`

Creates a new matrix of the specified size. Order is one of `row`, `col`, and defaults to
creating a matrix with row major order.


## `linear.type (x|X)`

Returns the string `"vector"` if the value is a vector, `"matrix"` if the value is a matrix, or
`nil` otherwise.


## `linear.size (x|X)`

Returns the length of vector `x`, or three values for the number of rows and columns as well as
the order of a matrix `X`.


## `linear.tvector (X, index)`

Returns a transposed (or minor order) vector referencing the underlying matrix `X`. This is a
column vector in case of a row major matrix, and a row vector in case of a column major matrix.


## `linear.sub (x [, start [, end]])`

Returns a sub vector referencing the underlying vector `x`. Start and end are inclusive. The
argument `start` defaults to `1`, and the argument `end` defaults to the length of `x`.


## `linear.sub (X [, rowstart [, colstart [, rowend [, colend]]]])`

Returns a sub matrix referencing the underlying matrix `X`. All bounding values are inclusive.
Start values default to 1, and end values default to the number of rows or columns of the matrix,
respectively.


## `linear.unwind (X1 {, Xi}, x)`

Unwinds one or more matrices `X1`, ..., `Xn` into a vector `x`. The number of elements of the
matrices and the vector must match.


## `linear.reshape (x, X1 {, Xi})`

Reshapes a vector `x` into one or more matrices `X1`, ..., `Xn`. The number of elements of the
vector and the matrices must match.


## `linear.totable (x|X)`

Returns the values of vector `x` or matrix `X` as a table.

For a vector, the function returns a list of numbers.

For a matrix, the function returns a list of lists of numbers. The nested lists of numbers are
the major order vectors of the matrix.


## `linear.tolinear (t [, order])`

Returns a vector or matrix based on the contents of table `t`.

If `t` is a list of numbers, the function returns a vector with the numbers as its values.

If `t` is a list of lists of numbers, the function returns a matrix. The nested lists of
numbers are the major order vectors of the matrix. The parameter `order` can be one of `row`
(default) or `col`, and controls whether a row major or a column major matrix is returned.
