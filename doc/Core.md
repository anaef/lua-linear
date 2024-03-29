# Core Functions

This section describes the core functions of Lua Linear.


## `linear.vector (length)`

Creates a new vector of the specified length. The components of the vector are initialized to 0.


## `linear.matrix (rows, cols [, order])`

Creates a new matrix of the specified size. Order is one of `"row"`, `"col"`, and defaults to
creating a matrix with row major order. The elements of the matrix are initialized to 0.


## `linear.totable (x|X)`

Returns the values of vector `x` or matrix `X` as a table.

For a vector, the function returns a list of numbers.

For a matrix, the function returns a list of lists of numbers. The nested lists of numbers
correspond to the major order vectors of the matrix.


## `linear.tolinear (t [, order])`

Returns a vector or matrix based on the contents of table `t`.

If `t` is a list of numbers, the function returns a vector with the numbers as its components.

If `t` is a list of lists of numbers, the function returns a matrix. The nested lists of
numbers correspond to the major order vectors of the matrix. The argument `order` can take the
value `"row"` (the default) or `"col"`, and controls whether a row major or a column major matrix
is returned. All nested lists of numbers must have the same length.


## `linear.tovector (list, selector)`

Returns a vector with components selected from the specified list of values. If the selector is a
string, the function indexes each value with the string as the key; if the selector is a function,
the function is called with each value as the sole argument. The result of the indexing operation
or of the function call must be a number or `nil`. If it is a number, the number is appended to
the vector; if it is `nil`, the result is ignored. The function generates an error if a result is
neither a number nor `nil`, or if the resulting vector would be empty.


## `linear.type (x|X)`

Returns the string `"vector"` if the value is a vector, `"matrix"` if the value is a matrix, or
`nil` otherwise.


## `linear.size (x|X)`

Returns the length of vector `x`, or three values for the number of rows and columns as well as
the order of matrix `X`.


## `linear.tvector (X, index)`

Returns a transposed (or minor order) vector referencing the underlying matrix `X`. This is a
column vector in case of a row major matrix, and a row vector in case of a column major matrix.


## `linear.sub (x [, start [, end]])`

Returns a sub vector referencing the underlying vector `x`. Start and end are inclusive. The
argument `start` defaults to `1`, and the argument `end` defaults to the length of vector `x`.


## `linear.sub (X [, rowstart [, colstart [, rowend [, colend]]]])`

Returns a sub matrix referencing the underlying matrix `X`. All bounding values are inclusive.
Start values default to 1, and end values default to the number of rows or columns of the matrix,
respectively.


## `linear.unwind (X1 {, Xi}, x)`

Unwinds one or more matrices `X1`, ..., `Xn` into a vector `x`. The number of elements of the
matrices must match the length of the vector. The function serializes the major order vectors of
the matrices into the vector.


## `linear.reshape (x, X1 {, Xi})`

Reshapes a vector `x` into one or more matrices `X1`, ..., `Xn`. The length of the vector must
match the number of elements of the matrices. The function deserializes the major order vectors of
the matrices from the vector.


## `linear.randomseed (seed)`

Re-seeds the random state. The argument `seed` must be an integer.


## `linear.ipairs (x|X)`

Enables ipairs-like iteration over vector `x` or matrix `X`.

> [!NOTE]
> The function is only provided for Lua 5.1. As of Lua 5.2, you can use the regular `ipairs`
> function to iterate over vectors and matrices.
