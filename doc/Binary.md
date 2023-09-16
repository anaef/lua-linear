# Binary Vector Functions

Binary vector functions operate on two vectors. They can be called with two vectors, a vector
and a matrix, or with two matrices.

If called with two vectors `x`and `y`, the function applies vector `x` to vector `y`. The
lengths of the vectors must match.

If called with a vector `x` and a matrix `Y`, the function applies vector `x` to the row or column
vectors of matrix `Y`. If `order` is `row` (default), vector `x` is applied to the row vectors of
matrix `Y`; if order is `col`, vector `x` is applied to the column vectors of matrix `Y`. The
lengths of the vectors must match. The `order` argument is required if the function takes extra
arguments; it can be set to `nil`.

If called with two matrices `X` and `Y`, the function applies the major order vectors of matrix
`X` to the major order vectors of matrix `Y`. The size and order of matrices `X` and `Y` must
match.


## `linear.swap (x|X, y|Y [, order])`

Swaps the elements of vectors `x` and `y`, formally $x \leftrightarrow y$.


## `linear.copy (x|X, y|Y [, order])`

Copies the elements of vector `x` to vector `y`, formally $y \leftarrow x$.


## `linear.axpy (x|X, y|Y [, order] [, alpha])`

Scales and adds the elements of vector `x` to vector `y`. formally $y \leftarrow \alpha x + y$.
The argument `alpha` defaults to `1.0`.


## `linear.axpby (x|X, y|Y [, order] [, alpha [, beta]])`

Scales and adds the elements of vector `x` to scaled vector `y`, formally $y \leftarrow \alpha x
+ \beta y$. The arguments `alpha` and `beta` default to `1.0` and `0.0`, respectively.


## `linear.mul (x|X, y|Y [, order] [, alpha])`

Multiplies the elements of vector `y` with the elments of vector `x` risen to power `alpha`,
formally $y \leftarrow x^\alpha y$. The argument `alpha` defaults to `1.0`.
