# Binary Vector Functions

Binary vector functions operate on two vectors. They can be called with two vectors, a vector
and a matrix, or with two matrices.

If called with two vectors `x`and `y`, the function applies vector `x` to vector `y`. The
lengths of the vectors must match.

If called with a vector `x` and a matrix `Y`, the function applies vector `x` to the vectors of
matrix `Y`. If `order` is `row` (the default), vector `x` is applied to the row vectors
of matrix `Y`; if order is `col`, vector `x` is applied to the column vectors of matrix `Y`. The
lengths of the vectors must match.

> [!IMPORTANT]
> If a binary vector function is called with a vector and a matrix, the `order` argument is
> required if the function receives extra arguments, such as `alpha`; the argument can be set to
> `nil` to imply the default order.

If called with two matrices `X` and `Y`, the function applies the major order vectors of matrix
`X` to the major order vectors of matrix `Y`. The size and order of matrices `X` and `Y` must
match.

The following function descriptions assume a call with two vectors `x` and `y`.


## `linear.axpy (x|X, y|Y [, order] [, alpha])`

Scales and adds vector `x` to vector `y`, formally $y \leftarrow \alpha x + y$. The argument
`alpha` defaults to `1.0`.


## `linear.axpby (x|X, y|Y [, order] [, alpha [, beta]])`

Scales and adds vector `x` to scaled vector `y`, formally $y \leftarrow \alpha x + \beta y$. The
arguments `alpha` and `beta` default to `1.0` and `0.0`, respectively.


## `linear.mul (x|X, y|Y [, order] [, alpha])`

Multiplies the components of vector `y` with the components of vector `x` risen to power `alpha`,
formally $y \leftarrow x^\alpha y$. The argument `alpha` defaults to `1.0`.


## `linear.swap (x|X, y|Y [, order])`

Swaps the components of vectors `x` and `y`, formally $x \leftrightarrow y$.


## `linear.copy (x|X, y|Y [, order])`

Copies the components of vector `x` to vector `y`, formally $y \leftarrow x$.
