# Unary Vector Functions

Unary vector functions operate on a vector. They can be called with a vector, or a matrix and
a vector.

If called with a vector `x`, the result of applying the function to that vector is returned.

If called with a matrix `X` and vector `y`, the results of applying the function to the vectors of
matrix `X` are assigned to vector `y`. If `order` is `row` (the default), the function is applied
to the row vectors of matrix `X`, and the length of vector `y` must match the number of rows of
the matrix; if `order` is `col`, the function is applied to the column vectors of matrix `X`, and
the length of vector `y` must match the number of columns of the matrix.

> [!IMPORTANT]
> If a unary vector function is called with a matrix and a vector, the `order` argument is
> required if the function receives extra arguments, such as `ddof`; the argument can be set to
> `nil` to imply the default order.

The following function descriptions assume a call with a matrix `X` and vector `y` using `row`
order.


## `linear.nrm2 (x|X [, y [, order]])`

Applies the Euclidean norm function (also known as L2 norm), formally
$y_i \leftarrow \lVert X^i \rVert_2$.


## `linear.asum (x|X [, y [, order]])`

Applies the absolute-value norm function (also known as L1 norm), formally
$y_i \leftarrow \lVert X^i \rVert_1$.


## `linear.sum (x|X [, y [, order]])`

Applies the sum function, formally $y_i \leftarrow \sum\nolimits_{j=1}^{N} X_{ij}$ where $N$ is the
length of the vectors of matrix `X`.


## `linear.mean (x|X [, y [, order]])`

Applies the mean value function, formally $y_i \leftarrow \frac{\sum\nolimits_{j=1}^{N} X_{ij}}{N}$
where $N$ is the length of the vectors of matrix `X`.


## `linear.var (x|X [, y [, order]] [, ddof])`

Applies the variance function with the specified delta degrees of freedom, `ddof`, formally
$y_i \leftarrow \frac{\sum\nolimits_{j=1}^N (X_{ij} - \bar{X^i})^2}{N - \textrm{ddof}}$ where
$\bar{X^i}$ is the mean value of vector $X^i$, and $N$ is its length. The non-negative argument
`ddof` defaults to `0` and must be less than $N$.


## `linear.std (x|X [, y [, order]] [, ddof])`

Applies the standard deviation function with the specified delta degrees of freedom, `ddof`,
formally $y_i \leftarrow \sqrt{\frac{\sum\nolimits_{j=1}^N (X_{ij} - \bar{X^i})^2}
{N - \textrm{ddof}}}$ where $\bar{X^i}$ is the mean value of vector $X^i$, and $N$ is its length.
The non-negative argument `ddof` defaults to `0` and must be less than $N$.
