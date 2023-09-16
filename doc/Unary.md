# Unary Vector Functions

Unary vector functions operate on a vector. Most unary vector functions can be called with
a vector, or a matrix.

If called with a vector `x`, the result of applying the function to that vector is returned.

If called with a matrix `X` and vector `y`, the results of applying the function to the row or
column vectors of the matrix are assigned to vector `y`. If `order` is `row` (default), the
function is applied to the row vectors of `X`, and the length of `y` must match the number of rows
of `X`; if `order` is `col`, the function is applied to the column vectors of `X`, and the length
of `y` must match the number of columns of `X`.


## `linear.nrm2 (x|X [, y [, order]])`

Applies the Euclidean norm (also known as L2 norm) function, $\lVert x \rVert_2$.


## `linear.asum (x|X [, y [, order]])`

Applies the absolute-value norm (also known as L1 norm) function, formally $\lVert x \rVert_1$.


## `linear.sum (x|X [, y [, order]])`

Applies the sum function, formally $\sum_{i=1}^N x_i$ where $N$ is the length of vector $x$.


## `linear.mean (x|X [, y [, order]])`

Applies the mean value function, formally $\frac{\sum_{i=1}^N x_i}{N}$ where $N$ is the length
of vector $x$.


## `linear.var (x|X [, y, order] [, ddof])`

Applies the variance function with the specified delta degrees of freedom, `ddof`, formally
$\frac{\sum_{i=1}^N (x_i - \bar{x})^2}{N - \textrm{ddof}}$ where $\bar{x}$ is the mean value of
vector `x` and $N$ is its length. The non-negative argument `ddof` defaults to `0` and must be
less than $N$.


## `linear.std (x|X [, y, order] [, ddof])`

Applies the standard deviation function with the specified delta degrees of freedom, `ddof`,
formally $\sqrt{\frac{\sum_{i=1}^N (x_i - \bar{x})^2}{N - \textrm{ddof}}}$ where $\bar{x}$ is the
mean value of `x` and $N$ is its length. The non-negative argument `ddof` default to `0` and must
be less than $N$.


## `linear.iamax (x)`

Applies the argmax function to the absolute values of `x`, formally $\mathop{\mathrm{argmax}}_i \ 
\| x_i \|$. The function can only be called with a vector.


## `linear.iamin (x)`

Applies the argmin function to the absolute values of `x`, formally $\mathop{\mathrm{argmin}}_i \ 
\| x_i \|$. The function can only be called with a vector.
