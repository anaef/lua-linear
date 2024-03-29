# Unary Vector Functions

Unary vector functions operate on a vector. They can be called with a vector, or a matrix and
a vector.

If called with a vector `x`, the result of applying the function to that vector is returned.

If called with a matrix `X` and vector `y`, the results of applying the function to the vectors of
matrix `X` are assigned to vector `y`. If `order` is `"row"` (the default), the function is
applied to the row vectors of matrix `X`, and the length of vector `y` must match the number of
rows of the matrix; if `order` is `"col"`, the function is applied to the column vectors of matrix
`X`, and the length of vector `y` must match the number of columns of the matrix.

> [!IMPORTANT]
> If a unary vector function is called with a matrix and a vector, the `order` argument is
> required if the function receives extra arguments, such as `ddof`; the argument can be set to
> `nil` to imply the default order.

The following function descriptions assume a call with a matrix `X` and vector `y` using `"row"`
order.


## `linear.sum (x|X [, y [, order]])`

Applies the sum function, formally $y_i \leftarrow \sum\nolimits_{j=1}^{N} X_{ij}$ where $N$ is the
length of the row vectors of matrix `X`.


## `linear.mean (x|X [, y [, order]])`

Applies the mean value function, formally $y_i \leftarrow \frac{\sum\nolimits_{j=1}^{N} X_{ij}}{N}$
where $N$ is the length of the row vectors of matrix `X`. The mean value is also known as the
first statistical moment, and it is a measure of central tendency.


## `linear.var (x|X [, y [, order]] [, ddof])`

Applies the variance function with the specified delta degrees of freedom, `ddof`, formally
$y_i \leftarrow \frac{\sum\nolimits_{j=1}^N (X_{ij} - \bar{X^i})^2}{N - \textrm{ddof}}$ where
$\bar{X^i}$ is the mean value of the $i$-th row vector of $X$, and $N$ is its length. The
non-negative argument `ddof` defaults to `0` and must be less than $N$. The variance is also known
as the second statistical moment, and it is a mesure of dispersion.


## `linear.std (x|X [, y [, order]] [, ddof])`

Applies the standard deviation function with the specified delta degrees of freedom, `ddof`,
formally $y_i \leftarrow \sqrt{\frac{\sum\nolimits_{j=1}^N (X_{ij} - \bar{X^i})^2}
{N - \textrm{ddof}}}$ where $\bar{X^i}$ is the mean value of the $i$-th row vector of $X$, and $N$
is its length. The non-negative argument `ddof` defaults to `0` and must be less than $N$. The
standard deviation is the square root of the variance.


## `linear.skew (x|X [, y [, order]] [, set])`

Applies the population or sample skewness function. The argument `set` can take the value `'p'`
(the default) or `'s'`. If set to `'p'`, the function calculates the population skewness, formally
$y_i \leftarrow \frac{\frac{1}{N} \sum\nolimits_{j=1}^N (X_{ij} - \bar{X^i})^3}
{(\frac{1}{N} \sum\nolimits_{j=1}^N (X_{ij} - \bar{X^i})^2)^\frac{3}{2}}$ where $\bar{X^i}$ is the
mean value of the $i$-th row vector of $X$, and $N$ is its length; if set to `'s'`, the function
calculates the sample skewness, formally $y_i \leftarrow \frac{\sqrt{N (N - 1)}}{N - 2} s_i$
where $s_i$ is the population skewness as defined before. The skewness is also known as the third
statistical moment, and it is a measure of asymmetry.


## `linear.kurt (x|X [, y [, order]] [, set])`

Applies the population or sample excess kurtosis function. The argument `set` can take the value
`'p'` (the default) or `'s'`. If set to `'p'`, the function calculates the population excess
kurtosis, formally $y_i \leftarrow \frac{\frac{1}{N} \sum\nolimits_{j=1}^N (X_{ij} -
\bar{X^i})^4}{(\frac{1}{N} \sum\nolimits_{j=1}^N (X_{ij} - \bar{X^i})^2)^2} - 3$ where $\bar{X^i}$
is the mean value of the $i$-th row vector of $X$, and $N$ is its length; if set to `'s'`, the
function calculates the sample excess kurtosis, formally $y_i \leftarrow \frac{N - 1}
{(N - 2) (N - 3)}((N + 1) k_i + 6)$ where $k_i$ is the population excess kurtosis as defined
before. The kurtosis is also known as the fourth statistical moment, and it is a measure
of heavy-tailedness.


## `linear.median (x|X [, y [, order]])`

Applies the median function, formally $y_i \leftarrow m(X^i)$ where $X^i$ is the $i$-th row
vector of $X$, and the function $m(x)$ creates a sorted temporary copy of $x$, and then returns
its central component if the length of $x$ is odd, or the arithmetic mean of its two central
components if the length of $x$ is even.


## `linear.mad (x|X [, y [, order]])`

Applies the median absolute deviation function, formally $y_i \leftarrow m(|X_{ij} - m(X^i)|)$
where $X^i$ is the $i$-th row vector of $X$, and the function $m(x)$ creates a sorted temporary
copy of $x$, and then returns its central component if the length of $x$ is odd, or the arithmetic
mean of its two central components if the length of $x$ is even.


## `linear.nrm2 (x|X [, y [, order]])`

Applies the Euclidean norm function (also known as L2 norm), formally
$y_i \leftarrow \lVert X^i \rVert_2$.


## `linear.asum (x|X [, y [, order]])`

Applies the absolute-value norm function (also known as L1 norm), formally
$y_i \leftarrow \lVert X^i \rVert_1$.


## `linear.min (x|X [, y [, order]])`

Applies the minimum function, formally $y_i \leftarrow \min_j X_{ij}$.


## `linear.max (x|X [, y [, order]])`

Applies the maximum function, formally $y_i \leftarrow \max_j X_{ij}$.
