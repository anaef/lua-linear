# Elementary Functions

Elementary functions operate on a single value. They can be called with a number, a vector,
or a matrix.

If called with a number `n`, the result of applying the function to that number is returned.

If called with a vector `x`, the function is applied in-place to the current components of the
vector.

If called with a matrix `X`, the function is applied in-place to the current elements of the
matrix.

The following function descriptions assume a call with a vector `x`.


## `linear.inc (n|x|X [, alpha])`

Applies the increment function, formally $x_i \leftarrow x_i + \alpha$. The argument `alpha`
defaults to `1.0`.


## `linear.scal (n|x|X [, alpha])`

Applies the scalar multiplication function, formally $x_i \leftarrow \alpha x_i$. The argument
`alpha` defaults to `1.0`.


## `linear.pow (n|x|X, [, alpha])`

Applies the power function, formally $x_i \leftarrow {x_i}^\alpha$. The argument `alpha` defaults
to `1.0`.


## `linear.exp (n|x|X)`

Applies the exponential function, formally $x_i \leftarrow e^{x_i}$.


## `linear.log (n|x|X)`

Applies the natural logarithm function, formally $x_i \leftarrow \log x_i$.


## `linear.sgn (n|x|X)`

Applies the sign function, formally $x_i \leftarrow \mathop{\mathrm{sgn}} \  x_i$. The result of
the sign function is $1$ if $x_i$ is larger than zero, $-1$ if $x_i$ is less than zero, and $x_i$
otherwise.


## `linear.abs (n|x|X)`

Applies the absolute value function, formally $x_i \leftarrow \| x_i \|$.


## `linear.logistic (n|x|X)`

Applies the logistic function, formally $x_i \leftarrow \frac{1}{1 + e^{-x_i}}$.


## `linear.tanh (n|x|X)`

Applies the hyperbolic tangent function, formally $x_i \leftarrow \tanh x_i$.


## `linear.apply (n|x|X, f)`

Applies function `f`, formally $x_i \leftarrow f(x_i)$.


## `linear.set (n|x|X [, alpha])`

Applies the set function, formally $x_i \leftarrow \alpha$. The argument `alpha` defaults to `1.0`.


## `linear.uniform (n|x|X)`

Applies the uniform random function, formally $x_i \leftarrow u \in \mathcal{U}(0, 1 - \epsilon)$.
The function returns statistically random, uniformly distributed values from the half-open
interval $[0, 1)$.


## `linear.normal (n|x|X)`

Applies the normal random function, formally $x_i \leftarrow n \in \mathcal{N}(0, 1)$. The
function returns statistically random, normally distributed values with a mean of $0$ and a
standard deviation of $1$.
