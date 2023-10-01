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


## `linear.clip (n|x|X [, min [, max]])`

Applies the clip function, formally $x_i \leftarrow \max(\textrm{min}, \min(\textrm{max}, x_i))$.
The argument `min` defaults to `0`, and the argument `max` defaults to `1`.


## `linear.uniform (n|x|X)`

Applies the uniform random function, formally $x_i \leftarrow u \sim \mathcal{U}(0, 1 - \epsilon)$.
The function returns statistically random, uniformly distributed values from the half-open
interval $[0, 1)$. You can re-seed the random state with the `linear.randomseed`
[core function](Core.md).

> [!NOTE]
> The random number generator used is optimized for statistical purposes, and is completely
> unsuitable for cryptographic use cases.


## `linear.normal (n|x|X)`

Applies the normal random function, formally $x_i \leftarrow n \sim \mathcal{N}(0, 1)$. The
function returns statistically random, normally distributed values with a mean of $0$ and a
standard deviation of $1$. You can re-seed the random state with the `linear.randomseed`
[core function](Core.md).

> [!NOTE]
> Please see the note above.


## `linear.normalpdf (n|x|X [, mu [, sigma]])`

Applies the normal probability density function with mean `mu` and standard deviation `sigma`,
formally $x_i \leftarrow \frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{1}{2}(\frac{x_i- \mu}{\sigma})^2}$.
The argument `mu` defaults to `0.0`, and the argument `sigma` defaults to `1.0`.


## `linear.normalcdf (n|x|X [, mu [, sigma]])`

Applies the normal cumulative distribution function with mean `mu` and standard deviation `sigma`,
formally $x_i \leftarrow \frac{1}{2}(1 + \mathop{\mathrm{erf}} \  (\frac{x_i - \mu}
{\sigma \sqrt{2}}))$. The argument `mu` defaults to `0.0`, and the argument `sigma` defaults to
`1.0`.


## `linear.normalqf (n|x|X [, mu [, sigma]])`

Applies the normal quantile function with mean `mu` and standard deviation `sigma`, formally
$\mu + \sigma \sqrt{2} \mathop{\mathrm{erf}}^{-1} \  (2 x_i - 1)$. The argument `mu`
defaults to `0.0`, and the argument `sigma` defaults to `1.0`. The function is the inverse of the
cumulative distribution function.
