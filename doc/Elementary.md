# Elementary Functions

Elementary functions operate on a single value. They can be called with a number, a vector,
or a matrix.

If called with a number `n`, the result of applying the function to that number is returned.

If called with a vector `x` or a matrix `X`, the function is applied in-place to the current
values of `x` or `X`.


## `linear.sgn (n|x|X)`

Applies the sign function, formally $\mathop{\mathrm{sgn}} \ x$. The result of the sign function
is `1` if `x` is larger than zero, `-1` if `x` is less than zero, and `x` otherwise.


## `linear.abs (n|x|X)`

Applies the absolute value function, formally $\| x \|$.


## `linear.exp (n|x|X)`

Applies the exponential function, formally $e^x$.


## `linear.log (n|x|X)`

Applies the natural logarithm function, formally $\log x$.


## `linear.logistic (n|x|X)`

Applies the logistic function, formally $\frac{1}{1 + e^{-x}}$.


## `linear.tanh (n|x|X)`

Applies the hyperbolic tangent function, formally $\tanh x$.


## `linear.apply (n|x|X, f)`

Applies function `f`, formally $f(x)$.


## `linear.set (n|x|X [, alpha])`

Applies the set function, formally $\alpha$. The argument `alpha` defaults to `1.0`. The
function returns the constant value `alpha`.


## `linear.uniform (n|x|X)`

Applies the uniform random function, formally $u \in \mathcal{U}(0, 1 - \epsilon)$. The
function returns statistically random, uniformly distributed values from the interval $[0, 1)$.


## `linear.normal (n|x|X)`

Applies the normal random function, formally $n \in \mathcal{N}(0, 1)$. The function returns
statistically random, normally distributed values with a mean of $0$ and a standard deviation
of $1$.


## `linear.inc (n|x|X [, alpha])`

Applies the increment function, formally $x + \alpha$. The argument `alpha` defaults to `1.0`.


## `linear.scal (n|x|X [, alpha])`

Applies the scale function, formally $\alpha x$. The argument `alpha` defaults to `1.0`.


## `linear.pow (n|x|X, [, alpha])`

Applies the power function, formally $x^\alpha$. The argument `alpha` defaults to `1.0`.
