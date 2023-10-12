# Lua Linear

## Introduction

Lua Linear provides comprehensive linear algebra and statistics support for the Lua programming
language. Where applicable, the BLAS and LAPACK implementations on the system are used.

Here are some quick examples.


### Dot Product

```lua
local linear = require("linear")

-- Calculate an inner product
local x = linear.tolinear({ 1, 2, 3 })
local y = linear.tolinear({ 3, 2, 1 })
print("x^T y", linear.dot(x, y))  -- should be 10
```

```
x^T y   10.0
```


### Normal Random Numbers

```lua
-- Generate and analyze normally distributed random numbers with mean 10, std 5, i.e., N(10,5)
local x = linear.vector(100)
linear.normal(x)   -- N(0,1)
linear.scal(x, 5)  -- N(0,5)
linear.inc(x, 10)  -- N(10,5)
print("N(10,5)", x[1], x[2], x[3], "...")
print("mean, std", linear.mean(x), linear.std(x, 1))  -- should approximate 10, 5
```

```
N(10,5) 7.6755368439453 11.817516987964 11.047801936107 ...
mean, std       9.6617642376046 4.8338730760376
```


### System of Linear Equations

```lua
-- Solve a system of linear equations: x + 2y = 7, 2x - y = 9
local A = linear.tolinear({ { 1, 2 }, { 2, -1 } })
local B = linear.matrix(2, 1)
local b = linear.tvector(B, 1)  -- column vector
b[1], b[2] = 7, 9
linear.gesv(A, B)
print("solutions", b[1], b[2])  -- should be 5, 1
```

```
solutions       5.0     1.0
```


### Pearson Correlation

```lua
-- Calculate pairwise Pearson correlation coefficients of 10k uniform random numbers
local A = linear.matrix(10000, 3)
linear.uniform(A)
local C = linear.matrix(3, 3)
linear.corr(A, C)
print("correlations", C[1][2], C[1][3], C[2][3])  -- should be close to zero
```

```
correlations    -0.0026144290886127     0.0063412466674457      -0.004072920697946
```

### Order Statistics

```lua
-- Approximate the normal distribution through percentiles, and compare with its CDF
local x = linear.vector(100000)
linear.normal(x)
local ranks = linear.ranks(100, "zq")  -- percentile ranks: [0.00, 0.01, ..., 0.99, 1.00]
local percentiles = linear.tolinear(linear.quantile(x, ranks))  -- sample percentiles
for i = 1, 3 do
	local rank = linear.rank(percentiles, x[i])  -- approximates the normal CDF of x[i]
	local cdf = linear.normalcdf(x[i])           -- actual CDF of x[i]
	print("cdf", x[i], cdf, rank - cdf)          -- difference should be close to 0
end
```

```
cdf     -0.69208067347055       0.24444333394129        -0.00072371335583435
cdf     0.41089310817189        0.65942454191657        -0.00053147240312124
cdf     -0.95943429761886       0.16867000192409        -0.00044993796354462
```


### Splines

```lua
-- Make a cubic spline interpolant of the Runge function, and analyze its mean absolute error
local function runge (x)
	return 1 / (1 + 25 * x^2)
end
local x, y = { }, { }
for i = -5, 5 do
	table.insert(x, i / 5)
	table.insert(y, runge(i / 5))
end
x, y = linear.tolinear(x), linear.tolinear(y)
local sp = linear.spline(x, y, "not-a-knot")
local d = { }
for i = -100, 100 do
	table.insert(d, math.abs(sp(i / 100) - runge(i / 100)))
end
print("mae", "not-a-knot", linear.mean(linear.tolinear(d)))  -- should be close to 0

-- As we know the derivative, a clamped spline should be even better
local function rungep (x)
	return (-50 * x) / (1 + 25 * x^2)^2
end
sp = linear.spline(x, y, "clamped", nil, rungep(-1), rungep(1))
d = { }
for i = -100, 100 do
	table.insert(d, math.abs(sp(i / 100) - runge(i / 100)))
end
print("mae", "clamped", linear.mean(linear.tolinear(d)))  -- should be closer to 0
```

```
mae     not-a-knot      0.0042786643621508
mae     clamped 0.004061824568681
```


## Build, Test, and Install

### Dependencies

You may need to install the following packages to build Lua Linear:

* libopenblas-dev
* liblapacke-dev


### Building and Installing with LuaRocks

To build and install with LuaRocks, run:

```
luarocks install lua-linear
```


### Building, Testing and Installing with Make

Lua Linear comes with a simple Makefile. Please adapt the Makefile to your environment, and then
run:

```
make
make test
make install
```


## Release Notes

Please see the [release notes](NEWS.md) document.


## Documentation

Please browse the [documentation](doc/) folder for the extensive documentation.


## Limitations

Lua Linear supports Lua 5.1, Lua 5.2, Lua 5.3, and Lua 5.4.

Lua Linear has been built and tested on Ubuntu Linux (64-bit).

Lua Linear uses the BLAS and LAPACK implementations on the system.


## License

Lua Linear is released under the MIT license. See LICENSE for license terms.
