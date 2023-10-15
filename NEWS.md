# Lua Linear Release Notes


## Release 1.3.0

- The `linear.svd` program function has been added, calculating the singular value decomposition
of matrices.

- NaN values are now handled transparently in the `linear.median`, `linear.mad`, `linear.quantile`,
and `linear.rank` functions.

- The order statistics functions `linear.ranks`, `linear.quantile`, and `linear.rank` have been
changed to work on vectors instead of lists, optimizing memory management.


## Release 1.2.0 (2023-10-12)

- The `linear.spline` program function has been added, supporting cubic spline interpolants with
the not-a-knot, clamped, and natural boundary conditions.

- Compilation errors against Lua 5.1 have been fixed, and tests have been amended with
compatibility logic.

- The rockspec has been modified to build with the make backend on Linux to use the -O3 flag
for vectorization.


## Release 1.1.0 (2023-10-01)

- The code base has been refactored into modules and made extensible.

- The vectorization of many functions has been optimized.

- The random state is now managed per Lua state, and the `linear.randomseed` function has been
added.

- The `linear.clip`, `linear.normalpdf`, `linear.normalcdf`, and `linear.normalqf` elementary
functions have been added.

- The `linear.min`, `linear.max`, `linear.skew`, `linear.kurt`, `linear.median`, and `linear.mad`
unary vector functions have been added.

- The `linear.ranks`, `linear.quantile` and `linear.rank` program functions have been added.

- Fixed the unwind and reshape functions when operating on a vector with an increment different
from one.

- The the default value of `beta` in the `linear.axpby` function has been set to `1` (changed from
`0`.)

> [!WARNING]
> If your code uses `linear.axpby` without an explicit value for `beta`, it must be updated.


## Release 1.0.0 (2023-09-20)

- Initial release.
