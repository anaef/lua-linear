# Lua Linear Release Notes


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
