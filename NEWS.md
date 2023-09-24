# Lua Linear Release Notes


## Release 1.1.0

- The code base has been refactored into modules and made extensible.

- The vectorization of many functions has been optimized.

- The random state is now managed per Lua state, and the `linear.randomseed` function has been
added.

- The `linear.min` and `linear.max` unary vector function have been added.

- Fixed the unwind and reshape functions when operating on a vector with an increment different
from one.


## Release 1.0.0 (2023-09-20)

- Initial release.
