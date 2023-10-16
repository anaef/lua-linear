# Types

This section describes the types of Lua Linear.


## `linear.vector`

A vector of double values, representing its *components*. Vectors have support for the `ipairs`
function, the length operator `#`, as well as index access to get and set the components of the
vector. Indexing is 1-based.

The length of a vector, $N$, must satisfy $1 \le N \le \mathrm{INT\\_MAX}$ where
$\mathrm{INT\\_MAX}$ is $2^{31} - 1$ on many systems.


## `linear.matrix`

A matrix of double values, representing its *elements*. Matrices have support for the `ipairs`
function, the length operator `#`, as well as index access to get the major vectors of the matrix.
Indexing is 1-based.

Each matrix has an order that is either row major or column major. For a row major matrix,
indexing returns row vectors; for a column major matrix, indexing returns column vectors.

The row and column dimensions of a matrix must each individually satisfy the requirement given
for the length of a vector above.
