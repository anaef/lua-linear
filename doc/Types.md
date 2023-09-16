# Types

This section descibes the types of Lua Linear.


## `linear.vector`

A vector of double values. Vectors have support for the `ipairs` function, the length operator
`#`, as well as index access to get and set the elements of the vector. Indexing is 1-based.


## `linear.matrix`

A matrix of double values. Matrices have support for the `ipairs` function, the length operator
`#`, as well as index access to get the vectors of the matrix. Indexing is 1-based.

Each matrix has an order that is either row major or column major. For a row major matrix,
indexing returns row vectors; for a column major matrix, indexing returns column vectors.
