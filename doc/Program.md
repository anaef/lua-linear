# Program Functions

Program functions perform a specific task, as described for each function.


## `linear.dot (x, y)`

Returns the dot product of two vectors, formally $x^T y$.


## `linear.ger (x, y, A [, alpha])`

Performs a vector-vector product and addition operation, formally $A \leftarrow \alpha x y^T + A$.
The argument `alpha` defaults to `1.0`.


## `linear.gemv (A, x, y [transpose, [, alpha [, beta]]])`

Performs a matrix-vector product and addition operation, formally
$y \leftarrow \alpha A x + \beta y$. The argument transpose is one of `notrans`, `trans`, and
defaults to `notrans`. If set to `trans`, the operation is performed on $A^T$. The arguments
`alpha` and `beta` default to `1.0` and `0.0`, respectively.


## `linear.gemm (A, B, C [, transposeA [, transposeB [, alpha [, beta]]]])`

Performs a matrix-matrix product and addition operation, formally
$C \leftarrow \alpha A B + \beta C$. The transpose arguments are one of `notrans`, `trans`, and
default to `notrans`. If set to `trans`, the operation is performed on $A^T$ and $B^T$,
respectively. The arguments `alpha` and `beta` default to `1.0` and `0.0`, respectively. The
order of the matrices must match.


## `linear.gesv (A, B)`

Solves systems of linear equations, formally $A X = B$. Matrix `A` must be square. The order of
the matrices must match.

On input, each column of matrix `B` represents the right-hand sides of a system. On output, the
solutions $X$ are stored in matrix `B`.

The function replaces the elements of matrix `A` with a factorization. It returns `true` if the
solutions have been computed, and `false` if the solutions could not be computed due to a zero
value in a factor, implying that matrix `A` does not have full rank.


## `linear.gels (A, B [, transpose])`

Solves overdetermined or underdetermined systems of linear equations, formally $A X = B$. Matrix
`A` is assumed to have full rank. The order of the matrices must match.

On input, each column of matrix `B` represents the right-hand sides of a system. On output, the
solutions $X$ are stored in in matrix `B`.

In case of overdetermined systems, the function solves the least squares problems
$\min \lVert b - A x \rVert_2$, and in case of underdetermined systems, the function finds minimum
L2 norm solutions.

The argument transpose is one of `notrans`, `trans`, and defaults to `notrans`. If set to `trans`,
the operation is performed on $A^T$.

The function replaces the elements of matrix `A` with a factorization. It returns `true` if the
solutions have been computed, and `false` if the solutions could not be computed due to a zero
value in a factor, implying that matrix `A` does not have full rank.


## `linear.inv (A)`

Inverts a matrix in-place, formally $A \leftarrow A^{-1}$. Matrix `A` must be square.

The function returns `true` if the calculation was successful, and `false` if the matrix could
not be inverted due to a zero value in a factor, implying that matrix `A` is singular.

> [!IMPORTANT]
> You should _not_ use the inverse of matrices to solve systems of linear equations. A numerically
> superior result is generally obtained by using a solver program function. 


## `linear.det (A)`

Returns the determinant of a matrix, formally $\det A$. Matrix `A` must be square.

The determinant is computed on a copy of the elements of matrix `A`, which remains unchanged.

The functions returns `0.0` if matrix `A` is singular.


## `linear.cov (A, B [, ddof])`

Calculates the pairwise covariances of the column vectors of $A$ with the specified delta
degrees of freedom, `ddof`, and places the covariances into B, formally $B_{ij}\leftarrow
\frac{\sum\nolimits_{x=1}^{N} (A_{xi} - \bar{A_i}) (A_{xj} - \bar{A_j})}{N - \textrm{ddof}}$
where $\bar{A_i}$ is the mean value of the $i$-th column vector of $A$, and $N$ is its length.
The non-negative argument `ddof` defaults to `0` and must be less than $N$.

> [!NOTE]
> The function is generally faster when matrix `A` is a column major matrix.


## `linear.corr (A, B)`

Calculates the pairwise Pearson product-moment correlation coefficients of the column vectors of
$A$ and places them into $B$, formally $B_{ij} \leftarrow \frac{\sum\nolimits_{x=1}^{N} (A_{xi} -
\bar{A_i}) (A_{xj} - \bar{A_j})}{\sqrt{\sum\nolimits_{x=1}^{N} (A_{xi} - \bar{A_i})^2 \times
\sum\nolimits_{x=1}^{N} (A_{xj} - \bar{A_j})^2}}$ where $\bar{A_i}$ is the mean value of the
$i$-th column vector of $A$, and $N$ is its length.

> [!NOTE]
> The function is generally faster when matrix `A` is a column major matrix.


## `linear.ranks (q [, mode])`

Returns a list of normalized ranks of the $q$-quantiles, formally $k / q$ for $0 \lt k \lt q$.
The argument `q` must be a positive integer. For example, if $q$ is $4$, the function returns
the normalized rank of the first, second, and third quartile, i.e., $[0.25, 0.50, 0.75]$.

The optional argument `mode` must be a string. If it includes the letter `'z'`, the list
additionally contains the normalized rank of $0$-th quantile, i.e., $0$; if it includes the letter
`'q'`, the list additionally contains the normalized rank of the $q$-th quantile, i.e., $1$.


## `linear.quantile (values, r)`

Returns the quantile with normalized rank `r` within the specified values. If `r` is a list of
values, the function returns a list of quantiles with the requested normalized ranks. The
normalized ranks must satisfy $0 \le r \le 1$.

The function creates a temporary, sorted copy of the values, and then uses linear interpolation
to calculate the quantiles.


## `linear.rank (values, q)`

Returns the normalized rank of value `q` within the specified values. If `q` is a list of values,
the function returns a list of the normalized ranks of the requested values. The normalized
ranks satisfy $0 \le r \le 1$.

The function creates a temporary, sorted copy of the values, and then uses linear interpolation
to calculate the normalized ranks.
