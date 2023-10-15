# Program Functions

Program functions perform a specific task, as described for each function.


## `linear.dot (x, y)`

Returns the dot product of two vectors, formally $x^T y$.


## `linear.ger (x, y, A [, alpha])`

Performs a vector-vector product and addition operation, formally $A \leftarrow \alpha x y^T + A$.
The argument `alpha` defaults to `1.0`.


## `linear.gemv (A, x, y [transpose, [, alpha [, beta]]])`

Performs a matrix-vector product and addition operation, formally
$y \leftarrow \alpha A x + \beta y$. The argument transpose can take the value `"notrans"` (the
default) or `"trans"`. If set to `"trans"`, the operation is performed on $A^T$. The arguments
`alpha` and `beta` default to `1.0` and `0.0`, respectively.


## `linear.gemm (A, B, C [, transposeA [, transposeB [, alpha [, beta]]]])`

Performs a matrix-matrix product and addition operation, formally
$C \leftarrow \alpha A B + \beta C$. The transpose arguments can take the value `"notrans"` (the
default) or `"trans"`. If set to `"trans"`, the operation is performed on $A^T$ and/or $B^T$,
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

The argument transpose can take the value `"notrans"` (the default) or `"trans"`. If set to
`"trans"`, the operation is performed on $A^T$.

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


## `linear.svd (A, U, s, VT [, ns])`

Calculates the singular value decomposition of matrix `A`, formally $A = U \Sigma V^T$ where
matrix `A` is an $m$ by $n$ matrix, i.e., it has $m$ rows and $n$ columns. The order of the
matrices must match. The columns of matrix `U` are set to the left singular vectors, the rows of
matrix `VT` are set to the right singular vectors, and vector `s` is set to the singular values of
matrix `A`, in descending order. The content of matrix `A` is destroyed. The function returns
`true` if the calculation was successful, and `false` if convergence failed.

If the argument `ns` is omitted, the function calculates all singular vectors. Matrices `U`
and `VT` must be square matrices of sizes $m$ and $n$, respectively.

If the argument `ns` is provided, it specifies the number of singular values to calculate and must
satisfy $1 \le \textrm{ns} \le \min(m, n)$. Matrix `U` must be an $m$ by $\textrm{ns}$ matrix, and
matrix `VT` must be an $\textrm{ns}$ by $n$ matrix. The function calculates the largest $ns$
singular values. If $\textrm{ns}$ is less than $\min(m, n)$, the function uses an eigenvalue
problem to compute the subset of the singular values.

Vector `s` must be of length $\min(m, n)$ regardless of whether `ns` is specified, and the vector
must not be a transposed vector.


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


## `linear.ranks (q, r [, mode])`

Sets vector `r` to the normalized ranks of the $q$-quantiles, formally $k / q$ for $0 \lt k \lt q$.
The argument `q` must be a positive integer. For example, if $q$ is $2$, the function sets the
normalized rank of the median, i.e., $[0.5]$, and if $q$ is $4$, the function sets the normalized
ranks of the first, second, and third quartile, i.e., $[0.25, 0.50, 0.75]$.

The optional argument `mode` must be a string. If it includes the letter `'z'`, the function
additionally sets the normalized rank of $0$-th quantile, i.e., $0$; if mode includes the letter
`'q'`, the function additionally sets the normalized rank of the $q$-th quantile, i.e., $1$.

The length of vector `r` must match the number of normalized ranks.


## `linear.quantile (v, r)`

Sets the components of vector `r` to their quantiles within the components of vector `v`.
On entry, each component of vector `r` specifies a normalized rank satisfying $0 \le r_i \le 1$;
on exit, each component is set to the quantile of the respective normalized rank.

The function creates a temporary, sorted copy of the values, and then uses linear interpolation
to calculate the quantiles.


## `linear.rank (v, q)`

Sets the components of vector `q` to their normalized ranks within the components of vector `v`.
On entry, each component of vector `q` specifies a value; on exit, each component is set to the
normalized rank of the respective value, satisfying $0 \le r_i \le 1$.

The function creates a temporary, sorted copy of the values, and then uses linear interpolation
to calculate the normalized ranks.


## `linear.spline (x, y [, boundary [, extrapolation [, da, db]]])`

Returns a cubic spline interpolant for the specified vectors `x` and `y`, where $y_i = f(x_i)$.
The returned function accepts a single argument from the domain of vector `x`, and returns the
interpolated value from the domain of vector `y`. The lengths of the vectors `x` and `y` must
match, and be at least `3` (or `4` in case of *not-a-knot* boundary conditions.) The components
of vector `x` must be strictly increasing.

The argument `boundary` controls the boundary conditions of the interpolant, and can take the
value `"not-a-knot"` (the default), `"clamped"`, or `"natural"`. If set to `"not-a-knot"`, the
third derivatives of the first and last pairs of polynomials are equated at their touch points,
formally $p_1'''(x_1) = p_2'''(x_1)$ and $p_{n-1}'''(x_{n-1}) = p_n'''(x_{n - 1})$; if set to
`"clamped"`, the first derivatives of the underlying function at the first and last value of
vector `x` are specified through the required arguments `da` and `db`, formally $p_1'(x_0) =
f'(x_0) = \textrm{da}$ and $p_n'(x_n) = f'(x_n) = \textrm{db}$; if set to `"natural"`, the second
derivatives of the first and last polynomial at the first and last value of vector `x` are equated
to $0$, formally, $p_1''(x_0) = p_n''(x_n) = 0$.

The argument `extrapolation` controls the extrapolation behavior of the interpolant, and can take
the value `"none"` (the default), `"const"`, `"linear"`, or `"cubic"`. If set to `"none"`, the
interpolation function generates an error when extrapolation is attempted; if set to `"const"`,
the function returns the first or last value of vector `y`, respectively; if set to `"linear"`",
the function expands the linear coefficient from the first or last polynomial, respectively; if
set to `"cubic"`, the function expands the full first or last polynomial, respectively.
