# Program Functions

Program functions perform specific tasks, as described for each function.


## `linear.dot (x, y)`

Returns the dot product of two vectors, formally $x^T y$.


## `linear.ger (x, y, A [, alpha])`

Performs a vector-vector product and addition operation, formally $A \leftarrow \alpha x y^T + A$.
The argument `alpha` defaults to `1.0`.


## `linear.gemv (A, x, y [transpose, [, alpha [, beta]]])`

Performs a matrix-vector product and addition operation, formally $y \leftarrow \alpha A x
+ \beta y$. The argument transpose is one of `notrans`, `trans`, and defaults to `notrans`. If
set to `trans`, the operation is performed on $A^T$. The arguments `alpha` and `beta` default to
`1.0` and `0.0`, respectively. 


## `linear.gemm (A, B, C [, transposeA [, transposeB [, alpha [, beta]]]])`

Performs a matrix-matrix product and addition operation, formally $C \leftarrow \alpha A B
+ \beta C$. The transpose arguments are one of `notrans`, `trans`, and default to `notrans`. If
set to `trans`, the operations is performed on $A^T$ and $B^T$, respectively. The arguments
`alpha` and `beta` default to `1.0` and `0.0`, respectively. 


## `linear.gesv (A, B)`

Solves systems of linear equations, formally $A X = B$. On input, each column of $B$ represents
the right-hand sides of a system. On output, the solutions $X$ are stored in $B$.


## `linear.gels (A, B [, transpose])`

Solves overdetermined or underdetermined systems of linear equations, formally $A X = B$. On input,
each column of $B$ represents the right-hand sides of a system. On output, the solutions $X$ are
stored in $B$. In case of overdetermined systems, the function solves the least squares problems
$min \lVert b - A x \rVert_2$, and in case of underdetermined systems, the function finds minimum
L2 norm solutions. The argument transpose is one of `notrans`, `trans`, and defaults to `notrans`.
If set to `trans`, the operation is performed on $A^T$.


## `linear.inv (A)`

Inverts a matrix in place, formally $A \leftarrow A^{-1}$. Returns 0 if the calculation was
successful, and a positive value if the matrix is singular and cannot be inverted. In this case,
the elements of the matrix are undefined when the function returns.


## `linear.det (A)`

Returns the determinant of a matrix, formally $\det A$.


## `linear.cov (A, B [, ddof])`

Caclulates the pairwise covariances of the column vectors of $A$ with the specified delta
degrees of freedom, `ddof`, and places the covariances into B, formally $B_{i, j}\leftarrow
\frac{\sum_{x=1}^N (A_{x,i} - \bar{A_i}) (A_{x,j} - \bar{A_j})}{N - \textrm{ddof}}$
where $\bar{A_i}$ is the mean value of the $i$-th column vector of $A$, and $N$ is the length
of the column vectors. The non-negative argument `ddof` defaults to `0` and must be less than $N$.


## `linear.corr (A, B)`

Caclulates the pairwise Pearson correlations of the column vectors of $A$ and places them
into $B$, formally $B_{i, j} \leftarrow \frac{\sum_{x=1}^N (A_{x,i} - \bar{A_i})
(A_{x,j} - \bar{A_j})}{\sqrt{\sum_{x=1}^N (A_{x,i} - \bar{A_i})^2 \times \sum_{x=1}^N (A_{x,j}
- \bar{A_j})^2}}$ where $\bar{A_i}$ is the mean value of the $i$-th column vector of $A$, and $N$
is the length of the column vectors.
