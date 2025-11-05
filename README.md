## Attribution

This Python implementation of the Cramér two-sample test is based on the R package authored and maintained by Carsten Franz <carsten.franz@gmail.com>. All descriptive text below is reproduced from the original documentation provided with that package.

## Cramer-Test for uni- and multivariate two-sample-problem

Provides Python routine for the so called two-sample Cramer-Test. This nonparametric two-sample-test on equality of the underlying distributions can be applied to multivariate data as well as univariate data. It offers two possibilities to approximate the critical value both of which are included in this package.

## Perform Cramer-Test for uni- and multivariate two-sample-problem

Perform Cramér-test for two-sample-problem. 

Both univariate and multivariate data is possible. For calculation of the critical value Monte-Carlo-bootstrap-methods and eigenvalue-methods are available. For the bootstrap access ordinary and permutation methods can be chosen as well as the number of bootstrap-replicates taken.

### Arguments

- **x**: First set of observations. Either in vector form (univariate) or in a matrix with one observation per row (multivariate).
- **y**: Second set of observations. Same dimension as `x`.
- **conf_level**: Confidence level of test. The default is `conf_level=0.95`.
- **replicates**: Number of bootstrap-replicates taken to obtain critical value. The default is `replicates=1000`. When using the eigenvalue method, this variable is unused.
- **sim**: Type of Monte-Carlo-bootstrap method or eigenvalue method. Possible values are `"ordinary"` (default) for normal Monte-Carlo-bootstrap, `"permutation"` for a permutation Monte-Carlo-bootstrap or `"eigenvalue"` for bootstrapping the limit distribution, evaluating the (approximate) eigenvalues being the weights of the limiting chisquared-distribution and using the critical value of this approximation (calculated via fast fourier transform). This method is especially good if the dataset is too large to perform Monte-Carlo-bootstrapping (although it must not be too large so the matrix eigenvalue problem can still be solved).
- **just_statistic**: Boolean variable. If `True` just the value of the Cramér-statistic is calculated and no bootstrap-replicates are produced.
- **kernel**: Character-string giving the name of the kernel function. The default is `"phiCramer"` which is the Cramér-test included in earlier versions of this package and which is used in the paper of Baringhaus and the author mentioned below. It is possible to use user-defined kernel functions here. The functions needs to be able to deal with matrix arguments. Kernel functions need to be defined on the positive real line with value 0 at 0 and have a non-constant completely monotone first derivative. An example is show in the Examples section below. Build-in functions are `"phiCramer"`, `"phiBahr"`, `"phiLog"`, `"phiFracA"` and `"phiFracB"`.
- **max_m**: Gives the maximum number of points used for the fast fourier transform. When using Monte-Carlo-bootstrap methods, this variable is unused.
- **K**: Gives the upper value up to which the integral for the calculation of the distribution function out of the characteristic function (Gurlands formula) is evaluated. The default ist 160. Careful: When increasing `K` it is necessary to increase `max_m` as well since the resolution of the points where the distribution function is calculated is `2π/K`. Thus, if just `K` is increased the maximum value, where the distribution function is calculated is lower. When using Monte-Carlo-bootstrap methods, this variable is unused.

- **random_state**: Optional seed or random generator for reproducibility. When using Monte-Carlo-bootstrap methods, this controls the random draws unless explicit resamples are supplied.

- **resamples**: Optional index matrix providing explicit bootstrap or permutation resamples. When supplied, `replicates` and `random_state` are ignored.

### Value

The returned value is an object of class `"cramertest"`, containing the following components:

- **method**: Describing the test in words.
- **d**: Dimension of the observations.
- **m**: Number of `x` observations.
- **n**: Number of `y` observations.
- **statistic**: Value of the Cramér-statistic for the given observations.
- **conf_level**: Confidence level for the test.
- **crit_value**: Critical value calculated by bootstrap method, eigenvalue method, respectively. When using the eigenvalue method, the distribution under the hypothesis will be interpolated linearly.
- **p.value**: Estimated p-value of the test.
- **result**: Contains `1` if the hypothesis of equal distributions should not be accepted and `0` otherwise.
- **sim**: Method used for obtaining the critical value.
- **replicates**: Number of bootstrap-replicates taken.
- **ev**: Contains eigenvalues and eigenfunctions when using the eigenvalue-method to obtain the critical value.
- **hypdist**: Contains the via fft reconstructed distribution function under the hypothesis. `$x` contains the x-values and `$Fx` the values of the distribution function at the positions.

### Details

The Cramér-statistic is given by

$$
T_{m,n} = \frac{mn}{m+n}\biggl(\frac{2}{mn}\sum_{i,j}^{m,n}\phi(\|\vec{X}_i-\vec{Y}_j\|^2)-\frac{1}{m^2}\sum_{i,j=1}^m\phi(\|\vec{X}_{i}-\vec{X}_{j}\|^2)-\frac{1}{n^2}\sum_{i,j=1}^n\phi(\|\vec{Y}_{i}-\vec{Y}_{j}\|^2)\biggr).
$$

The function $\\phi$ is the kernel function mentioned in the Parameters section. The proof that the Monte-Carlo-Bootstrap and eigenvalue methods work is given in the reference given below. Other build-in kernel functions are $\\phi\_{Cramer}(z)=\\sqrt{z}/2$ (recommended for location alternatives), $\\phi\_{Bahr}(z)=1-\\exp(-z/2)$ (recommended for dispersion as well as location alternatives), $\\phi\_{log}(z)=\\log(1+z)$ (preferrably for location alternatives), $\\phi\_{FracA}(z)=1-1/(1+z)$ (preferrably for dispersion alternatives) and $\\phi\_{FracB}(z)=1-1/(1+z)^2$ (also for dispersion alternatives). Test performance was investigated in the below referenced 2010 publication. The idea of using this statistic is due to L. Baringhaus, University of Hanover.

## References

- Baringhaus, L. and Franz, C. (2004) *On a new multivariate two-sample test*, Journal of Multivariate Analysis, 88, p. 190-206.
- Baringhaus, L. and Franz, C. (2010) *Rigid motion invariant two-sample tests*, Statistica Sinica 20, 1333-1361.
- Bahr, R. (1996) *Ein neuer Test fuer das mehrdimensionale Zwei-Stichproben-Problem bei allgemeiner Alternative*, German, Ph.D. thesis, University of Hanover.

## Examples

```
# comparison of two univariate normal distributions
import numpy as np
from pycramer import cramer_test, phi_bahr

rng = np.random.default_rng()
x = rng.normal(0, 1, size=20)
y = rng.normal(0.5, 1, size=50)
cramer_test(x, y)

# comparison of two multivariate normal distributions with permutation test:
from numpy.random import default_rng

rng = default_rng()
x = rng.multivariate_normal(mean=(0, 0), cov=np.diag((1, 1)), size=20)
y = rng.multivariate_normal(mean=(0.3, 0), cov=np.diag((1, 1)), size=50)
cramer_test(x, y, sim="permutation")

# comparison of two univariate normal distributions with Bahrs Kernel
rng = default_rng()
x = rng.normal(0, 1, size=20)
y = rng.normal(0, 2, size=50)
cramer_test(x, y, sim="eigenvalue", kernel=phi_bahr)
```
