# IndependentComponentAnalysis

[![Build Status](https://github.com/baggepinnen/IndependentComponentAnalysis.jl/workflows/CI/badge.svg)](https://github.com/baggepinnen/IndependentComponentAnalysis.jl/actions)
[![Coverage](https://codecov.io/gh/baggepinnen/IndependentComponentAnalysis.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/baggepinnen/IndependentComponentAnalysis.jl)


This package modifies the implementation of the FastICA algorithm from [MultivariateStats.jl](https://github.com/JuliaStats/MultivariateStats.jl) to make it more than 5x faster. This comes at the expense of taking an additional dependency, on [LoopVectorization.jl](https://github.com/chriselrod/LoopVectorization.jl).

This package also modifies the interface to the algorithm slightly, use it like this:

```julia
using IndependentComponentAnalysis
X = randn(4,100)
k = 2 # Number of components to extract
fit(ICA, X, k, alg = FastICA();
                fun       = Tanh(),
                do_whiten = true,
                maxiter   = 100,
                tol       = 1e-6,
                mean      = nothing,
                winit     = nothing)
```

- The options for `fun` are `Tanh(a::Real)` and `Gaus()`
