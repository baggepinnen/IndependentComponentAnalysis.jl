module IndependentComponentAnalysis

using Statistics, LinearAlgebra, Random

using LoopVectorization
using SLEEFPirates
import MultivariateStats
import MultivariateStats: reconstruct
using MultivariateStats: _invsqrtm!, fullmean, ICA, preprocess_mean, centralize, extract_kv, fit, indim, outdim, transform, mean, ConvergenceException



export fit, Tanh, Gaus, ica!, ICA, FastICA, indim, outdim, transform, mean
export ICAInfo, mixingmatrix, unmixingmatrix, reconstruct

"""
    ICAInfo

FastICA diagnostics: `converged`, `iterations`, and `residual` (the maximum
absolute deviation of successive component inner-product magnitudes from unity).
`numerical_rank` is the covariance rank used for whitening, or `nothing` when
whitening is disabled or `ica!` is called directly.
"""
struct ICAInfo{T<:Real}
    converged::Bool
    iterations::Int
    residual::T
    numerical_rank::Union{Nothing,Int}
end

"""
    unmixingmatrix(model::ICA)

Return the matrix mapping centered measurements to source components:
`unmixingmatrix(model) * (X .- mean(model))`.
"""
unmixingmatrix(model::ICA) = copy(transpose(model.W))

"""
    mixingmatrix(model::ICA)

Return the Moore–Penrose pseudoinverse of `unmixingmatrix(model)`, mapping source
components to centered measurements. For a reduced component count this gives
the minimum-norm reconstruction in the retained measurement subspace.
"""
mixingmatrix(model::ICA) = pinv(unmixingmatrix(model))

"""
    reconstruct(model::ICA, sources; components=:, add_mean=true)

Reconstruct measurements from a source vector or matrix, with one row per model
component, as returned by `transform(model, X)`. `components` selects a component
index or collection of distinct indices; unselected components contribute zero.
The model mean is added once when `add_mean=true`. An empty selection returns
the mean, or zeros when `add_mean=false`. Source scale and ordering follow the
fitted model and are not identifiable as physical source amplitudes or labels.
"""
function reconstruct(model::ICA, sources::AbstractVecOrMat{<:Real};
                     components=:, add_mean::Bool=true)
    k = size(model.W, 2)
    size(sources, 1) == k || throw(DimensionMismatch("Sources must have one row per model component."))
    inds = components isa Colon ? collect(1:k) :
           components isa Integer ? [components] : collect(components)
    all(i -> i isa Integer && 1 <= i <= k, inds) || throw(ArgumentError("Component indices must be integers between 1 and $k."))
    length(unique(inds)) == length(inds) || throw(ArgumentError("Component indices must be distinct."))
    inds = Int[inds...]
    A = mixingmatrix(model)
    selected = sources isa AbstractVector ? sources[inds] : sources[inds, :]
    X = A[:, inds] * selected
    add_mean && (X .+= mean(model))
    return X
end


# Independent Component Analysis

#### FastICA type

abstract type AbstractICAAlg end

struct FastICA <: AbstractICAAlg end



abstract type ICAGDeriv end

struct Tanh{T} <: ICAGDeriv
    a::T
end

Tanh() = Tanh{Float32}(1.0f0)

evaluate(f::Tanh{T}, x::T) where T<:Real = (a = f.a; t = tanh(a * x); (t, a * (1 - t * t)))


function update_UE!(f::Tanh, U::AbstractMatrix{T}, E1::AbstractVector{T}) where T
    n,k = size(U)
    a = T(f.a)
    @inbounds for j = 1:k
        _s = zero(T)
        @simd for i = 1:n
            t = SLEEFPirates.tanh_fast(a * U[i,j])
            U[i,j] = t
            _s += a * (1 - t^2)
        end
        E1[j] = _s / n
    end
end

function update_UE!(f::Tanh{T}, U::AbstractMatrix{CT}, E1::AbstractVector) where {T, CT <: Complex}
    n,k = size(U)
    a = f.a
    @inbounds for j = 1:k
        _s = zero(T)
        for i = 1:n
            t = complex(SLEEFPirates.tanh_fast(a * real(U[i,j])), SLEEFPirates.tanh_fast(a * imag(U[i,j])))
            U[i,j] = t
            _s += a * (1 - abs2(t))
        end
        E1[j] = _s / n
    end
end

struct Gaus <: ICAGDeriv end

evaluate(f::Gaus, x::T) where T<:Real = (x2 = x * x; e = exp(-x2/2); (x * e, (1 - x2) * e))

function update_UE!(f::Gaus, U::AbstractMatrix{T}, E1::AbstractVector{T}) where T
    n,k = size(U)
    @inbounds for j = 1:k
        _s = zero(T)
        @avx for i = 1:n
            u = U[i,j]
            u2 = abs2(u)
            e = exp(-u2/2)
            U[i,j] = u * e
            _s += (1 - u2) * e
        end
        E1[j] = _s / n
    end
end


# Fast ICA
#
# Reference:
#
#   Aapo Hyvarinen and Erkki Oja
#   Independent Component Analysis: Algorithms and Applications.
#   Neural Network 13(4-5), 2000.
#
"""
    ica!(FastICA(), W, X, fun, maxiter, tol; return_info=false,
         on_nonconvergence=:log)

Update the component matrix `W` using whitened observations `X`. Return `W`,
or `(W, info::ICAInfo)` when `return_info=true`. Nonconvergence handling is
`:log` (log an error and return the final iterate), `:throw` (throw
`ConvergenceException`), or `:ignore` (return the final iterate without logging).
"""
function ica!(::FastICA, W::AbstractMatrix{T},      # initialized component matrix, size (m, k)
                  X::AbstractMatrix{T},      # (whitened) observation sample matrix, size(m, n)
                  fun::ICAGDeriv,         # approximate neg-entropy functor
                  maxiter::Int,           # maximum number of iterations
                  tol::Real; return_info::Bool=false,
                  on_nonconvergence::Symbol=:log) where T

    # argument checking
    m = size(W, 1)
    k = size(W, 2)
    size(X, 1) == m || throw(DimensionMismatch("Sizes of W and X mismatch."))
    n = size(X, 2)
    1 <= k <= min(m, n) || throw(DimensionMismatch("k must be between 1 and min(m, n)."))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive."))
    tol > 0 || throw(ArgumentError("tol must be positive."))
    on_nonconvergence in (:log, :throw, :ignore) || throw(ArgumentError("on_nonconvergence must be :log, :throw, or :ignore."))
    all(isfinite, X) || throw(ArgumentError("Observations must be finite."))
    all(isfinite, W) && rank(W) == k || throw(ArgumentError("The initial component matrix must be finite and have full column rank."))

    @debug "FastICA Algorithm" m=m n=n k=k

    # pre-allocated storage
    Wp = Matrix{T}(undef, m, k)    # to store previous version of W
    U  = Matrix{T}(undef, n, k)    # to store w'x & g(w'x)
    Y  = Matrix{T}(undef, m, k)    # to store E{x g(w'x)} for components
    E1 = Vector{T}(undef, k)       # store E{g'(w'x)} for components

    # normalize each column
    for j = 1:k
        w = view(W,:,j)
        rmul!(w, one(T) / sqrt(sum(abs2, w)))
    end

    # main loop
    chg = real(T)(NaN)
    t = 0
    converged = false
    @inbounds while !converged && t < maxiter
        t += 1
        copyto!(Wp, W)

        # apply W of previous step
        mul!(U, X', W) # u <- w'x

        # compute g(w'x) --> U and E{g'(w'x)} --> E1
        update_UE!(fun, U, E1)

        # compute E{x g(w'x)} --> Y
        rmul!(mul!(Y, X, U), one(T) / n)

        # update w: E{x g(w'x)} - E{g'(w'x)} w := y - e1 * w
        for j = 1:k
            w = view(W,:,j)
            y = view(Y,:,j)
            e1 = E1[j]
            @. w = y - e1 * w
        end

        # symmetric decorrelation: W <- W * (W'W)^{-1/2}
        copyto!(W, W * _invsqrtm!(W'W))

        # compare with Wp to evaluate a conversion change
        chg = maximum(abs.(abs.(diag(W'Wp)) .- one(T)))
        converged = (chg < tol)

        @debug "Iteration $t" change=chg tolerance=tol
    end
    if !converged
        if on_nonconvergence == :throw
            throw(ConvergenceException(maxiter, chg, oftype(chg, tol)))
        elseif on_nonconvergence == :log
            @error("Did not converge", maxiter, chg, oftype(chg, tol))
        end
    end
    info = ICAInfo(converged, t, chg, nothing)
    return return_info ? (W, info) : W
end

#### interface function

"""
    fit(ICA, X, k; alg=FastICA(), rng=Random.default_rng(), return_info=false,
        rank_rtol=sqrt(eps(eltype(X))), rank_atol=0, kwargs...)

Fit `k` independent components to measurements in the rows of `X`. Random
initialization uses `rng`; a supplied `winit` bypasses random initialization.
Return an `ICA` model, or `(model, info::ICAInfo)` when `return_info=true`.
`on_nonconvergence` accepts the policies documented for `ica!`.

With `do_whiten=true`, retain covariance eigenvalues greater than
`max(rank_atol, rank_rtol * largest_eigenvalue)`. Requesting more components
than this numerical rank throws `ArgumentError`. Both thresholds must be finite
and nonnegative. With `do_whiten=false`, data must already be whitened and
`info.numerical_rank` is `nothing`.
"""
function MultivariateStats.fit(::Type{ICA}, X::AbstractMatrix{T}, k::Int;
                              alg::AbstractICAAlg = FastICA(), kwargs...) where T<:AbstractFloat
    fit(ICA, X, k, alg; kwargs...)
end

function MultivariateStats.fit(::Type{ICA}, X::AbstractMatrix{T},        # sample matrix, size (m, n)
                          k::Int,                                        # number of independent components
                          alg::AbstractICAAlg;               # choice of algorithm
                          fun::ICAGDeriv = Tanh(one(T)),                 # approx neg-entropy functor
                          do_whiten::Bool = true,                        # whether to perform pre-whitening
                          maxiter::Integer = 100,                        # maximum number of iterations
                          tol::Real = 1.0e-6,                            # convergence tolerance
                          mean = nothing,                                # pre-computed mean
                          winit = nothing,
                          rng::AbstractRNG = Random.default_rng(),
                          return_info::Bool = false,
                          on_nonconvergence::Symbol = :log,
                          rank_rtol::Real = sqrt(eps(float(T))),
                          rank_atol::Real = zero(float(T))) where T<:Real

    # check input arguments
    m, n = size(X)
    n > 1          || error("There must be more than one samples, i.e. n > 1.")
    1 <= k <= min(m, n) || throw(ArgumentError("k must be between 1 and min(m, n)."))
    maxiter > 0    || throw(ArgumentError("maxiter must be positive."))
    tol > 0        || error("tol must be positive.")
    isfinite(rank_rtol) && rank_rtol >= 0 || throw(ArgumentError("rank_rtol must be finite and nonnegative."))
    isfinite(rank_atol) && rank_atol >= 0 || throw(ArgumentError("rank_atol must be finite and nonnegative."))
    all(isfinite, X) || throw(ArgumentError("Observations must be finite."))

    # preprocess data
    mv = preprocess_mean(X, mean)
    Z::Matrix{T} = centralize(X, mv)
    all(isfinite, Z) || throw(ArgumentError("Centered observations must be finite."))

    W0 = zeros(T, 0, 0)  # whitening matrix
    numerical_rank = nothing
    if do_whiten
        C = rmul!(Z * transpose(Z), one(T) / (n - 1))
        Efac = eigen(Symmetric(C))
        threshold = max(rank_atol, rank_rtol * maximum(Efac.values))
        numerical_rank = count(>(threshold), Efac.values)
        k <= numerical_rank || throw(ArgumentError("Requested $k components, but the covariance numerical rank is $numerical_rank at eigenvalue threshold $threshold. Reduce k or adjust rank_rtol/rank_atol."))
        ord = sortperm(Efac.values; rev=true)
        (v, P) = extract_kv(Efac, ord, k)
        W0 = rmul!(P, Diagonal(1 ./ sqrt.(v)))
        Z = W0'Z
    end

    # initialize
    W = winit === nothing ? randn(rng, T, size(Z,1), k) : Matrix{T}(winit)
    size(W) == (size(Z, 1), k) || throw(DimensionMismatch("The initial component matrix must have size ($(size(Z, 1)), $k)."))

    # invoke core algorithm
    _, iteration_info = ica!(alg, W, Z, fun, Int(maxiter), tol;
        return_info=true, on_nonconvergence=on_nonconvergence)

    # construct model
    if do_whiten
        W = W0 * W
    end
    model = ICA(mv, W)
    info = ICAInfo(iteration_info.converged, iteration_info.iterations,
        iteration_info.residual, numerical_rank)
    return return_info ? (model, info) : model
end



export duet, istft, mixture
include("duet.jl")

end
