module IndependentComponentAnalysis

using Statistics, LinearAlgebra

using LoopVectorization
using SLEEFPirates
import MultivariateStats
using MultivariateStats: _invsqrtm!, fullmean, ICA, preprocess_mean, centralize, extract_kv, fit, indim, outdim, transform, mean, ConvergenceException



export fit, Tanh, Gaus, ica!, ICA, FastICA, indim, outdim, transform, mean


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
    _s = zero(T)
    a = T(f.a)
    @inbounds for j = 1:k
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
    _s = zero(T)
    a = f.a
    @inbounds for j = 1:k
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
    _s = zero(T)
    @inbounds for j = 1:k
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
function ica!(::FastICA, W::AbstractMatrix{T},      # initialized component matrix, size (m, k)
                  X::AbstractMatrix{T},      # (whitened) observation sample matrix, size(m, n)
                  fun::ICAGDeriv,         # approximate neg-entropy functor
                  maxiter::Int,           # maximum number of iterations
                  tol::Real) where T      # convergence tolerance

    # argument checking
    m = size(W, 1)
    k = size(W, 2)
    size(X, 1) == m || throw(DimensionMismatch("Sizes of W and X mismatch."))
    n = size(X, 2)
    k <= min(m, n) || throw(DimensionMismatch("k must not exceed min(m, n)."))

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
    chg = T(NaN)
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
        chg = maximum(abs.(abs.(diag(W*Wp')) .- one(T)))
        converged = (chg < tol)

        @debug "Iteration $t" change=chg tolerance=tol
    end
    converged || @error("Did not converge", maxiter, chg, oftype(chg, tol))
    return W
end

#### interface function

function MultivariateStats.fit(::Type{ICA}, X::AbstractMatrix{T},        # sample matrix, size (m, n)
                          k::Int,                                        # number of independent components
                          alg::AbstractICAAlg = FastICA();               # choice of algorithm
                          fun::ICAGDeriv = Tanh(one(T)),                 # approx neg-entropy functor
                          do_whiten::Bool = true,                        # whether to perform pre-whitening
                          maxiter::Integer = 100,                        # maximum number of iterations
                          tol::Real = 1.0e-6,                            # convergence tolerance
                          mean = nothing,                                # pre-computed mean
                          winit = nothing) where T<:Real # init guess of W, size (m, k)

    # check input arguments
    m, n = size(X)
    n > 1          || error("There must be more than one samples, i.e. n > 1.")
    k <= min(m, n) || error("k must not exceed min(m, n).")
    maxiter > 1    || error("maxiter must be greater than 1.")
    tol > 0        || error("tol must be positive.")

    # preprocess data
    mv = preprocess_mean(X, mean)
    Z::Matrix{T} = centralize(X, mv)

    W0 = zeros(T, 0, 0)  # whitening matrix
    if do_whiten
        C = rmul!(Z * transpose(Z), 1.0 / (n - 1))
        Efac = eigen(C)
        ord = sortperm(Efac.values; rev=true)
        (v, P) = extract_kv(Efac, ord, k)
        W0 = rmul!(P, Diagonal(1 ./ sqrt.(v)))
        Z = W0'Z
    end

    # initialize
    W = winit === nothing ? randn(T, size(Z,1), k) : copy(winit)

    # invoke core algorithm
    ica!(alg, W, Z, fun, maxiter, tol)

    # construct model
    if do_whiten
        W = W0 * W
    end
    return ICA(mv, W)
end



export duet, istft, mixture
include("duet.jl")

end
