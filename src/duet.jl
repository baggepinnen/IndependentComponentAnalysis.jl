using DSP, ImageFiltering, Images, SparseArrays, FFTW, RecipesBase

import DSP.Periodograms: compute_window

function istft(S::AbstractMatrix{Complex{T}}, wlen::Int, overlap::Int; onesided, nfft=nextfastfft(wlen), window=hanning) where T
    winc = wlen-overlap
    win, norm2 = compute_window(window, wlen)
    win² = win.^2
    nframes = size(S,2)-1
    outlen = nfft + nframes*winc
    out = zeros(T, outlen)
    tmp1 = Array{eltype(S)}(undef, size(S, 1))

    if onesided
        tmp2 = zeros(T, nfft)
        p = FFTW.plan_irfft(tmp1, nfft)
    else
        tmp2 = zeros(Complex{T}, nfft)
        p = FFTW.plan_ifft(tmp1)
    end
    wsum = zeros(outlen)
    for k = 1:size(S,2)
        copyto!(tmp1, 1, S, 1+(k-1)*size(S,1), length(tmp1))
        mul!(tmp2, p, tmp1)
        # tmp2 ./= AbstractFFTs.normalization(tmp2, size(tmp2)) # QUESTION: Not sure how to make this work

        ix = (k-1)*winc
        @inbounds for n=1:nfft
            out[ix+n] += real(tmp2[n])*win[n]
            wsum[ix+n] += win²[n]
        end

    end
    mw = median(wsum)
    @inbounds for i = 1:length(wsum)
        if wsum[i] >= 1e-3*mw
            out[i] /= wsum[i]
        end
    end
    out
end


function find_extrema(A, n)
    # q = quantile(vec(A), 0.95)
    # A = copy(A)
    # A .= max.(A, q)
    inds = findlocalmaxima(A)
    isempty(inds) && error("Did not find any peaks in the delay/amplitude histogram, try modifying the `kernel_size` or any of the other parameters.")
    amps = A[inds]
    perm = sortperm(amps, rev=true)
    inds[perm[1:min(n,length(perm))]]
end

function separate(x1::AbstractArray{T},x2,αpeak::Vector{<:Real}, δpeak::Vector{<:Real}, S1, S2, freqs, nfft; kwargs...) where T
    numsources = length(αpeak)
    αpeak = @. (αpeak + sqrt(αpeak^2 + 4)) / 2

    bestsofar = fill(T(Inf), size(S1))
    bestind = zeros(Int, size(S1))
    for i = 1:numsources
        score = @. abs(αpeak[i] * cis(-freqs * δpeak[i]) * S1 - S2)^2 /
           (1 + αpeak[i]^2)
        mask = score .< bestsofar
        bestind[mask] .= i
        bestsofar[mask] .= score[mask]
    end
    masks = map(1:numsources) do i
        mask = bestind .== i
    end
    est = map(1:numsources) do i
        # mask = imfilter(masks[i], Kernel.gaussian(1))
        mask = masks[i]
        M = [
            zeros(eltype(S1), 1, size(S1, 2)) # Add in zero DC component
            @. ((S1 + αpeak[i] * cis(δpeak[i] * freqs) * S2) / (1 + αpeak[i]^2)) * mask
        ]
        istft(M, nfft, nfft÷2; kwargs...)
    end
    reduce(hcat, est), masks
end



struct DUET
    "histogram"
    A
    "ranges for amp and delay values"
    ar
    "ranges for amp and delay values"
    dr
    "peak locations"
    av
    "peak locations"
    dv
    "amp map"
    α
    "delay map"
    δ
    masks
end

@recipe function plot(h::DUET)
    xguide --> "δ"
    yguide --> "A"
    title --> "Almplitude-delay histogram"
    @series begin
        seriestype --> :heatmap
        h.dr, h.ar, h.A
    end
    @series begin
        seriestype := :scatter
        label --> "Local maxima"
        markercolor --> :red
        h.dv, h.av
    end
end

"""
    est, H = duet( x1, x2, n_sources, n = 1024;
        p            = 1, # amplitude power used to weight histogram
        q            = 0, # delay power used to weight histogram
        amax         = 0.7,
        dmax         = 3.6,
        abins        = 35, # number of histogram bins
        dbins        = 50, # number of histogram bins
        kernel_size  = 1, # Controls the smoothing of the histogram.
        window       = hanning,
        bigdelay     = false,
        kernel_sizeδ = 0.1,
        kwargs..., # These are sent to the stft function
    )

DUET is an algorithm for blind source separation. It works on stereo mixtures and can separate any number of sources as long as they do not overlap in the time-frequency domain.

## `p` and `q`
The paper gives the following guidelines:
- `p = 0, q = 0`: the counting histogram proposed in the original DUET algorithm
- `p = 1, q = 0`: motivated by the ML symmetric attenuation estimator
- `p = 1, q = 2`: motivated by the ML delay estimator
- `p = 2, q = 0`: in order to reduce delay estimator bias
- `p = 2, q = 2`: for low signal-to-noise ratio or speech mixtures
From the paper referenced below:
"`p = 1, q = 0` is a good default choice. When the sources are not
equal power, we would suggest `p = 0.5, q = 0` as it prevents the dominant
source from hiding the smaller source peaks in the histogram."

- `bigdelay` indicates whether or not the two microphones are far apart. If `true`, the delay `δ` is estimated using the differential method (see the paper sec 8.4.1) and the delay map is smoothed using `kernel_sizeδ`.

Implementation based on *Rickard, Scott. (2007). The DUET blind source separation algorithm. 10.1007/978-1-4020-6479-1_8.*
"""
function duet(
    x1,
    x2,
    n_sources,
    n     = 1024;
    p     = 1,
    q     = 0, # powers used to weight histogram
    amax  = 0.7,
    dmax  = 3.6,
    abins = 35,
    dbins = 50,
    kernel_size = 1,
    window = hanning,
    onesided = true,
    bigdelay = false,
    kernel_sizeδ = 0.1,
    kwargs...,
)

    S1 = stft(x1, n, n÷2; window = window, onesided=onesided, kwargs...)
    S2 = stft(x2, n, n÷2; window = window, onesided=onesided, kwargs...)
    S1, S2 = S1[2:end, :], S2[2:end, :] # remove dc
    if onesided
        # freqs = FFTW.rfftfreq(n)[2:end] .* (2pi)
        freqs = (1:n÷2) .* (2pi / n) # We don't need the negative freqs since we use onesided
    else
        # freqs = [(1:n÷2); -reverse((1:n÷2))] .* (2pi / n)
        freqs = FFTW.fftfreq(n)[2:end] .* (2pi)
        freqs = freqs[1:size(S1, 1)]
    end

    R21 = (S2 .+ eps()) ./ (S1 .+ eps()) # ratio of spectrograms
    α = abs.(R21) # relative attenuation
    α = @. α - 1 / α # symmetric attenuation

    if bigdelay
        Δω = 2pi / n
        Δω*dmax > π && @warn("frequency resolution not sufficient for the chosen maximum delay.")
        δ0 = @. R21 * conj(S2)/S1 / abs2(R21)
        if kernel_sizeδ > 0
            δ0 = imfilt(δ0, kernel_sizeδ) # It appears to be slightly better to filter before angle, but more tests needed
        end
        δ = @. -angle(δ0) / freqs # NOTE: I removed the Δω divisor here and it seems to have made it more stable
    else
        δ = @. -angle(R21) / freqs # δ relative delay
    end


    Sweight = @. (abs(S1) * abs(S2))^p * abs(freqs)^q # weights

    αmask = @. (abs(α) < amax) & (abs(δ) < dmax)
    α_vec = α[αmask]
    δ_vec = δ[αmask]
    Sweight = Sweight[αmask]
    # determine histogram indices
    αind = @. round(Int, 1 + (abins - 1) * (α_vec + amax) / (2amax))
    δind = @. round(Int, 1 + (dbins - 1) * (δ_vec + dmax) / (2dmax))
    # FULL-SPARSE TRICK TO CREATE 2D WEIGHTED HISTOGRAM
    A = Matrix(sparse(αind, δind, Sweight, abins, dbins))
    if kernel_size > 0
        A = imfilt(A, kernel_size)
    end
    # abins, dbins = size(A)
    ar,dr = LinRange(-amax , amax , abins ), LinRange(-dmax , dmax , dbins )
    inds = find_extrema(A, n_sources)
    av = [ar[i[1]] for i in inds]
    dv = [dr[i[2]] for i in inds]

    est, masks = separate(x1,x2, av, dv, S1, S2, freqs, n; (window=window, onesided=onesided, kwargs...)...)
    H = DUET(A,ar,dr,av,dv,α,δ,masks)

    est, H
end

function imfilt(x,kernel_size)
    K = KernelFactors.gaussian(kernel_size)
    imfilter(x, kernelfactors((K,K)))
end

function imfilt1(x,kernel_size)
    K = KernelFactors.gaussian(kernel_size)
    imfilter(x, K[:,1:1])
end




"""
    mix = mixture(signals, amps, [delays::Vector{Int}])
    mixes = mixture(signals, amps::Vector{Vector}, [delays::Vector{Vector{Int}}])

Mix together signals using amplitudes `amps` and `delays` (`delays` is specified in samples). If `amps` and the optional `delays` are vectors of vectors, then a vector of mixtures is returned. A vector of vectors is converted to a matrix using `M = reduce(hcat, mixes)`.

The returned signals will be shorter than the input signals corresponding to the maximum difference between delays, e.g., if `delays = [-2,0,5]`, the returned mixture will be 7 samples shorter.
"""
function mixture(signals, amps, delays::Vector{Int}=zeros(Int, length(amps)))
    length(signals) == length(amps) == length(delays) || throw(ArgumentError("The length of the inputs must be the same"))
    N = length(signals[1])
    mind, maxd = extrema(delays)
    maxdiff = maxd-mind
    Nn = N-maxdiff-mind
    ind = -mind+1:Nn
    inds = [ind .+ d for d in delays]
    sum(amps[i]*signals[i][inds[i]] for i in eachindex(signals))
end



function mixture(signals, amps::Vector{<:Vector{<:Number}}, delays::Vector{Vector{Int}}=[zeros(Int, length(a)) for a in amps])
    Nmix = length(amps)
    N = length(signals[1])
    mind, maxd = extrema(reduce(vcat,delays))
    maxdiff = maxd-mind
    Nn = N-maxdiff-mind
    ind = -mind+1:Nn
    map(amps, delays) do amps, delays
        length(signals) == length(amps) == length(delays) || throw(ArgumentError("The length of the inputs must be the same"))
        inds = [ind .+ d for d in delays]
        sum(amps[i]*signals[i][inds[i]] for i in eachindex(signals))
    end
end
