using DSP, ImageFiltering, SparseArrays, FFTW, RecipesBase

import DSP.Periodograms: compute_window

function istft(S::AbstractMatrix{Complex{T}}, wlen::Int, overlap::Int=wlen÷2; onesided=true, nfft=nextfastfft(wlen), window=hanning) where T
    wlen > 0 || throw(ArgumentError("The window length must be positive."))
    0 <= overlap < wlen || throw(ArgumentError("The overlap must be nonnegative and smaller than the window length."))
    nfft >= wlen || throw(ArgumentError("The transform length must be at least the window length."))
    expected_rows = onesided ? nfft ÷ 2 + 1 : nfft
    size(S, 1) == expected_rows || throw(DimensionMismatch("The spectrum size does not match the transform length."))
    size(S, 2) > 0 || throw(ArgumentError("The spectrogram must contain at least one frame."))
    winc = wlen-overlap
    win, norm2 = compute_window(window, wlen)
    win² = win.^2
    nframes = size(S,2)-1
    outlen = wlen + nframes*winc
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
        @inbounds for n=1:wlen
            out[ix+n] += real(tmp2[n])*win[n]
            wsum[ix+n] += win²[n]
        end

    end
    mw = median(wsum)
    @inbounds for i = eachindex(wsum,out)
        if wsum[i] >= 1e-3*mw
            out[i] /= wsum[i]
        end
    end
    out
end


function find_extrema(A, n; min_peak_distance=0, min_peak_ratio=0)
    peak = maximum(A)
    isfinite(peak) && peak > 0 || throw(ArgumentError("No positive histogram energy remains. Check the energy threshold, attenuation bounds, and delay bounds."))
    inds = filter(i -> A[i] > 0 && A[i] >= min_peak_ratio * peak, findlocalmaxima(A))
    sort!(inds; by=i -> A[i], rev=true)
    selected = CartesianIndex{2}[]
    for i in inds
        all(j -> hypot((i - j).I...) >= min_peak_distance, selected) || continue
        push!(selected, i)
        length(selected) == n && break
    end
    isempty(selected) && throw(ArgumentError("No distinct histogram peaks were found. Adjust histogram bounds, smoothing, or peak thresholds."))
    if length(selected) < n
        @warn "Fewer distinct histogram peaks than requested sources; returning only the detected sources." requested=n detected=length(selected)
    end
    return selected
end

function separate(αₚ::Vector{<:Real}, δₚ::Vector{<:Real}, S1::Matrix{Complex{T}}, S2, freqs, n;
                  reliable=trues(size(S1, 1)-1, size(S1, 2)), mask_type=:hard,
                  mask_temperature=0.1, kwargs...) where T
    N_sources = length(αₚ)
    αₚ = @. (αₚ + sqrt(αₚ^2 + 4)) / 2
    cisδf = Vector{Complex{T}}(undef, length(freqs))
    function model_score(i)
        @. cisδf = cis(-freqs * δₚ[i])
        @views @. abs2(αₚ[i] * cisδf * S1[2:end,:] - S2[2:end,:]) / (1 + αₚ[i]^2)
    end
    if mask_type == :hard
        bestsofar = fill(T(Inf), size(reliable))
        bestind = zeros(Int, size(reliable))
        for i in 1:N_sources
            score = model_score(i)
            selected = reliable .& (score .< bestsofar)
            bestind[selected] .= i
            bestsofar[selected] .= score[selected]
        end
        masks = [bestind .== i for i in 1:N_sources]
    else
        scores = map(model_score, 1:N_sources)
        @views energy = abs2.(S1[2:end, :]) .+ abs2.(S2[2:end, :])
        minimum_score = reduce((a, b) -> min.(a, b), scores)
        masks = map(scores) do score
            mask = zeros(T, size(reliable))
            @. mask[reliable] = exp(-((score[reliable] - minimum_score[reliable]) /
                energy[reliable]) / mask_temperature)
            mask
        end
        total = reduce(+, masks)
        for mask in masks
            mask[reliable] ./= total[reliable]
        end
    end
    spectrograms = map(1:N_sources) do i
        S = zeros(eltype(S1), size(S1))
        @. cisδf = cis(freqs * δₚ[i])
        @views @. S[2:end,:] = ((S1[2:end,:] + αₚ[i] * cisδf * S2[2:end,:]) /
            (1 + αₚ[i]^2)) * masks[i]
        S
    end
    est = map(spectrograms) do S
        istft(S, n, n÷2; kwargs...)
    end
    reduce(hcat, est), masks, spectrograms
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
    spectrograms
    "Time-frequency bins retained by the relative power threshold."
    reliable_bins
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

`min_bin_power` is a relative power threshold in `[0, 1)`, applied separately
to each microphone's largest non-DC time-frequency bin. A bin is used only when
both microphone powers exceed their thresholds. Excluded bins have zero masks
and do not contribute to the histogram. The default excludes zero-power bins.
`H.reliable_bins` records the retained bins, excluding the DC row.

`mask_type=:hard` assigns each retained bin to its minimum-error source.
`mask_type=:soft` uses weights proportional to
`exp(-(score - minimum_score) / (mask_temperature * combined_bin_power))`,
normalized to sum to one across sources in each retained bin. The score is the
squared residual of the source's attenuation-delay model divided by
`1 + attenuation^2`. `mask_temperature` must be finite and positive; smaller
values produce more concentrated assignments. Soft masks do not guarantee
improved separation when source models or histogram peaks are inaccurate.

`min_peak_distance` sets the minimum Euclidean spacing between selected peaks
in histogram-bin coordinates. `min_peak_ratio` excludes peaks below that fraction
of the histogram maximum. Both default to zero. When fewer distinct peaks are
available than requested, the function emits a warning and returns that smaller
number of sources. A histogram without positive energy throws `ArgumentError`.
Peak selection does not infer the physical number of sources.

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

For large delays, it appears to be more efficient to set `dmax` to something small (20), `nfft ≈ 8n, p=0.5, q=2`

- `bigdelay` indicates whether or not the two microphones are far apart. If `true`, the delay `δ` is estimated using the differential method (see the paper sec 8.4.1) and the delay map is smoothed using `kernel_sizeδ`.



Implementation based on *Rickard, Scott. (2007). The DUET blind source separation algorithm. 10.1007/978-1-4020-6479-1_8.*
"""
function duet(
    x1,
    x2,
    n_sources,
    n     = 1024;
    nfft  = n,
    p     = 1,
    q     = 0,
    amax  = 0.7,
    dmax  = 3.6,
    abins = 35,
    dbins = 50,
    kernel_size = 1,
    window = hanning,
    onesided = true,
    bigdelay = false,
    kernel_sizeδ = 0.1,
    min_bin_power::Real = 0,
    mask_type::Symbol = :hard,
    mask_temperature::Real = 0.1,
    min_peak_distance::Real = 0,
    min_peak_ratio::Real = 0,
    kwargs...,
)

    length(x1) == length(x2) || throw(DimensionMismatch("Microphone signals must have equal lengths."))
    n_sources isa Integer && n_sources > 0 || throw(ArgumentError("n_sources must be a positive integer."))
    n > 1 && nfft >= n || throw(ArgumentError("The window length must exceed one and nfft must be at least the window length."))
    length(x1) >= n || throw(ArgumentError("Signals must contain at least one complete window."))
    all(isfinite, x1) && all(isfinite, x2) || throw(ArgumentError("Microphone signals must be finite."))
    isfinite(amax) && amax > 0 && isfinite(dmax) && dmax > 0 || throw(ArgumentError("Attenuation and delay bounds must be finite and positive."))
    abins isa Integer && abins > 1 && dbins isa Integer && dbins > 1 || throw(ArgumentError("Histogram dimensions must be integers greater than one."))
    isfinite(p) && p >= 0 && isfinite(q) || throw(ArgumentError("Histogram powers must be finite and p must be nonnegative."))
    isfinite(kernel_size) && kernel_size >= 0 && isfinite(kernel_sizeδ) && kernel_sizeδ >= 0 || throw(ArgumentError("Smoothing widths must be finite and nonnegative."))
    isfinite(min_bin_power) && 0 <= min_bin_power < 1 || throw(ArgumentError("min_bin_power must be in [0, 1)."))
    mask_type in (:hard, :soft) || throw(ArgumentError("mask_type must be :hard or :soft."))
    isfinite(mask_temperature) && mask_temperature > 0 || throw(ArgumentError("mask_temperature must be finite and positive."))
    isfinite(min_peak_distance) && min_peak_distance >= 0 || throw(ArgumentError("min_peak_distance must be finite and nonnegative."))
    isfinite(min_peak_ratio) && 0 <= min_peak_ratio <= 1 || throw(ArgumentError("min_peak_ratio must be in [0, 1]."))

    S1 = stft(x1, n, n÷2; nfft = nfft, window = window, onesided=onesided, kwargs...)
    S2 = stft(x2, n, n÷2; nfft = nfft, window = window, onesided=onesided, kwargs...)
    # S1, S2 = S1[2:end, :], S2[2:end, :] # remove dc
    S1[1, :] .= 0; S2[1, :] .= 0 # remove dc
    @views S1w, S2w = S1[2:end,:], S2[2:end,:]
    if onesided
        freqs = FFTW.rfftfreq(nfft)[2:end] .* (2pi)
    else
        freqs = FFTW.fftfreq(nfft)[2:end] .* (2pi)
        # freqs = freqs[1:size(S1, 1)]
    end

    power1, power2 = abs2.(S1w), abs2.(S2w)
    reliable = (power1 .> min_bin_power * maximum(power1)) .&
               (power2 .> min_bin_power * maximum(power2))
    R21 = map((s1, s2) -> iszero(s1) ? zero(s1) : s2 / s1, S1w, S2w)
    α = abs.(R21) # relative attenuation
    α = @. α - 1 / α # symmetric attenuation

    if bigdelay
        Δω = 2pi / nfft
        Δω*dmax > π && @warn("frequency resolution not sufficient for the chosen maximum delay.")
        size(R21, 1) > 1 || throw(ArgumentError("Large-delay estimation requires at least two nonzero frequency bins."))
        δ0 = similar(R21)
        @views @. δ0[1:end-1, :] = R21[2:end, :] * conj(R21[1:end-1, :])
        δ0[end, :] .= δ0[end-1, :]
        adjacent_reliable = copy(reliable)
        @views adjacent_reliable[1:end-1, :] .&= reliable[2:end, :]
        adjacent_reliable[end, :] .= adjacent_reliable[end-1, :]
        δ0[.!adjacent_reliable] .= 0
        if kernel_sizeδ > 0
            δ0 = imfilt(δ0, kernel_sizeδ)
        end
        δ = @. -angle(δ0) / Δω
    else
        δ = @. -angle(R21) / freqs # δ relative delay
    end


    Sweight = @. (abs(S1w) * abs(S2w))^p * abs(freqs)^q # weights

    αmask = @. reliable & isfinite(Sweight) & (Sweight > 0) & (abs(α) < amax) & (abs(δ) < dmax)
    if bigdelay
        αmask .&= adjacent_reliable
    end
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
    inds = find_extrema(A, n_sources; min_peak_distance=min_peak_distance,
        min_peak_ratio=min_peak_ratio)
    av = [ar[i[1]] for i in inds]
    dv = [dr[i[2]] for i in inds]

    est, masks, spectrograms = separate(av, dv, S1, S2, freqs, n;
        reliable=reliable, mask_type=mask_type, mask_temperature=mask_temperature,
        (nfft=nfft, window=window, onesided=onesided, kwargs...)...)
    H = DUET(A,ar,dr,av,dv,α,δ,masks,spectrograms,reliable)

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
