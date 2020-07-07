using DSP, ImageFiltering, Images, SparseArrays, FFTW, LPVSpectral
default(size=(1000,1000))

import DSP.Periodograms: compute_window

function istft(S::AbstractMatrix{Complex{T}}, wlen::Int, overlap::Int; nfft=nextfastfft(wlen), window=hanning) where T
    winc = wlen-overlap
    win, norm2 = compute_window(window, wlen)
    win² = win.^2
    nframes = size(S,2)-1
    outlen = nfft + nframes*winc
    out = zeros(T, outlen)
    tmp1 = Array{eltype(S)}(undef, size(S, 1))
    tmp2 = zeros(T, nfft)

    # plan = DSP.Periodograms.forward_plan(ones(nfft), zeros(Complex{T}, (nfft >> 1)+1)) # NOTE: this didn't seem to improve much
    # p = inv(plan)
    p = FFTW.plan_irfft(tmp1, nfft)
    wsum = zeros(outlen)
    for k = 1:size(S,2)
        copyto!(tmp1, 1, S, 1+(k-1)*size(S,1), length(tmp1))
        mul!(tmp2, p, tmp1)
        # tmp2 ./= AbstractFFTs.normalization(tmp2, size(tmp2)) # QUESTION: Not sure how to make this work

        ix = (k-1)*winc
        @inbounds for n=1:nfft
            out[ix+n] += tmp2[n]*win[n]
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
    q = quantile(vec(A), 0.5)
    A = copy(A)
    A .= max.(A, q)
    inds = findlocalmaxima(A)
    amps = A[inds]
    perm = sortperm(amps, rev=true)
    inds[perm[1:min(n,length(perm))]]
end

function separate(x1,x2,apeak, dpeak, S1, S2, fmat, nfft; kwargs...)
    @show numsources = length(apeak)
    peaka = @. (apeak + sqrt(apeak^2 + 4)) / 2

    bestsofar = Inf * ones(size(S1))
    bestind = zeros(size(S1))
    for i = 1:numsources
        score = @. abs(peaka[i] * cis(-fmat * dpeak[i]) * S1 - S2)^2 /
           (1 + peaka[i]^2)
        mask = score .< bestsofar
        bestind[mask] .= i
        bestsofar[mask] .= score[mask]
    end
    est = map(1:numsources) do i
        mask = bestind .== i
        M = [
            zeros(1, size(S1, 2)) # Add in zero DC component
            @. ((S1 + peaka[i] * cis(dpeak[i] * fmat) * S2) / (1 + peaka[i]^2)) * mask
        ]
        istft(M, nfft, nfft÷2; kwargs...)
    end
    reduce(hcat, est)
end



struct Hist
    A
    ar
    dr
    av
    dv
end

@recipe function plot(h::Hist)
    xguide --> "δ"
    yguide --> "A"
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


function duet(
    x1,
    x2,
    n     = 1024;
    n_sources,
    p     = 1,
    q     = 0, # powers used to weight histogram
    amax  = 0.7,
    dmax  = 3.6,
    abins = 35,
    dbins = 50,
    kernel_size = 1,
    window = hanning,
    kwargs...,
)

    S1 = stft(x1, n, n÷2; window = window, onesided=true, kwargs...)
    S2 = stft(x2, n, n÷2; window = window, onesided=true, kwargs...)
    S1, S2 = S1[2:end, :], S2[2:end, :] # remove dc
    freq = [(1:n÷2); -reverse((1:n÷2))] .* (2pi / n) # NOTE: not sure about the semantics of this concatenation
    fmat = freq[1:size(S1, 2)]'

    R21 = (S2 .+ eps()) ./ (S1 .+ eps()) # ratio of spectrograms
    a = abs.(R21) # relative attenuation
    α = @. a - 1 / a

    δ = @. -imag(log(R21)) / fmat # 'δ ' relative delay

    Sweight = @. (abs(S1) * abs(S2))^p * abs(fmat)^q # weights

    amask = @. (abs(α) < amax) & (abs(δ) < dmax)
    α_vec = α[amask]
    δ_vec = δ[amask]
    Sweight = Sweight[amask]
    # determine histogram indices
    αind = @. round(Int, 1 + (abins - 1) * (α_vec + amax) / (2amax))
    δind = @. round(Int, 1 + (dbins - 1) * (δ_vec + dmax) / (2dmax))
    # FULL-SPARSE TRICK TO CREATE 2D WEIGHTED HISTOGRAM
    A = Matrix(sparse(αind, δind, Sweight, abins, dbins))
    # smooth the histogram - local average 3-by-3 neighboring bins
    # A = twoDsmooth(A, kernel_size)
    K = KernelFactors.gaussian(kernel_size)
    A = imfilter(A, kernelfactors((K,K)))
    abins, dbins = size(A)
    ar,dr = LinRange(-amax , amax , abins ), LinRange(-dmax , dmax , dbins )
    inds = find_extrema(A, n_sources)
    av = [ar[i[1]] for i in inds]
    dv = [dr[i[2]] for i in inds]

    H = Hist(A,ar,dr,av,dv)

    est = separate(x1,x2, av, dv, S1, S2, fmat, n; (window=window, kwargs...)...)

    est, H

end

##
t = 0:0.1:2000
f = LinRange(0.9, 1.2, length(t))
x1 = sin.(t.*f)
x2 = @. 1.0 * sign(sin(2t + 0.15) + 0.01 * randn())
W = abs.(randn(2,2))
W ./ (2sum(W, dims=2))
R = W*[x1 x2]'
r1,r2 = R[1,:], R[2,:]

# A,ar,dr = hist2d(r1, r2, abins=50, dbins=60, dmax=2.5, amax=0.8)
# maximum(A)
# collect(product(ar,dr))[argmax(A)]
# av = [ar[i[1]] for i in inds]
# dv = [dr[i[2]] for i in inds]
# # surface(ar,dr,A, xlabel="A", ylabel="δ")
# heatmap(dr,ar,A, xlabel="δ", ylabel="A")
# scatter!(dv, av, m=(:red,))
##
est, H = duet(r1, r2, n_sources = 2, abins=30, dbins=51, dmax=2.5, amax=0.8, kernel_size=1)
plot(est, lab="Estimated components", c=:blue)
plot!([x1 x2], lab="True signal", c=:green)
plot!([r1 r2], lab="Received signal", c=:red)

##
plot(H)
##

# plot(plot.(welch_pgram.([x1,x2], 1024, fs=100))..., xscale=:log10)
plot(
    plot.(spectrogram.([x1, x2], 1024, fs=100, window=hanning), title="Sources")...,
    plot.(spectrogram.([r1, r2], 1024, fs=100, window=hanning), title="Received signals")...,
    plot.(spectrogram.(eachcol(est), 1024, fs=100, window=hanning), title="Extracted components")...,
    layout=(3,2), colorbar=false, ylabel="", xlabel=""
)

##
# contour(ar,dr,A, xlabel="A", ylabel="δ")


nfft = 128
T = Float64
plan = DSP.Periodograms.forward_plan(ones(nfft), zeros(Complex{T}, (nfft >> 1)+1))

inv(plan)
