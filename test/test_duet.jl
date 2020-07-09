using DSP
@testset "istft" begin
    @info "Testing istft"

    n    = 128
    nfft = 256
    x    = randn(1024)

    window = DSP.hamming
    S      = stft(x, n; window = window)
    xi     = istft(S, n; window = window)
    @test sum(abs, x - xi) < sqrt(eps())

    window = DSP.hamming
    S      = stft(x, n; window = window, nfft = nfft)
    xi     = istft(S, n; window = window, nfft = nfft)
    @test sum(abs, x - xi) < sqrt(eps())

    window = DSP.rect
    S      = stft(x, n; window = window, nfft = nfft)
    xi     = istft(S, n; window = window, nfft = nfft)
    @test sum(abs, x - xi) < sqrt(eps())

end


using Plots
##
t      = 0:0.1:2000
# f    = LinRange(0.9, 1.2, length(t))
x1     = sin.(t)
x2     = @. 1.3 * sign(sin(2t + 0.2) ) + 0.001 * randn()
W      = [0.3 0.7; 0.6 0.4]
R      = W * [x1 x2]'
r1, r2 = R[1, :], R[2, :]

##
est, H = duet(
    r1,
    r2,
    2,
    1024,
    abins       = 30,
    dbins       = 51,
    dmax        = 3.5,
    amax        = 2.8,
    kernel_size = 1,
    q           = 0,
    onesided    = true,
)
if isinteractive()
    plotinds = 10000:10200
    plot(plotinds, est[plotinds,:], lab = "Estimated components", c = :blue)
    plot!(plotinds, [x1 x2][plotinds,:], lab = "True signal", c = :black)
    plot!(plotinds, [r1 r2][plotinds,:], lab = "Received signal", c = :orange) |> display
end

@test any(<(0), H.av)
# @test any(>(0), H.dv)

plot(H)
##

# plot(plot.(welch_pgram.([x1,x2], 1024, fs=100))..., xscale=:log10)
isinteractive() && plot(
    plot.(spectrogram.([x1, x2], 1024, fs = 100, window = hanning), title = "Sources")...,
    plot.(
        spectrogram.([r1, r2], 1024, fs = 100, window = hanning),
        title = "Received signals",
    )...,
    plot.(
        spectrogram.(eachcol(est), 1024, fs = 100, window = hanning),
        title = "Extracted components",
    )...,
    layout = (3, 2),
    colorbar = false,
    ylabel = "",
    xlabel = "",
)
