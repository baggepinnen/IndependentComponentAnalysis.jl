using IndependentComponentAnalysis, Test, LinearAlgebra, Random, DSP
import Statistics: cov

@testset "ICA diagnostics and reproducibility" begin
    X = randn(MersenneTwister(3), 3, 2000).^3
    model, info = fit(ICA, X, 3; rng=MersenneTwister(4), return_info=true, maxiter=500)
    repeated, repeated_info = fit(ICA, X, 3; rng=MersenneTwister(4), return_info=true, maxiter=500)
    @test info isa ICAInfo
    @test info.converged
    @test 1 <= info.iterations <= 500
    @test info.residual < 1e-6
    @test info.numerical_rank == 3
    @test model.W == repeated.W
    @test info.iterations == repeated_info.iterations
    @test fit(ICA, X, 3; rng=MersenneTwister(4), maxiter=500).W == model.W

    initial = Matrix{Float64}(I, 3, 3)
    rng = MersenneTwister(5)
    untouched = copy(rng)
    fit(ICA, X, 3; winit=initial, rng=rng, tol=Inf)
    @test rand(rng) == rand(untouched)
    @test initial == Matrix{Float64}(I, 3, 3)
    _, raw_info = fit(ICA, X, 2; do_whiten=false, tol=Inf, return_info=true)
    @test raw_info.numerical_rank === nothing

    for policy in (:log, :ignore)
        result = if policy == :log
            @test_logs (:error, "Did not converge") fit(ICA, X, 3;
                winit=initial, maxiter=1, tol=eps(), return_info=true, on_nonconvergence=policy)
        else
            @test_logs fit(ICA, X, 3; winit=initial, maxiter=1, tol=eps(),
                return_info=true, on_nonconvergence=policy)
        end
        @test !result[2].converged
        @test result[2].iterations == 1
        @test result[2].residual > eps()
    end
    @test_throws IndependentComponentAnalysis.ConvergenceException fit(ICA, X, 3;
        winit=initial, maxiter=1, tol=eps(), on_nonconvergence=:throw)
    @test_throws ArgumentError fit(ICA, X, 3; on_nonconvergence=:unknown)
    @test_throws ArgumentError fit(ICA, X, 3; winit=zeros(3, 3))
    @test_throws ArgumentError fit(ICA, X, 3; winit=ones(3, 3))
    @test_throws ArgumentError fit(ICA, fill(NaN, 3, 20), 2)

    observations = Float64[1 1 -1 -1; 1 -1 1 -1]
    W = Matrix{Float64}(I, 2, 2)
    updated, core_info = ica!(FastICA(), W, observations, Tanh(1.0), 2, 1e-8; return_info=true)
    @test updated === W
    @test core_info.converged
    @test core_info.iterations == 1
    @test core_info.residual ≈ 0 atol=1e-14
    @test core_info.numerical_rank === nothing
end

@testset "Rank-aware whitening" begin
    for T in (Float32, Float64)
        S = randn(MersenneTwister(8), T, 2, 1000).^3
        X = vcat(S, sum(S; dims=1), zeros(T, 1, size(S, 2)))
        @test_throws ArgumentError fit(ICA, X, 3)
        model, info = fit(ICA, X, 2; tol=Inf, return_info=true, rng=MersenneTwister(9))
        @test info.numerical_rank == 2
        @test all(isfinite, model.W)
        @test eltype(model.W) == T
        @test model.W' * cov(X; dims=2) * model.W ≈ Matrix{T}(I, 2, 2)
        @test_throws ArgumentError fit(ICA, zeros(T, 3, 20), 1)
        @test_throws ArgumentError fit(ICA, ones(T, 3, 20), 1)
    end
    X = randn(MersenneTwister(10), 3, 1000)
    X[3, :] .*= 1e-5
    @test_throws ArgumentError fit(ICA, X, 3; rank_rtol=1e-8)
    @test_throws ArgumentError fit(ICA, X, 3; rank_rtol=0, rank_atol=1e-8)
    _, info = fit(ICA, X, 3; rank_rtol=1e-12, tol=Inf, return_info=true)
    @test info.numerical_rank == 3
    for key in (:rank_atol, :rank_rtol), value in (-1.0, NaN, Inf)
        @test_throws ArgumentError fit(ICA, X, 2; (key => value,)...)
    end
end

@testset "Component reconstruction" begin
    X = randn(MersenneTwister(11), 3, 1000).^3 .+ [2.0, -1.0, 3.0]
    for mean_option in (nothing, 0, [1.0, 2.0, -3.0])
        model = fit(ICA, X, 3; rng=MersenneTwister(12), tol=Inf, mean=mean_option)
        sources = transform(model, X)
        A, B = mixingmatrix(model), unmixingmatrix(model)
        @test size(A) == (3, 3)
        @test B * A ≈ Matrix{Float64}(I, 3, 3)
        @test B * (X .- mean(model)) ≈ sources
        @test reconstruct(model, sources) ≈ X
        @test reconstruct(model, sources[:, 1]) ≈ X[:, 1]
        @test reconstruct(model, sources; add_mean=false) ≈ X .- mean(model)
        parts = [reconstruct(model, sources; components=i, add_mean=false) for i in 1:3]
        @test sum(parts) .+ mean(model) ≈ X
        @test reconstruct(model, sources; components=[3, 1], add_mean=false) ≈ parts[1] + parts[3]
        @test reconstruct(model, sources; components=Int[]) ≈ repeat(mean(model), 1, size(X, 2))
        @test iszero(reconstruct(model, sources[:, 1]; components=Int[], add_mean=false))
        @test_throws DimensionMismatch reconstruct(model, sources[1:2, :])
        @test_throws ArgumentError reconstruct(model, sources; components=[1, 1])
        @test_throws ArgumentError reconstruct(model, sources; components=[0])
        @test_throws ArgumentError reconstruct(model, sources; components=[4])
        @test_throws ArgumentError reconstruct(model, sources; components=[1.5])
    end
    sources = randn(MersenneTwister(13), 2, 1000)
    X = [1.0 0.2; -0.4 1.0; 0.3 0.7] * sources .+ [1.0, 2.0, 3.0]
    model = fit(ICA, X, 2; tol=Inf)
    @test size(mixingmatrix(model)) == (3, 2)
    @test reconstruct(model, transform(model, X)) ≈ X
    noisy_X = X .+ 0.1 .* randn(MersenneTwister(14), size(X))
    @test reconstruct(model, transform(model, noisy_X)) ≈
        pinv(model.W') * model.W' * (noisy_X .- mean(model)) .+ mean(model)
end

@testset "DUET thresholds and peak diagnostics" begin
    t = 0:1023
    s1, s2 = sinpi.(t ./ 8), cospi.(t ./ 4)
    x1, x2 = s1 + s2, 0.5 .* s1 + 2 .* s2
    settings = (; window=DSP.rect, min_bin_power=1e-8, amax=2.0,
        dmax=2.0, abins=17, dbins=9, kernel_size=0)
    est, H = @test_logs duet(x1, x2, 2, 128; settings...)
    @test sort(H.av) ≈ [-1.5, 1.5]
    @test size(est) == (length(x1), 2)
    @test est[:, argmin(H.av)] ≈ s1 atol=1e-12
    @test est[:, argmax(H.av)] ≈ s2 atol=1e-12
    @test all(sum(H.masks) .== H.reliable_bins)
    @test count(H.reliable_bins) == 2 * size(H.reliable_bins, 2)
    for mask in H.masks
        @test all(iszero, mask[.!H.reliable_bins])
    end
    scaled, scaled_H = duet(1e-8 .* x1, 1e-8 .* x2, 2, 128; settings...)
    @test H.av == scaled_H.av
    @test H.reliable_bins == scaled_H.reliable_bins
    @test scaled ≈ 1e-8 .* est
    @test_logs (:warn, r"Fewer distinct histogram peaks") duet(x1, x2, 3, 128; settings...)
    @test_logs (:warn, r"Fewer distinct histogram peaks") duet(x1, x2, 2, 128;
        settings..., min_peak_distance=20)
    _, pruned = @test_logs (:warn, r"Fewer distinct histogram peaks") duet(x1, x2, 2, 128;
        settings..., min_peak_ratio=0.5)
    @test length(pruned.av) == 1
    @test_throws ArgumentError duet(zeros(1024), zeros(1024), 2, 128)
    @test_throws ArgumentError duet(x1, zeros(1024), 2, 128)
    @test_throws ArgumentError duet(x1, x2, 2, 128; settings..., amax=0.1)
    @test_throws DimensionMismatch duet(x1, x2[1:end-1], 2, 128)
    @test_throws ArgumentError duet(x1, x2, 0, 128)
    for (key, value) in ((:min_bin_power, -1), (:min_bin_power, 1),
            (:min_bin_power, NaN), (:mask_type, :unknown), (:mask_temperature, 0),
            (:mask_temperature, Inf), (:min_peak_distance, -1), (:min_peak_ratio, 2))
        @test_throws ArgumentError duet(x1, x2, 2, 128; (key => value,)...)
    end
end

@testset "DUET soft masks" begin
    S1 = ComplexF64[0 0 0; 1 1 1; 1 1 1]
    S2 = ComplexF64[0 0 0; 0.5 1 2; 0.5 1 2]
    freqs = [pi/2, pi]
    reliable = Bool[1 1 1; 0 1 1]
    _, hard, _ = IndependentComponentAnalysis.separate([-1.5, 1.5], [0.0, 0.0],
        S1, S2, freqs, 4; reliable=reliable, window=DSP.rect)
    for temperature in (1e-300, 0.01, 0.1, 1.0)
        est, masks, spectra = IndependentComponentAnalysis.separate([-1.5, 1.5], [0.0, 0.0],
            S1, S2, freqs, 4; reliable=reliable, mask_type=:soft,
            mask_temperature=temperature, window=DSP.rect)
        @test sum(masks) ≈ Float64.(reliable)
        @test all(m -> all(x -> 0 <= x <= 1, m), masks)
        @test masks[1][1, 2] ≈ 0.5
        @test masks[2][1, 2] ≈ 0.5
        @test masks[1][1, 1] > masks[2][1, 1]
        @test masks[1][1, 3] < masks[2][1, 3]
        @test masks[1][2, 1] == masks[2][2, 1] == 0
        @test all(isfinite, est)
        @test all(S -> all(isfinite, S), spectra)
        temperature == 0.01 && @test masks[1][1, 1] > 0.99
    end
    @test hard[1][1, 1]
    @test hard[2][1, 3]

    t = 0:1023
    x1 = sinpi.(t ./ 8) + cospi.(t ./ 4)
    x2 = 0.5 .* sinpi.(t ./ 8) + 2 .* cospi.(t ./ 4)
    for onesided in (true, false)
        est, H = duet(x1, x2, 2, 128; window=DSP.rect, min_bin_power=1e-8,
            amax=2.0, dmax=2.0, abins=17, dbins=9, kernel_size=0,
            mask_type=:soft, onesided=onesided)
        @test all(isfinite, est)
        @test sum(H.masks) ≈ Float64.(H.reliable_bins)
    end

    x1 = repeat(randn(MersenneTwister(15), 256), 8)
    x2 = 0.7 .* circshift(x1, 7)
    _, H = duet(x1, x2, 1, 256; window=DSP.rect, min_bin_power=0.01,
        bigdelay=true, mask_type=:soft, amax=2.0, dmax=12, abins=41,
        dbins=49, kernel_size=0, kernel_sizeδ=0)
    @test H.dv ≈ [7]
    @test sum(H.masks) ≈ Float64.(H.reliable_bins)
end
