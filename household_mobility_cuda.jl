using Interpolations, Distributions, Random
using DataFrames, Statistics, NLsolve, Profile
using CairoMakie, Parameters, JLD2, GLM, HypothesisTests
using CUDA, CUDAKernels, KernelAbstractions, Tullio
using Optim, ForwardDiff, BenchmarkTools

# Parameters 
function HouseholdProblem(;β = 0.96, α = 0.2, ρ = 0.9, R = 1.03, 
                            σ = 0.2, δ = 0.8, ϕ = 0.3, γ = 1.5, 
                            η = 0.95, rng = Xoshiro(2024))
    # Utility function
    u(c, n) = (γ == 1 ? η*log(c) + (1-η)*log(1-n) : ((c^η)*((1-n)^(1-η)))^(1-γ)/(1-γ))
    # Grids
    ϵ_grids = collect(range(-2*σ/sqrt(1-ρ^2), 2*σ/sqrt(1-ρ^2), length = 10))
    a_grids = collect(range(0.0, 20.0, length = 20))
    s_grids = collect(range(1e-8, 3.0, length = 10))
    c_grids = collect(range(1e-5, 1.0, length = 20))
    n_grids = collect(range(0.0, 1.0, length = 20))
    # Monte Carlo
    ξ = σ*randn(rng, 300)
    return (; β, α, ρ, R, σ, δ, ϕ, γ, η, u, ϵ_grids, a_grids, s_grids, c_grids, n_grids, ξ)
end

HP = HouseholdProblem()

function T(v; HP = HP)
    @unpack β, α, ρ, R, σ, δ, ϕ, γ, η, u, ϵ_grids, a_grids, s_grids, c_grids, n_grids, ξ = HP
    # Interpolation
    v_func = LinearInterpolation((a_grids, s_grids, ϵ_grids), v, extrapolation_bc = Line())
    cu_v = cu(v_func)
    # Set up the grids
    cu_a = cu(a_grids)
    cu_s = cu(s_grids)
    cu_ϵ = cu(ϵ_grids)
    cu_c = cu(c_grids)
    cu_n = cu(n_grids)
    cu_ξ = cu(ξ)
    n_a, n_s, n_ϵ, n_c, n_n, n_ξ = length(a_grids), length(s_grids), length(ϵ_grids), length(c_grids), length(n_grids), length(ξ)

    # Flow utility
    @tullio income[i,j,k,l] := $R * cu_a[i] + $α * cu_s[j] * exp(cu_ϵ[k]) * cu_n[l]
    @tullio consumption[i,j,k,l,m] := income[i,j,k,l] * cu_c[m]
    @tullio utility[i,j,k,l,m] := u(consumption[i,j,k,l,m], cu_n[l]) 
    # State transition
    @tullio new_a[i,j,k,l,m,n] := income[i,j,k,l] - consumption[i,j,k,l,m] (i in 1:n_a, j in 1:n_s, k in 1:n_ϵ, l in 1:n_n, m in 1:n_c, n in 1:n_ξ)
    @tullio new_s[i,j,k,l,m,n] := $δ * cu_s[j] + $ϕ * cu_n[l] (i in 1:n_a, j in 1:n_s, k in 1:n_ϵ, l in 1:n_n, m in 1:n_c, n in 1:n_ξ)
    @tullio new_ϵ[i,j,k,l,m,n] := $ρ * cu_ϵ[k] + cu_ξ[n] (i in 1:n_a, j in 1:n_s, k in 1:n_ϵ, l in 1:n_n, m in 1:n_c, n in 1:n_ξ)
    weight = cu(fill(1/n_ξ, n_ξ))
    new_a, new_s, new_ϵ = parent(new_a), parent(new_s), parent(new_ϵ)
    f_u = cu_v.(new_a, new_s, new_ϵ)
    # EV
    @tullio EV[i,j,k,l,m] := $β * weight[n] * f_u[i,j,k,l,m,n] 
    # Candidate 
    @tullio candidate[i,j,k,l,m] := utility[i,j,k,l,m] + EV[i,j,k,l,m]
    # Find the maximum
    Tv = Array(dropdims(mapreduce(identity, max, candidate, dims = (4,5)), dims = (4,5)))
    ind = Array(dropdims(argmax(candidate, dims = (4,5)), dims = (4,5)))
    consumption_prop = c_grids[getindex.(ind, 5)]
    labor = n_grids[getindex.(ind, 4)]
    
    return (; v = Tv, policy_c = consumption_prop, policy_n = labor)
end

v = zeros(length(HP.a_grids), length(HP.s_grids), length(HP.ϵ_grids))

function VFI(v; HP = HP, tol = 1e-6, max_iter = 1000)
    @unpack β, α, ρ, R, σ, δ, ϕ, γ, η, u, ϵ_grids, a_grids, s_grids, ξ = HP
    sol = fixedpoint(v -> T(v).v, v, iterations = max_iter, xtol = tol, m = 3, show_trace = true)
    return (; v = sol.zero, c = T(sol.zero).policy_c, n = T(sol.zero).policy_n)
end

sol = VFI(v)

# Save the results
jldsave("household_mobility.jld2", v = sol.v, c = sol.c, n = sol.n)
v, c, n = load("household_mobility.jld2", "v", "c", "n")

# Simulation
#initial distribution 
function simulate(;T = 300, N = 1000, sol = sol, HP = HP)
    @unpack β, α, ρ, R, σ, δ, ϕ, γ, η, u, ϵ_grids, a_grids, s_grids, ξ = HP
    v, c, n = sol
    a = zeros(N, T+1)
    s = zeros(N, T+1)
    ϵ = zeros(N, T+1)
    consumption = zeros(N, T)
    labor = zeros(N, T)
    a[:, 1] = exp.(randn(N))
    s[:, 1] = rand(N)
    ϵ[:, 1] = 0.01*randn(N)
    c_func = LinearInterpolation((HP.a_grids, HP.s_grids, HP.ϵ_grids), c, extrapolation_bc = Line())
    n_func = LinearInterpolation((HP.a_grids, HP.s_grids, HP.ϵ_grids), n, extrapolation_bc = Line())
    for t in 1:T
        disposable = R*a[:, t] + α*s[:, t].*exp.(ϵ[:, t])
        consumption[:, t] = c_func.(a[:, t], s[:, t], ϵ[:, t]) .* disposable
        labor[:, t] = n_func.(a[:, t], s[:, t], ϵ[:, t])
        a[:, t+1] = R*a[:, t] + α*s[:, t].*exp.(ϵ[:, t]).*labor[:, t] - consumption[:, t]
        s[:, t+1] = δ*s[:, t] + ϕ*labor[:, t]
        ϵ[:, t+1] = ρ*ϵ[:, t] + σ*randn(N)
    end
    return DataFrame(id = vec(transpose(repeat(1:N, 1, T))), t = repeat(1:T, N), 
                    init_asset = vec(transpose(a[:, 1:T])), 
                    init_human_capital = vec(transpose(s[:, 1:T])), init_ϵ = vec(transpose(ϵ[:, 1:T])), 
                    consumption = vec(transpose(consumption)), labor = vec(transpose(labor)), 
                    asset = vec(transpose(a[:, 2:T+1])), human_capital = vec(transpose(s[:, 2:T+1])))
end

df = simulate()

# Plot the mean and sd of asset
begin
    fig = Figure()
    ax1 = Axis(fig[1, 1]; xlabel = "t", title = "Mean and SD of Asset")
    new_df = combine(groupby(df, :t), :asset => mean, :asset => std)
    xlims!(ax1, 0, 300)
    lines!(ax1, new_df.t, new_df.asset_mean, color = :blue, label = "Mean")
    lines!(ax1, new_df.t, new_df.asset_std, color = :red, label = "SD")
    Legend(fig[1, 2], ax1, framevisible = false)
    ax2 = Axis(fig[2, 1]; xlabel = "t")
    xlims!(ax2, 0, 300)
    lines!(ax2, new_df.t, new_df.asset_std ./ new_df.asset_mean, color = :green, label = "SD/Mean")
    Legend(fig[2, 2], ax2, framevisible = false)
    fig
end

# Plot the distribution dynamics
begin
    S = 30
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "Asset", title = "Asset Distribution")
    for t in 1:S
        density!(ax, df[df.t .== t, :asset], strokecolor = RGBAf(t/S, 0.0, 0.0, 1.0), color = (:blue, 0.0), strokewidth = 1)
    end
    fig
end

