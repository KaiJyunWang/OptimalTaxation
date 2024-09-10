using Interpolations, Distributions, Random
using DataFrames, Statistics, NLsolve
using CairoMakie, Parameters, Optimization 
using Optim, ForwardDiff, BenchmarkTools
using JLD2

# Parameters 
function HouseholdProblem(;β = 0.96, α = 0.2, ρ = 0.9, R = 1.03, 
                            σ = 0.2, δ = 0.8, ϕ = 0.3, γ = 1.5, 
                            η = 0.95, rng = Xoshiro(2024))
    # Utility function
    u(c, n) = c > 0 ? ((γ == 1 ? η*log(c) + (1-η)*log(1-n) : ((c^η)*((1-n)^(1-η)))^(1-γ)/(1-γ))) : -1e10 + 1e-10*c
    # Grids
    ϵ_grids = collect(range(-2*σ/sqrt(1-ρ^2), 2*σ/sqrt(1-ρ^2), length = 5))
    a_grids = collect(range(0.0, 20.0, length = 20))
    s_grids = collect(range(1e-8, 3.0, length = 10))
    # Monte Carlo
    ξ = σ*randn(rng, 300)
    return (; β, α, ρ, R, σ, δ, ϕ, γ, η, u, ϵ_grids, a_grids, s_grids, ξ)
end

HP = HouseholdProblem()

function T(v; HP = HP)
    @unpack β, α, ρ, R, σ, δ, ϕ, γ, η, u, ϵ_grids, a_grids, s_grids, ξ = HP
    # Interpolation
    v_func = LinearInterpolation((a_grids, s_grids, ϵ_grids), v, extrapolation_bc = Line())
    Tv = similar(v)
    policy_c = similar(v)
    policy_n = similar(v)
    for (i, a) in enumerate(a_grids)
        for (j, s) in enumerate(s_grids)
            for (k, ϵ) in enumerate(ϵ_grids)
                function obj(x)
                    n = 0.5 + atan(x[1])/π
                    c = (R*a + α*s*exp(ϵ)*n)*(0.5 + atan(x[2])/π)
                    new_a = R*a + α*s*exp(ϵ)*n - c 
                    new_s = δ*s + ϕ*n
                    return -u(c, n) - β*mean(v_func.(new_a, new_s, ρ*ϵ .+ ξ))
                end
                td = TwiceDifferentiable(obj, zeros(2); autodiff = :forward)
                res = optimize(td, zeros(2), Newton())
                Tv[i, j, k] = -res.minimum
                policy_n[i,j,k] = (0.5 + atan(res.minimizer[1])/π)
                policy_c[i,j,k] = (R*a + α*s*exp(ϵ)*policy_n[i,j,k])*(0.5 + atan(res.minimizer[2])/π)
            end
        end
    end
    return (; v = Tv, policy_c = policy_c, policy_n = policy_n)
end

v = zeros(length(HP.a_grids), length(HP.s_grids), length(HP.ϵ_grids))

function VFI(v; HP = HP, tol = 1e-6, max_iter = 1000)
    @unpack β, α, ρ, R, σ, δ, ϕ, γ, η, u, ϵ_grids, a_grids, s_grids, ξ = HP
    sol = fixedpoint(v -> T(v).v, v, iterations = max_iter, xtol = tol, m = 3)
    return (; v = sol.zero, c = T(sol.zero).policy_c, n = T(sol.zero).policy_n)
end

sol = VFI(v)

# Save the results
jldsave("household_mobility.jld2", v = sol.v, c = sol.c, n = sol.n)
v, c, n = load("household_mobility.jld2", "v", "c", "n")

# Simulation
#initial distribution 
function simulate(;T = 1000, N = 1000, sol = sol, HP = HP)
    @unpack β, α, ρ, R, σ, δ, ϕ, γ, η, u, ϵ_grids, a_grids, s_grids, ξ = HP
    v, c, n = sol
    a = zeros(N, T+1)
    s = zeros(N, T+1)
    ϵ = zeros(N, T+1)
    consumption = zeros(N, T)
    labor = zeros(N, T)
    a[:, 1] = randn(N)
    s[:, 1] = rand(N)
    ϵ[:, 1] = 0.01*randn(N)
    c_func = LinearInterpolation((HP.a_grids, HP.s_grids, HP.ϵ_grids), c, extrapolation_bc = Line())
    n_func = LinearInterpolation((HP.a_grids, HP.s_grids, HP.ϵ_grids), n, extrapolation_bc = Line())
    for t in 1:T
        consumption[:, t] = c_func.(a[:, t], s[:, t], ϵ[:, t])
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
    ax = Axis(fig[1, 1]; xlabel = "Time", ylabel = "Asset")
    new_df = combine(groupby(df, :t), :asset => mean, :asset => std)
    xlims!(ax, 0, 1000)
    lines!(ax, new_df.t, new_df.asset_mean, color = :blue, label = "Mean")
    lines!(ax, new_df.t, new_df.asset_std, color = :red, label = "SD")
    Legend(fig[1, 2], ax, framevisible = false)
    fig
end

# Plot the distribution dynamics of asset
begin
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "Asset", ylabel = "Density")
    for t in 1:10:1000
        density!(ax, df[df.t .== t, :asset], strokecolor = RGBAf(t/1000, 0.5, 1 - t/1000, 1.0), color = (:blue, 0.0), strokewidth = 2)
    end
    fig
end
save("asset_distribution.png", fig)