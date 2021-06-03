using Epidemia
using Dates: DateTime, Day
using Test

using ReverseDiff
import Zygote
using Optim
using StatsPlots
using Distributions
using ParameterHandling
using ParameterHandling: value
using UnPack
using TransformVariables
using LogDensityProblems
using AdvancedHMC
using DynamicHMC

# @testset "Epidemia.jl" begin
struct RandomWalkProblem2{Ty, Tx, Trt, Tσ}
    y::Ty
    x::Tx
    epirt::Trt
    σ::Tσ
end

p = let
    d = dates = DateTime("2020-04-01"):Day(1):DateTime("2021-01-01") |> collect
    σ = 1e-6
    T = 100
    x = 1:length(d)
    y = sin.(2π/T*x) + rand(Normal(0,σ), length(x))

    epirt = RandomWalkModel(
        d,
        r0_prior = Normal(0.,0.1),
        σ_rw_prior = InverseGamma(3., .01),
        link = identity,
        invlink = identity
    )

    RandomWalkProblem2( y, d, epirt, σ )
end

plot(p.x, p.y)

## ===========================================================================

function (p::RandomWalkProblem2)(θ)
    @unpack y, epirt, σ = p
    @unpack r0, rw = θ

    ℓ  = 0.
    ℓ += logpdf(epirt, θ)
    ŷ = vcat(r0, rw)
    ℓ += logpdf(MvNormal(ŷ, σ), y)
    return ℓ
end
Distributions.logpdf(p::RandomWalkProblem2, θ) = p(θ)
## ===========================================================================
# Estimate parameters with TransformVariables + LogDensityProblems + Optim

function problem_transformation(p::RandomWalkProblem2)
    as(( σ_rw = asℝ₊, r0 = asℝ, rw = as(Array, length(p.y)-1)))
end

t = problem_transformation(p)
θ = rand(epirt)
x = TransformVariables.inverse(t, θ)
P = TransformedLogDensity(t, p)

∇P = ADgradient(Val(:Zygote), P)
loss(x) = -LogDensityProblems.logdensity(P, x)
grad(x) = -LogDensityProblems.logdensity_and_gradient(∇P, x)[2]


opt = optimize(loss, grad, x0; method=LBFGS(), inplace=false)

x = opt.minimizer
θ = TransformVariables.transform(t, x)
ŷ = vcat(vcat(θ.r0, θ.rw))
plot!(p.x, ŷ)

## ===========================================================================
# Estimate parameters with ParameterHandling + DynamicHMC
using Random
results = mcmc_with_warmup(Random.GLOBAL_RNG, ∇P, 1000; reporter = ProgressMeterReport())

θs = transform.(t, results.chain)
plot( mean( getindex.(θs, :rw) ) )
@show DynamicHMC.Diagnostics.summarize_tree_statistics(results.tree_statistics)
## ===========================================================================
# Estimate parameters with ParameterHandling + Optim

epirt_params = rand(epirt)
t = (
    σ_rw = positive(epirt_params.σ_rw),
    r0   = epirt_params.r0,
    rw   = epirt_params.rw
)
x0, unflatten = flatten( t )
unpack(x) = value(unflatten(x))
θ0 = unpack(x0)

loss(x) = -logpdf(p, unpack(x))
grad(x) = only(Zygote.gradient( loss, x ))

opt = optimize(loss, grad, x0; method=LBFGS(), inplace=false)

x = opt.minimizer
θ = unpack(x)
ŷ = vcat(vcat(θ.r0, θ.rw))
plot!(p.x, ŷ)

## ===========================================================================
# Estimate parameters with ParameterHandling + AdvancedHMC

Distributions.logpdf(p::RandomWalkProblem2, x::AbstractArray) =
    Distributions.logpdf(p, unpack(x))

function _logdensity_getter(p, x)
    logdensity(x) = logpdf(p, x)
end
logdensity = _logdensity_getter(p, x)

function _logdensity_and_gradient_getter(p, x)
    function logdensity_and_gradient(x)
        logdensity, back = Zygote.pullback(x->logpdf(p, x), x)
        ∂θ = first(back(1.0))
        return logdensity, ∂θ
    end
end

logdensity_and_gradient = _logdensity_and_gradient_getter(p, x)

# Sampling parameter settings
n_samples, n_adapts = 2000, 1000

# Define metric space, Hamiltonian, sampling method and adaptor
metric      = DiagEuclideanMetric( length(x) )
hamiltonian = Hamiltonian(metric, logdensity, logdensity_and_gradient)
initial_ϵ   = find_good_stepsize(hamiltonian, x0)
integrator  = Leapfrog(initial_ϵ)
proposal    = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
adaptor     = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

# Perform inference.
samples, stats = sample(hamiltonian, proposal, x0, n_samples, adaptor, n_adapts; progress=true)
θs = unpack.(samples)
plot( mean( getindex.(θs, :rw) ) )

## Plotting / inspection utilities from MCMCChains
using MCMCChains
using OrderedCollections

(t,_) = iterate(ts)
"Converts a vector of named tuples to a tuple of vectors."
function _vectup2tupvec(ts::AbstractVector{<:NamedTuple})
    ks = keys(first(ts))
    t  = first(ts)
    dic= OrderedDict()
    for t in ts
        for (k,v) in zip(keys(t), values(t))
            if isa(v, Number)
                if k in keys(dic)
                    push!(dic[k], v)
                else
                    dic[k] = [v]
                end
            else
                for (i, x) in enumerate(v)
                    kx = Symbol("$k[$i]")
                    if kx in keys(dic)
                        push!(dic[kx], x)
                    else
                        dic[kx] = [x]
                    end
                end
            end
        end
    end
    return (; dic...)
end

function _tupvec2chain(tv)
    nsamples = length(first(tv))
    nparams  = length(tv)
    ks = keys(tv)
    d = Array{Float64}(undef, nsamples, nparams, 1)
    n = Vector{Symbol}(undef, nparams)
    for (i, k) in enumerate(ks)
        d[:, i, 1] = tv[k]
        n[i] = k
    end
    MCMCChains.Chains(d, n)
end

function samples2chain(s)
    tv = vectup2tupvec(s)
    return tupvec2chain(tv)
end

n = filter( x->!occursin(r"\[", x), String.(names(ch)))
p = plot(ch[n])



## =========================================================================
# integrate into Turing

# using Turing
# @model function turingmodel(p)
#     θ ~ p
#     Turing.@addlogprob! logpdf(p, θ)
# end
#
# m = turingmodel(p)
# m()
