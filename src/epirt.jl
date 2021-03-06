import Random
using Distributions
using Dates: DateTime, Day
using UnPack: @unpack
import Bijectors

abstract type EpiRt end
## ===========================================================================
# Random Walk

struct RandomWalkDistribution{Tn, Ts} <: ContinuousMultivariateDistribution
    n::Tn
    s::Ts
end

Random.rand(rng::Random.AbstractRNG, d::RandomWalkDistribution{Tn, Ts}) where {Tn, Ts} = begin
    @unpack n, s = d
    step_prior = Normal(zero(Ts), s)
    x = Vector{Ts}(undef, n)

    ######## random walk
    x[1] = rand(step_prior)
    for i in 2:n
        x[i] = x[i-1] + rand(step_prior)
    end
    return x
end

Distributions.logpdf(d::RandomWalkDistribution{Tn, Ts}, x::AbstractVector{T}) where {Tn, Ts, T} =
    logpdf( MvNormal( d.n-1, d.s ), diff(x) )# + logpdf( Normal( zero(Ts), x[1] ) )
Bijectors.bijector(d::RandomWalkDistribution) = Bijectors.Identity{1}()

## ===========================================================================
"""
random walk with variable step interval
"""
struct RWModel{Tno, Tnt, Tns, Tin, Tli, Til} <: EpiRt
    nobs        ::Tno
    ntotal      ::Tnt
    nsteps      ::Tns
    index       ::Tin
    link        ::Tli
    invlink     ::Til
end

function stepindex(n, rwstep)
    index = Vector{Int64}(undef, n)
    nsteps = 1
    for i in eachindex(index)
        index[i] = nsteps
        i % rwstep == 0 && ( nsteps += 1 )
    end
    index
end

RWModel(
    dates     ::AbstractVector;
    endobs    = dates[end],
    rwstep    ::Int = 1,
    link      = exp,
    invlink   = log,
) = begin

	#@assert all( dates[1]:Day(1):dates[end] .== dates )
	ntotal  = length(dates)
	nobs    = findfirst(==(endobs), dates)
    index   = stepindex(nobs-1, rwstep)
    nsteps  = length( unique( index ) )

	RWModel(
        nobs,
		ntotal,
		nsteps,
		index,
		link,
		invlink
	)
end

(epirt::RWModel{Tno, Tnt, Tns, Tin, Tli, Til})(
	θ, ::Val{false}
) where {Tno, Tnt, Tns, Tin, Tli, Til} = begin
    @unpack rw, r0 = θ
    @unpack nobs, link = epirt

    rt         = Vector{typeof(r0)}(undef, nobs)
    rt[1]      = r0
    rt[2:nobs] = link.(rw)
    return rt
end

(epirt::RWModel{Tno, Tnt, Tns, Tin, Tli, Til})(
	θ, ::Val{true}
) where {Tno, Tnt, Tns, Tin, Tli, Til} = begin
    @unpack rw, r0 = θ
    @unpack ntotal, nobs, link = epirt

    rt              = Vector{typeof(r0)}(undef, ntotal)
    rt[1]           = r0
    rt[2:nobs]      = link.(rw)
    rt[nobs+1:end] .= rt[nobs]
    return rt
end

## =============================================================================

struct RWModelDistribution2{Tr0, Tσ, Tbm} <: ContinuousMultivariateDistribution
	r0_prior   ::Tr0
	σ_rw_prior ::Tσ
	rwmodel::Tbm
end

# RWModelDistribution2(
# 	r0_prior   ::Distribution,
# 	σ_rw_prior ::Distribution,
# 	args...; kwargs...
# ) = begin
# 	RWModelDistribution2(
# 		r0_prior, σ_rw_prior,
# 		RWModel(args...; kwargs...)
# 	)
# end

Base.rand(rng::Random.AbstractRNG, epirt::RWModelDistribution2) = begin
	@unpack r0_prior, σ_rw_prior, rwmodel = epirt
    @unpack nsteps, invlink = rwmodel

    ######## draw from univariate prior
    σ_rw = rand(rng, σ_rw_prior )
    r0   = rand(rng, r0_prior )

    ######## draw from random walk prior
    rw_prior = RandomWalkDistribution(nsteps, σ_rw)
    rw = rand(rng, rw_prior)
    return (; σ_rw, r0, rw)
end

Distributions.logpdf(epirt::RWModelDistribution2, θ) = begin
    @unpack rw, r0, σ_rw = θ
    @unpack r0_prior, σ_rw_prior, rwmodel = epirt
	@unpack nsteps, invlink = rwmodel

    rw_prior   = RandomWalkDistribution(nsteps, σ_rw)
    step_prior = Normal(zero(σ_rw), σ_rw) #NOTE same prior as in RandomWalkDistribution
    step = first(rw) - invlink(r0)

    target  = logpdf(σ_rw_prior, σ_rw)
    target += logpdf(r0_prior, r0)
    target += logpdf(step_prior, step)
    target += logpdf(rw_prior, rw)
    return target
end

Bijectors.bijector(d::RWModelDistribution2) = begin
	@unpack r0_prior, σ_rw_prior = d
	bj_r0 = Bijectors.bijector(r0_prior)
	bj_σ  = Bijectors.bijector(σ_rw_prior)
	bj_rw = Bijectors.Identity{1}()
	return Bijectors.Stacked( (bj_r0, bj_σ, bj_rw) )
end


## ===========================================================================
"""
random walk with constant early phase
"""
struct ConstantPhaseRWModel{Tr0, Tr1, Tσ, Tnc, Tno, Tnt, Tns, Tin, Tli, Til} <: EpiRt
    r0_prior    ::Tr0
    r1_prior    ::Tr1
    σ_rw_prior  ::Tσ
    nconst      ::Tnc
    nobs        ::Tno
    ntotal      ::Tnt
    nsteps      ::Tns
    index       ::Tin
    link        ::Tli
    invlink     ::Til
end

ConstantPhaseRWModel(
    dates     ::Vector{DateTime},
    startrw   ::DateTime;
    endobs    ::DateTime = dates[end],
    r0_prior  ::ContinuousUnivariateDistribution = truncated(Normal(3.5, 1.), 0, Inf),
    r1_prior  ::ContinuousUnivariateDistribution = truncated(Normal(1.,0.5), 0, Inf),
    σ_rw_prior::ContinuousUnivariateDistribution = truncated(Normal(0.02,0.01), 0, Inf),
    rwstep    ::Int = 1,
    link = exp,
    invlink = log,
) = begin

    @assert all( dates[1]:Day(1):dates[end] .== dates )
    ntotal  = length(dates)
    nconst  = findfirst(==(startrw), dates)-1
    nobs    = findfirst(==(endobs), dates)
    index   = stepindex(nobs-nconst-1, rwstep)
    nsteps  = length( unique( index ) )

	ConstantPhaseRWModel(
		r0_prior,
		r1_prior,
		σ_rw_prior,
		nconst,
		nobs,
		ntotal,
		nsteps,
		index,
		link,
		invlink
	)
end

(epirt::ConstantPhaseRWModel{Tr0, Tr1, Tσ, Tnc, Tno, Tnt, Tns, Tin, Tli, Til})(
	θ, ::Val{false}
) where {Tr0, Tr1, Tσ, Tnc, Tno, Tnt, Tns, Tin, Tli, Til} = begin
    @unpack rw, r0, r1 = θ
    @unpack nconst, nobs, link = epirt

    rt = Vector{typeof(r0)}(undef, nobs)
    rt[1:nconst]     .= r0
    rt[nconst+1]      = r1
    rt[nconst+2:end]  = link.(rw)
    return rt
end

(epirt::ConstantPhaseRWModel{Tr0, Tr1, Tσ, Tnc, Tno, Tnt, Tns, Tin, Tli, Til})(
	θ, ::Val{true}
) where {Tr0, Tr1, Tσ, Tnc, Tno, Tnt, Tns, Tin, Tli, Til} = begin
    @unpack rw, r0, r1 = θ
    @unpack nconst, ntotal, nobs, link, index = epirt

    rt = Vector{typeof(r0)}(undef, ntotal)
    rt[1:nconst]      .= r0
    rt[nconst+1]       = r1
    rt[nconst+2:nobs]  = link.(rw[index])
    rt[nobs+1:end]    .= rt[nobs]
    return rt
end

Base.rand(rng::Random.AbstractRNG, epirt::ConstantPhaseRWModel) = begin
    @unpack r0_prior, r1_prior, σ_rw_prior, nsteps, invlink = epirt

    ######## draw from univariate prior
    σ_rw = rand(rng, σ_rw_prior )
    r0   = rand(rng, r0_prior )
    r1   = rand(rng, r1_prior )

    ######## draw from random walk prior
    rw0 = invlink(r1)
    rw_prior = RandomWalkDistribution(nsteps, σ_rw)
    rw = rand(rng, rw_prior )
    return (; σ_rw, r0, r1, rw)
end

Distributions.logpdf(epirt::ConstantPhaseRWModel, θ) = begin
    @unpack rw, r0, r1, σ_rw = θ
    @unpack r0_prior, r1_prior, σ_rw_prior, nsteps, invlink = epirt

    rw_prior   = RandomWalkDistribution(nsteps, σ_rw)
    step_prior = Normal(zero(σ_rw), σ_rw) #NOTE same prior as in RadomWalk
    step = rw[1] - invlink(r1)

    target  = logpdf(σ_rw_prior, σ_rw)
    target += logpdf(r0_prior, r0)
    target += logpdf(r1_prior, r1)
    target += logpdf(step_prior, step)
    target += logpdf(rw_prior, rw)
    return target
end

## ===========================================================================


# """
# # Arguments
#
# formula: An object of class formula which determines the linear predictor for the reproduction rates. The left hand side must take the form R(group, date), where group and date variables. group must be a factor vector indicating group membership (i.e. country, state, age cohort), and date must be a vector of class Date. This is syntactic sugar for the reproduction number in the given group at the given date.
#
# link: The link function. This must be either "identity", "log", or a call to scaled_logit.
#
# center: If TRUE then the covariates for the regression are centered to have mean zero. All of the priors are then interpreted as prior on the centered covariates. Defaults to FALSE.
#
# prior: Same as in stan_glm. In addition to the rstanarm provided priors, a shifted_gamma can be used. Note: If autoscale=TRUE in the call to the prior distribution then automatic rescaling of the prior may take place.
#
# prior_intercept: Same as in stan_glm. Prior for the regression intercept (if it exists).
#
# prior_covariance: Same as in stan_glmer. Only used if the formula argument specifies random effects.
#
# # Details
#
# epirt has a formula argument which defines the linear predictor, an argument link defining the link function, and additional arguments to specify priors on parameters making up the linear predictor.
# A general R formula gives a symbolic description of a model. It takes the form y ~ model, where y is the response and model is a collection of terms separated by the + operator. model fully defines a linear predictor used to predict y. In this case, the “response” being modeled are reproduction numbers which are unobserved. epirt therefore requires that the left hand side of the formula takes the form R(group, date), where group and date refer to variables representing the region and date respectively. The right hand side can consist of fixed effects, random effects, and autocorrelation terms.
#
# """
# function epirt(
#   formula,
#   link = "log",
#   center = FALSE,
#   prior = Normal(1, 0.5),
#   prior_intercept = Normal(1., 0.5),
#   prior_covariance = rstanarm::decov(scale = 0.5)
# ) end
#
#
# """
# A call to rw can be used in the 'formula' argument of epim, allowing random walks for the reproduction number.
#
# time:	An optional name defining the random walk time periods for each date and group. This must be a column name found in the data argument to epim. If not specified, determined by the dates column implied by the formula argument to epim is used.
#
# gr: Same as for time, except this defines the grouping to use for the random walks. A separate walk is defined for each group. If not specified a common random walk is used for all groups.
#
# prior_scale: The steps of the walks are independent zero mean normal with an unknown scale hyperparameter. This scale is given a half-normal prior. prior_scale sets the scale parameter of this hyperprior.
# """
# rw(time = "date", gr = "country", prior_scale = 0.2) = RWModel(time, gr, prior_scale)

# epirt(
#   formula,
#   link = "log",
#   prior = Normal(1, 0.5),
#   prior_intercept = Normal(1., 0.5),
#   prior_covariance = rstanarm::decov(scale = 0.5),
#   ...
# )
