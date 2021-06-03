"""
A modular package that allows users to define a hierachical model of epidemic
spreading from multiple building blocks:

1. EpiRt

2. EpiInf

3. EpiObs

"""
module Epidemia
__precompile__(false)

# import Random
# using Distributions
# using Dates: DateTime, Day
# using UnPack: @unpack
# import Bijectors

include("epirt.jl")

export
  RandomWalk,
  RandomWalkModel,
  ConstantPhaseRandomWalkModel,
  logpdf,
  rand


end # module
