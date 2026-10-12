"""Diffusion primitives promoted from MiniWorld.

base/             — abstract Diffuser / DiffusionScheduler / DiffusionSolver
edm/              — default Euclidean EDM (eps-prediction) with Karras/AF3
preconditioning (c_noise = log(sigma/sigma_data)/4)
decoupled_xpred/  — decoupled x-prediction (VE, separate R/T noise); the
diffuser used by the MiniWorld folding model + inference
"""

from miniworld_engine.architecture.diffusion.base import (
    AtomChainMap,
    Diffuser,
    DiffusionScheduler,
    DiffusionSolver,
    ModelFn,
)
from miniworld_engine.architecture.diffusion.decoupled_xpred import (
    DecoupledXPredScheduler,
    XPredDecoupledDiffuser,
    XPredDecoupledSolver,
)
from miniworld_engine.architecture.diffusion.edm import (
    AF3Solver,
    EDMScheduler,
    EuclideanDiffuser,
)
from miniworld_engine.architecture.diffusion.edm.preconditioned import (
    PreconditionedEulerSolver,
)
from miniworld_engine.architecture.diffusion.edm.sampling import (
    EulerSampler,
    PreconditionedDenoiser,
)
from miniworld_engine.architecture.diffusion.edm.schedules import (
    LogNormalNoise,
    PowerLawSchedule,
)

__all__ = [
    "AF3Solver",
    "AtomChainMap",
    "DecoupledXPredScheduler",
    "Diffuser",
    "DiffusionScheduler",
    "DiffusionSolver",
    "EDMScheduler",
    "EuclideanDiffuser",
    "EulerSampler",
    "LogNormalNoise",
    "ModelFn",
    "PowerLawSchedule",
    "PreconditionedDenoiser",
    "PreconditionedEulerSolver",
    "XPredDecoupledDiffuser",
    "XPredDecoupledSolver",
]
