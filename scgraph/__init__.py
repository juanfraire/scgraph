"""scgraph: stochastic contact graphs and SLA-aware DTN routing under uncertainty.

Companion library for the paper "Stochastic Contact Graphs: Rethinking the
Contact Plan for the 6G Space-Air-Ground Continuum" (IEEE OJ-COMS).
Equation numbers in docstrings refer to main.tex.
"""

from scgraph.contact import DetContact, StochasticContact
from scgraph.graph import LatentModel, Realization, StochasticContactGraph

__all__ = [
    "DetContact",
    "StochasticContact",
    "StochasticContactGraph",
    "Realization",
    "LatentModel",
]
