"""Core-preserving, phase-calibrated innovation forecast correction."""
from .GTRPhaseCore import PhaseInnovationModel

class Model(PhaseInnovationModel):
    def __init__(self, configs):
        super().__init__(configs, adaptive=False)
