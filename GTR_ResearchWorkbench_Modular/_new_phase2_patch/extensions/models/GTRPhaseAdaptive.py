"""As GTRPhaseInnovation, with a small past-observed agreement-conditioned gain."""
from .GTRPhaseCore import PhaseInnovationModel

class Model(PhaseInnovationModel):
    def __init__(self, configs):
        super().__init__(configs, adaptive=True)
