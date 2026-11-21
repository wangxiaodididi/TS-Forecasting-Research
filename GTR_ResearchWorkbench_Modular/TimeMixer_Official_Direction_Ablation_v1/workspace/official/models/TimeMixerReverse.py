"""Official TimeMixer PDM direction counterfactual.

This subclass uses the complete official Model.forward() and all original
operators/parameter tensors. The only intervention is to exchange which
component (seasonal/trend) receives each original direction-specific module.
For identical RNG seed, all trainable tensors are initialized identically to
the official model (up to the two module-path names).
"""
from models import TimeMixer


class Model(TimeMixer.Model):
    def __init__(self, configs):
        super().__init__(configs)
        for block in self.pdm_blocks:
            original_season_bottom_up = block.mixing_multi_scale_season
            original_trend_top_down = block.mixing_multi_scale_trend
            # In-place swap: no extra parameter initialization, same PDM/FMM.
            block.mixing_multi_scale_season = original_trend_top_down
            block.mixing_multi_scale_trend = original_season_bottom_up
