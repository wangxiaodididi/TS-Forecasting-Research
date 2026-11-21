"""Official TimeMixer Model subclass: only temporal forecast heads are modified."""
import os
import torch.nn as nn
from models import TimeMixer
from models.residual_head import ResidualTemporalProjection

class Model(TimeMixer.Model):
    def __init__(self, configs):
        super().__init__(configs)
        rank=int(os.environ.get('TIMEMIXER_HEAD_RANK','8'))
        mode=os.environ.get('TIMEMIXER_HEAD_MODE','nonlinear')
        scale=float(os.environ.get('TIMEMIXER_HEAD_SCALE','0.1'))
        self.predict_layers=nn.ModuleList([
            ResidualTemporalProjection(module,rank=rank,mode=mode,scale=scale)
            for module in self.predict_layers
        ])
