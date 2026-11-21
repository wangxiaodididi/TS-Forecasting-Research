"""Controlled replacement for TimeMixer's temporal prediction layers.

No dependencies on any other research workspace. Each head's base Linear uses
exactly the original shape. A small zero-initialized low-rank residual supplies
additional nonlinear horizon mapping capacity, without modifying PDM or FMM.
"""
import torch
from torch import nn
from torch.nn import functional as F

class ResidualTemporalProjection(nn.Module):
    def __init__(self, base:nn.Linear, rank:int=8, mode:str='nonlinear', scale:float=0.1):
        super().__init__()
        if not isinstance(base,nn.Linear):raise TypeError('Need original TimeMixer nn.Linear temporal head')
        if mode not in ('linear','nonlinear'):raise ValueError('mode must be linear or nonlinear')
        if rank < 1:raise ValueError('rank >=1')
        self.base=base
        self.reduce=nn.Linear(base.in_features,rank,bias=False)
        self.expand=nn.Linear(rank,base.out_features,bias=False)
        self.mode=mode;self.scale=float(scale)
        nn.init.normal_(self.reduce.weight,mean=0.,std=0.02)
        nn.init.zeros_(self.expand.weight)
    def forward(self,x):
        z=self.reduce(x)
        if self.mode=='nonlinear':z=F.gelu(z)
        return self.base(x)+self.scale*self.expand(z)
