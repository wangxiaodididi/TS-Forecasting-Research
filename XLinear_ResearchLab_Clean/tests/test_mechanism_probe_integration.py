"""Mock 官方网络与数据提供器，检查诊断加载/反事实/门控 hooks 全链路。"""
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from src.models.registry import build_variant_model
from src.training.mechanism_probe import evaluate, summarize

class Gate(nn.Module):
    def __init__(self,dim,hidden):
        super().__init__()
        self.weight=nn.Sequential(nn.Linear(dim,hidden),nn.ReLU(),nn.Linear(hidden,dim),nn.Sigmoid())
    def forward(self,x): return x*self.weight(x)

class Backbone(nn.Module):
    def __init__(self,d,c,seq):
        super().__init__()
        self.d_model=d;self.channel=c
        self.projection=nn.Sequential(nn.Linear(seq,d),nn.Dropout(0))
        self.glob_token=nn.Parameter(torch.ones(1,c,d))
        self.en_attention=Gate(2*d,16)
        self.ex_attention=Gate(2*c,7)
    def forward(self,x):
        b,c,l=x.shape;d=self.d_model
        emb=self.projection(x)
        en=self.en_attention(torch.cat((emb,self.glob_token.expand(b,-1,-1)),dim=-1))
        endogenous=en[:,:,:d];glob_t=en[:,:,d:]
        cross=self.ex_attention(torch.cat((emb,glob_t),dim=1).transpose(1,2))
        glob=cross[:,:,c:].transpose(1,2)
        return torch.cat((endogenous,glob),dim=-1)

class Model(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.norm=config.usenorm
        self.backbone=Backbone(config.d_model,config.enc_in,config.seq_len)
        self.head=nn.Sequential(nn.Dropout(config.head_dropout),nn.Linear(config.d_model*2,config.pred_len))
    def forward(self,x):
        if self.norm:
            mu=x.mean(1,keepdim=True).detach();x=x-mu
            std=torch.sqrt(x.var(1,keepdim=True,unbiased=False)+1e-5);x=x/std
        y=self.head(self.backbone(x.permute(0,2,1))).permute(0,2,1)
        if self.norm:y=y*std[:,:1,:]+mu[:,:1,:]
        return y

class DS(Dataset):
    def __init__(self):
        rng=np.random.default_rng(22)
        self.x=rng.normal(size=(8,96,7)).astype('float64')
        self.y=rng.normal(size=(8,144,7)).astype('float64')
    def __len__(self):return len(self.x)
    def __getitem__(self,i):return self.x[i],self.y[i],np.zeros((96,1)),np.zeros((144,1))

def main():
    outer=types.ModuleType('models');outer.XLinear=types.SimpleNamespace(Model=Model)
    data=types.ModuleType('data_provider');factory=types.ModuleType('data_provider.data_factory')
    factory.data_provider=lambda options,split:(DS(),DataLoader(DS(),batch_size=4))
    save={k:sys.modules.get(k) for k in ('models','data_provider','data_provider.data_factory')}
    sys.modules['models']=outer;sys.modules['data_provider']=data;sys.modules['data_provider.data_factory']=factory
    try:
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            p=dict(d_model=8,channels=7,t_ff=16,c_ff=7,head_dropout=.2,embed_dropout=0,t_dropout=0,c_dropout=0,batch_size=4,data='ETTh1',file='ETTh1.csv')
            spec=dict(profile=p,horizon=96,dataset='ETTh1')
            options=types.SimpleNamespace(seq_len=96,pred_len=96,label_len=48,features='M',target='OT',d_model=8,enc_in=7,t_ff=16,c_ff=7,usenorm=1,embed_dropout=0,head_dropout=.2,t_dropout=0,c_dropout=0)
            full=[]
            for variant in ('official','dyn_token','stat_token'):
                spec['variant']=variant
                torch.manual_seed(2025)
                klass=build_variant_model(Model,variant,.25)
                model=klass(options)
                if variant!='official':
                    with torch.no_grad(): model.backbone.gain_logit.fill_(-.16)
                model.eval()
                folder=root/variant
                d=folder/'checkpoints'/'job';d.mkdir(parents=True)
                torch.save(model.state_dict(),d/'checkpoint.pth')
                if variant=='official':
                    ds=DS();dl=DataLoader(ds,batch_size=4,shuffle=False,drop_last=True)
                    with torch.no_grad():pred=np.concatenate([model(x.float()).numpy() for x,*_ in dl])
                    cache=folder/'workspace'/'results'/'job';cache.mkdir(parents=True)
                    np.save(cache/'pred.npy',pred)
                rec=evaluate(folder,spec.copy(),'test',.25,torch.device('cpu'),0,1.) if variant=='official' else evaluate(folder,spec.copy(),'val',.25,torch.device('cpu'),0,1.)
                assert len(rec)==8
                if variant!='official':
                    assert any(x['prediction_abs_change']>0 for x in rec)
                    assert all(np.isfinite(x['token_relative_norm']) for x in rec)
                full.extend(rec)
            assert len(summarize(full))==3
            print('PASS: stub checkpoint -> deterministic data -> gate hooks -> token off -> prediction parity')
    finally:
        for k,v in save.items():
            if v is None:sys.modules.pop(k,None)
            else:sys.modules[k]=v

if __name__=='__main__':main()
