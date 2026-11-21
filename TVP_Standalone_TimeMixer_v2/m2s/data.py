"""Fixed-split, historical-input-only data for M-to-S forecasting.
Original CSV contains real measured/quoted variables only, with target OT last.
No synthetic indicators, rolling features or future covariates.
"""
from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

CONFIGS = {
    'Hubei': (60, (1,5)),
    'Guangdong': (60, (1,5)),
    'CAISO': (168, (24,48)),
    'PGCB': (168, (24,48)),
    'NESO': (336, (48,96)),
    'OPSD': (168, (24,48)),
}

class Windows(Dataset):
    def __init__(self, scaled, windows, target_only):
        self.x=scaled
        self.w=windows
        self.target_only=target_only
    def __len__(self):return len(self.w)
    def __getitem__(self,i):
        a,b,c,d=self.w[i]
        x=self.x[a:b+1]
        y=self.x[c:d+1,-1:]
        if self.target_only:x=x[:,-1:]
        return torch.from_numpy(x),torch.from_numpy(y)

@dataclass
class ReadyData:
    name:str
    input_len:int
    horizon:int
    columns:list[str]
    dates:np.ndarray
    raw:np.ndarray
    scaled:np.ndarray
    windows:dict
    mean:np.ndarray
    std:np.ndarray
    missing_count:dict
    train_end:int

    def loader(self,split,batch_size=32,max_windows=0,seed=2024,target_only=False,workers=0):
        w=self.windows[split]
        if max_windows>0 and len(w)>max_windows:
            if split=='train':
                rng=np.random.default_rng(seed)
                ids=np.sort(rng.choice(len(w),size=max_windows,replace=False))
            else:
                ids=np.unique(np.linspace(0,len(w)-1,num=max_windows,dtype=int))
            w=w[ids]
        ds=Windows(self.scaled,w,target_only)
        return DataLoader(ds,batch_size=batch_size,shuffle=(split=='train'),num_workers=workers,drop_last=False)


def load(root:Path,name:str,horizon:int)->ReadyData:
    if name not in CONFIGS:raise ValueError('Unknown dataset '+str(name))
    length,hs=CONFIGS[name]
    if horizon not in hs:raise ValueError(f'{name}: horizon choices are {hs}')
    csv_path=root/'data'/f'{name}.csv'
    indices=root/'m2s'/'window_indices'/f'{name}_L{length}_H{horizon}.csv'
    df=pd.read_csv(csv_path)
    if df.columns[0]!='date' or df.columns[-1]!='OT' or len(df.columns)<3:
        raise ValueError(f'{csv_path}: require date, auxiliaries, OT (target last)')
    cols=list(df.columns[1:])
    if len(cols)!=len(set(cols)):raise ValueError('Duplicate feature names')
    raw=df[cols].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float64)
    if not np.isfinite(raw[:,-1]).all():raise ValueError(f'{name}: OT target has missing/inf values')
    raw[~np.isfinite(raw)] = np.nan
    idx=pd.read_csv(indices)
    windows={}
    for split in ('train','val','test'):
        part=idx[idx['split']==split]
        w=part[['x_start','x_end','y_start','y_end']].to_numpy(dtype=np.int64)
        if len(w)==0:raise ValueError(f'{name}: split {split} has no windows')
        if ((w[:,1]-w[:,0]+1)!=length).any() or ((w[:,3]-w[:,2]+1)!=horizon).any():
            raise ValueError(f'{name}: incorrect window lengths in {split}')
        if ((w[:,2]-w[:,1])!=1).any() or (w<0).any() or (w[:,3]>=len(raw)).any():
            raise ValueError(f'{name}: invalid or disconnected windows in {split}')
        windows[split]=w
    train_end=int(windows['train'][:,3].max())
    if not (train_end < int(windows['val'][:,2].min()) and
            int(windows['val'][:,3].max()) < int(windows['test'][:,2].min())):
        raise ValueError(f'{name}: train/val/test target periods overlap')
    tr=raw[:train_end+1]
    observed=np.isfinite(tr)
    num=observed.sum(axis=0)
    sums=np.nansum(tr,axis=0)
    mean=np.divide(sums,num,out=np.zeros(raw.shape[1],dtype=np.float64),where=num>0)
    differences=np.where(observed,tr-mean,0.0)
    std=np.sqrt(np.divide(np.sum(differences* differences,axis=0),num,
                          out=np.zeros(raw.shape[1],dtype=np.float64),where=num>0))
    std=np.where(np.isfinite(std)&(std>1e-8),std,1.0)
    # Mean imputation solely at model input time. No change to original CSV.
    scaled=((np.where(np.isfinite(raw),raw,mean)-mean)/std).astype(np.float32)
    if not np.isfinite(scaled).all():raise ValueError(f'{name}: nonfinite after scaling')
    missing={col:int(np.isnan(raw[:,j]).sum()) for j,col in enumerate(cols)}
    return ReadyData(name,length,horizon,cols,df['date'].astype(str).to_numpy(),raw,
                    scaled,windows,mean,std,missing,train_end)
