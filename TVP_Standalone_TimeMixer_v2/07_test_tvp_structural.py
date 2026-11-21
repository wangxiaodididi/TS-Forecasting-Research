#!/usr/bin/env python3
"""Small CPU sanity tests, including label-availability non-leakage checks."""
import tempfile,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from importlib import import_module
m=import_module('06_tvp_structural_trials')

def main():
    rng=np.random.default_rng(46);H=5;S=4;C=3;N=80
    b=rng.normal(size=(N,H,C))*.2;p=rng.normal(size=(N,H,C,S))*.1
    y=b+np.einsum('nhcs,s->nhc',p,np.array([.1,-.2,.3,0.])) + rng.normal(size=b.shape)*.03
    init=m.ridge_init(b[:30],p[:30],y[:30]);q=np.array([.001,.001,.001,.001]);active=(0,2)
    z,coef,updates=m.online_predict(b,p,y,init,active,q,H)
    assert z.shape==b.shape and updates==N-H
    assert np.allclose(coef[:,1],init[1]) and np.allclose(coef[:,3],init[3])
    # Change a test label at origin j; forecasts through i < j+H must not change.
    changed=y.copy(); j=12;changed[j]+=99
    z2,_,_=m.online_predict(b,p,changed,init,active,q,H)
    assert np.array_equal(z[:j+H],z2[:j+H]),'Early forecasts changed from future label'
    assert not np.allclose(z[j+H:],z2[j+H:]),'No matured label feedback found'
    # A source with zero q remains online-adjustable (but all inactive sources stay fixed).
    # The full pipeline must be runnable on synthetic val/test arrays.
    with tempfile.TemporaryDirectory() as tmp:
        val=Path(tmp)/'val.npz';test=Path(tmp)/'test.npz'
        b2=rng.normal(size=(40,H,C))*.2;p2=rng.normal(size=(40,H,C,S))*.1;y2=b2+rng.normal(size=b2.shape)*.02
        np.savez(val,base=b,parts=p,y=y);np.savez(test,base=b2,parts=p2,y=y2)
        rows,manifest=m.run(val,test,Path(tmp)/'out',base_q=.001)
        assert len(rows)==4 and manifest['update_counts']['tvp_selective']==len(b2)-H
        assert (Path(tmp)/'out'/'structural_metrics.csv').exists()
    print('PASS: shapes, frozen coefficients, H-delay non-leakage, synthetic pipeline.')
if __name__=='__main__':main()
