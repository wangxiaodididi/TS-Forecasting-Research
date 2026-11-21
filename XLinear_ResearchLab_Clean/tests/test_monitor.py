"""无需 GPU 的真实工程自测：指标解析 + 只读训练日志注入。运行 python -m tests.test_monitor"""
from __future__ import annotations
import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from src.training.telemetry import MetricParser
from src.training.official_entry import instrument_train_function

class FakeLoss:
    def __init__(self,n):self.n=n
    def item(self):return float(self.n)

class FakeTrainer:
    def train(self,setting):
        history=[]
        train_steps=4
        for epoch in range(2):
            train_loss=[]
            for i in range(train_steps):
                loss=FakeLoss((epoch+1)*10+i)
                train_loss.append(loss.item())
            history.extend(train_loss)
        return history

def test():
    original=FakeTrainer.train
    patched=instrument_train_function(original,2)
    a=FakeTrainer();b=FakeTrainer()
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):
        actual=patched(b,'example')
    assert actual==original(a,'example')
    assert stream.getvalue().count('LAB_BATCH')==4
    with TemporaryDirectory() as temporary:
        parser=MetricParser(Path(temporary))
        for line in stream.getvalue().splitlines():parser.consume(line)
        parser.consume('Epoch: 1, Steps: 4 | Train Loss: 0.2345000 Vali Loss: 0.3456000 Test Loss: 0.4000000')
        parser.close()
        import pandas as pd
        bt=pd.read_csv(Path(temporary)/'metrics'/'batches.csv')
        ep=pd.read_csv(Path(temporary)/'metrics'/'epochs.csv')
        assert len(bt)==4 and len(ep)==1
        assert ep.loc[0,'train_loss']==.2345
        assert ep.loc[0,'val_loss']==.3456
        assert bt.iloc[-1]['epoch']==2
        assert bt.iloc[-1]['batch']==4
    print('PASS: 额外Batch日志不改变函数结果；实时Epoch/Batch指标解析正确。')

if __name__=='__main__':test()
