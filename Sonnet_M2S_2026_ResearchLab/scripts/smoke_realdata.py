#!/usr/bin/env python3
"""Few TRAIN batches + one VALIDATION batch with official Darts loader/model; no test selection."""
import argparse, json, time
from types import SimpleNamespace
import torch

SPLITS = {'energy': (21169, 5136, 8759), 'etth1': (8640, 2880, 2880)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=SPLITS, default='energy')
    parser.add_argument('--steps', type=int, default=4)
    args = parser.parse_args()
    from sonnet.mts_model.models.Sonnet import Model
    from sonnet.data.dartsDataloader import CustomDartsDataModule
    torch.manual_seed(2026)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    H, SEQ = ((24,24) if args.dataset=='energy' else (96,336))
    data = CustomDartsDataModule(
        args.dataset, batch_size=4, seq_length=SEQ, pred_length=H,
        task='MB', do_dataset_split_ratio=False, dataset_split_numeric=SPLITS[args.dataset],
        scale=True,
    )
    data.setup()  # Darts fetches public original data automatically if not cached
    nvars=data.data_train.metadata['num_instances']
    cfg=SimpleNamespace(
        enc_in=nvars,c_out=1,seq_len=SEQ,pred_len=H,d_model=16,n_atoms=4,
        revin=0,alpha=0.5,downsample_factors=[1],data_group='darts',
    )
    model=Model(cfg,datamodule=data,lr=0.001).cpu()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    print(json.dumps({'dataset':args.dataset,'mode':'smoke_only','nvars':nvars,
                      'input_shape':[4,SEQ,nvars],'output_shape':[4,H,1],
                      'n_train_windows':len(data.data_train),'n_val_windows':len(data.data_val),
                      'n_test_windows':len(data.data_test)}))
    start=time.perf_counter()
    model.train()
    for i, (x,y) in zip(range(args.steps),data.train_dataloader()):
        optimizer.zero_grad(set_to_none=True)
        output=model(x)
        assert output.shape == y.shape, (output.shape,y.shape)
        loss=torch.nn.functional.mse_loss(output,y)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        print('train_step',i+1,'mse',float(loss.detach()))
    model.eval()
    with torch.no_grad():
        x,y=next(iter(data.val_dataloader()))
        pred=model(x)
        mae_h=float((pred[:,-1,:]-y[:,-1,:]).abs().mean())
        mae_seq=float((pred-y).abs().mean())
    print(json.dumps({'val_horizon_mae_smoke':mae_h,'val_seq_mae_smoke':mae_seq,
                      'seconds':round(time.perf_counter()-start,2),
                      'note':'Few-step smoke results ARE NOT paper-level reproduction'}))


if __name__=='__main__':
    main()
