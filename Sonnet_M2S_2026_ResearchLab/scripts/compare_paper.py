#!/usr/bin/env python3
"""Compare user-supplied *tar_mae* from unmodified official run to paper Table 1.
No attempt is made to parse Lightning's variable table formatting automatically.
"""
import argparse,csv,json,pathlib
PAPER={('energy',24):0.3621,('energy',48):0.4120,('energy',72):0.4036,('energy',168):0.3980,
       ('etth1',96):0.2145,('etth1',192):0.2330,('etth1',336):0.2392,('etth1',720):0.2768}
ROOT=pathlib.Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['energy','etth1'],required=True)
    p.add_argument('--h',type=int,required=True)
    p.add_argument('--tar-mae',type=float,required=True,help='Official Lightning tar_mae, NOT seq_mae')
    p.add_argument('--run',default='user_run')
    args=p.parse_args()
    key=(args.dataset,args.h)
    if key not in PAPER: p.error('Unsupported original paper Table 1 dataset/horizon')
    val=PAPER[key]
    data={'run':args.run,'dataset':args.dataset,'H':args.h,'paper_tar_mae':val,
          'observed_tar_mae':args.tar_mae,'difference':round(args.tar_mae-val,7),
          'relative_difference_percent':round((args.tar_mae/val-1)*100,3),
          'metric':'tar_mae','status':'observed vs paper; no automatic agreement threshold'}
    path=ROOT/'results'/'paper_alignment.csv'
    first=not path.exists()
    with path.open('a',newline='',encoding='utf-8') as fp:
        w=csv.DictWriter(fp,fieldnames=data.keys())
        if first:w.writeheader()
        w.writerow(data)
    print(json.dumps(data,indent=2))
    print('Appended to',path)


if __name__=='__main__':main()
