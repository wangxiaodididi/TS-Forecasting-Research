"""Streamlit UI. All controls run on a local, SSH-forwarded server, never public port."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import streamlit as st
from lab.paths import ROOT,RUNS,VENDOR,DATA
from lab.config import MODEL_FILES, DATASETS
from lab.cli import start,stop

st.set_page_config(page_title='GTR Research Workbench',page_icon='📊',layout='wide')
st.title('GTR 预测模型研究工作台')
st.caption('ICLR 2026 · pinned official baseline · signal correction · read-only monitoring')

with st.sidebar:
    st.subheader('新建训练任务')
    dataset=st.selectbox('数据集',list(DATASETS),index=1)
    model=st.selectbox('模型',list(MODEL_FILES))
    horizon=st.selectbox('预测长度', [96,192,336,720])
    seed=st.number_input('随机种子',min_value=1,max_value=999999,value=2024)
    mode=st.radio('训练模式',['smoke','paper'],format_func=lambda m:'CPU 小样本检查' if m=='smoke' else '论文同配置训练')
    if st.button('启动后台训练',type='primary',use_container_width=True):
        try:
            rid=start(dataset,model,horizon,int(seed),mode,detach=True)
            st.success(f'已启动 {rid}')
        except Exception as exc:st.error(str(exc))
    st.divider()
    st.caption('原始基线不接受自定义超参数。扩展模型的参数请修改 configs/experiments/*.yaml。')
    st.caption('官方源码：'+('已就绪' if (VENDOR/'run.py').exists() else '尚未安装'))

RUNS.mkdir(exist_ok=True)
ids=[p.name for p in sorted(RUNS.iterdir(),reverse=True) if p.is_dir() and (p/'status.json').exists()]
if not ids:
    st.info('尚无任务。先在终端安装原始 GTR 源码，然后从左侧启动 CPU 小样本。')
    st.code('python -m lab.cli install-official\npython -m lab.cli launch --dataset ETTm1 --model GTR --horizon 96 --mode smoke',language='bash')
    st.stop()
selected=st.selectbox('选择实验记录',ids)
path=RUNS/selected

def read_status():
    try:return json.loads((path/'status.json').read_text(encoding='utf-8'))
    except Exception:return {}

def read_csv(name):
    p=path/name
    if not p.exists():return pd.DataFrame()
    try:return pd.read_csv(p)
    except (OSError,pd.errors.EmptyDataError):return pd.DataFrame()

def read_jsonl(name):
    p=path/name
    if not p.exists():return pd.DataFrame()
    try:
        records=[]
        for line in p.read_text(encoding='utf-8').splitlines():
            try:records.append(json.loads(line))
            except json.JSONDecodeError:pass
        return pd.DataFrame.from_records(records)
    except OSError:return pd.DataFrame()

@st.fragment(run_every='3s')
def live_monitor():
    q=read_status()
    a,b,c,d=st.columns(4)
    a.metric('训练状态',q.get('state','unknown'))
    b.metric('当前 Epoch',f"{q.get('epoch',0)} / {q.get('expected_epochs','?')}")
    c.metric('最新 Train Loss',f"{q['train_loss']:.5f}" if isinstance(q.get('train_loss'),(float,int)) else '—')
    d.metric('最新 Val Loss',f"{q['val_loss']:.5f}" if isinstance(q.get('val_loss'),(float,int)) else '—')
    try:
        expected=int(q.get('expected_epochs') or 0)
        if expected>0:st.progress(min(1.0,int(q.get('epoch') or 0)/expected),text=f"Epoch {q.get('epoch',0)} / {expected}（Early Stopping 可能提前结束）")
    except Exception:pass
    logs=read_csv('loss.csv')
    if not logs.empty:
        st.subheader('实时训练与验证损失（真实 Epoch 输出）')
        st.line_chart(logs.set_index('epoch')[['train_loss','val_loss']],height=320)
        best=logs.loc[logs['val_loss'].idxmin()]
        col1,col2,col3=st.columns(3)
        col1.metric('最佳验证 Epoch',int(best.epoch))
        col2.metric('最低验证 Loss',f'{best.val_loss:.6f}')
        col3.metric('当前验证 − 训练',f"{(logs.iloc[-1].val_loss-logs.iloc[-1].train_loss):+.6f}")
        st.caption('验证 Loss 仅在每个 Epoch 完成后产生；原论文训练器也打印测试 Loss，但工作台不把测试 Loss 用于选模。')
    else: st.info('正在等待第一个 Epoch 结束；下方仍可查看实时日志和 GPU。')
    events=read_jsonl('events.jsonl')
    if not events.empty and 'type' in events.columns:
        batches=events[events.type=='batch']
        if not batches.empty:
            st.subheader('训练 Batch Loss（原作者每 100 批打印一次）')
            st.line_chart(batches.reset_index(drop=True)[['batch_loss']],height=220)
    g=read_jsonl('gpu.jsonl')
    if not g.empty and 'util_gpu_pct' in g.columns:
        st.subheader('GPU / 显存')
        c1,c2=st.columns(2)
        with c1:st.line_chart(g[['util_gpu_pct']],height=200)
        with c2:st.line_chart(g[['memory_used_mb']],height=200)
    st.subheader('实时终端日志')
    log=path/'console.log'
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0,f.seek(0,2)-14000));tail=f.read().decode('utf-8','replace')
        st.code(tail[-12000:] or '等待日志...',language='text')
    else:st.caption('任务准备中')
    if q.get('state') in ('queued','running') and st.button('停止本次训练',key='stop_'+selected):
        try:stop(selected);st.warning('已发送停止信号。')
        except Exception as exc:st.error(str(exc))

live_monitor()
t1,t2,t3=st.tabs(['实验配置与可复现性','解释与诊断指南','历史实验对比'])
with t1:
    c=path/'config.json';p=path/'provenance.json'
    if c.is_file():st.json(json.loads(c.read_text(encoding='utf-8')))
    if p.is_file():st.json(json.loads(p.read_text(encoding='utf-8')))
    env=path/'environment.json'
    if env.is_file():
        st.subheader('实验环境与数据版本')
        st.json(json.loads(env.read_text(encoding='utf-8')))
    st.caption(f'实验目录：{path}')
    st.caption('每次任务创建 source/ 快照；修改工程中的模块不会影响正在执行的任务。')
with t2:
    st.markdown('''**解释模块**：`extensions/models/GTRSignalCore.py` 可以输出 `gtr_prediction`、`low_correction`、`high_correction` 和 `prediction`。训练预测的可分解性不是因果解释。

**重点诊断**：验证集最优 Epoch、训练/验证损失差距、分支修正幅度、不同高频能量样本的条件误差、可学习周期矩阵 Q 的变化。后两项需通过独立诊断脚本运行，而不是根据测试集选择模型。''')
    st.code('python -m lab.inspect --model GTRHaarHigh --dataset ETTm1 --horizon 96',language='bash')
with t3:
    rows=[]
    for rid in ids:
        run=RUNS/rid
        try:
            cfg=json.loads((run/'config.json').read_text(encoding='utf-8'))
            s=json.loads((run/'status.json').read_text(encoding='utf-8'))
            val=read_csv('loss.csv') if rid==selected else (pd.read_csv(run/'loss.csv') if (run/'loss.csv').exists() else pd.DataFrame())
            rows.append({'id':rid,'dataset':cfg['dataset'],'model':cfg['model'],'horizon':cfg['pred_len'],'mode':cfg['mode'],
                         'state':s.get('state'),'best_val_loss':val['val_loss'].min() if not val.empty else None,
                         'test_mse':s.get('final_test_mse'),'test_mae':s.get('final_test_mae')})
        except Exception:pass
    if rows:st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    st.caption('不同 train/validation/test 划分和数据预处理不得直接作为同协议比较。')
