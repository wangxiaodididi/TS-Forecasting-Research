"""Live Phase-1 ablation comparison in the same Streamlit application."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import pandas as pd
import streamlit as st
from phase1.core import BATCHES,RUNS,load_json,read_loss,BASELINE_METRICS

st.set_page_config(page_title='GTR Phase 1 Ablation', layout='wide')
st.title('GTR 第一轮结构消融 · 实时对比')
st.caption('固定官方训练协议；不修改核心检索模块；GPU 任务按模型顺序运行。')
if not BATCHES.exists() or not list(BATCHES.glob('*/state.json')):
    st.info('尚无研究批次。先运行：bash 08_Phase1_GPU.sh（或 bash 07_Phase1_CPU.sh）。')
    st.stop()
ids=sorted((p.parent.name for p in BATCHES.glob('*/state.json')),reverse=True)
selected=st.selectbox('选择实验批次',ids)

@st.fragment(run_every='5s')
def live():
    batch=BATCHES/selected;state=load_json(batch/'state.json')
    st.write('批次状态：**'+state.get('phase','unknown')+'**', '　当前模型：',state.get('current_model','—'))
    if state.get('reason'):st.error(state['reason'])
    q=[];series=[]
    work=[]
    if state.get('baseline_run'):work.append(('GTR',state['baseline_run']))
    work += [(item['model'],item['run_id']) for item in state.get('jobs',[])]
    for name,runid in work:
        path=RUNS/runid/'status.json'
        if not path.exists():continue
        s=load_json(path)
        loss=read_loss(runid)
        best=min(loss,key=lambda r:r['val_loss']) if loss else None
        q.append({'模型':name,'状态':s.get('state'),'Epoch':s.get('epoch',0),
                  '最佳验证 MSE':best['val_loss'] if best else None,
                  '最佳 Epoch':best['epoch'] if best else None,'实验 ID':runid})
        for e in loss:
            series.append({'epoch':e['epoch'],'model':name,'val_loss':e['val_loss']})
    if q:
        st.subheader('按验证集选择改造方案')
        st.dataframe(pd.DataFrame(q),hide_index=True,use_container_width=True)
    if series:
        st.subheader('各模型验证集 MSE · 逐 Epoch')
        df=pd.DataFrame(series)
        pivot=df.pivot(index='epoch',columns='model',values='val_loss').sort_index()
        st.line_chart(pivot,height=340)
    st.caption('相同超参数不意味着相同最佳 Epoch；比较验证集最优表现。测试指标不用于调参。')
    if st.checkbox('仅在模型与参数已经锁定后，显示最终测试集指标',key='unlock_'+selected):
        t=[]
        for name,runid in work:
            s=load_json(RUNS/runid/'status.json')
            t.append({'模型':name,'Test MSE':s.get('final_test_mse'),'Test MAE':s.get('final_test_mae')})
        st.dataframe(pd.DataFrame(t),hide_index=True,use_container_width=True)
        if state.get('dataset')=='ETTm1' and state.get('horizon')==96:
            st.caption('ICLR 2026 论文 ETTm1/H96 参考 Test MSE=0.305, MAE=0.349。')
    log=batch/'orchestrator.log'
    if log.exists():
        st.subheader('批次调度日志')
        st.code(log.read_text(encoding='utf-8',errors='replace')[-5000:],language='text')

live()
