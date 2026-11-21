"""Phase2 dashboard: validation-first model comparison and mechanism diagnostics."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import pandas as pd
import streamlit as st
from phase2.batch import BATCHES, resolve_batch, validation_rows, load_json, read_loss
from lab.paths import RUNS

st.set_page_config(page_title='GTR Phase2 – Phase Innovation',layout='wide')
st.title('GTR · Phase-Calibrated Innovation')
st.caption('论文问题：全局周期相位参考能解释哪些局部变化？未解释的历史变化是否仍具有预测价值？')
st.warning('本页面只用验证集选结构；作者原始训练脚本虽会自动计算测试指标，但这里不展示测试结果。')
if not BATCHES.exists() or not list(BATCHES.glob('*/state.json')):
    st.info('尚无第二轮实验。运行：bash 11_Phase2_CPU.sh；随后 bash 12_Phase2_GPU.sh Weather 96 2024')
    st.stop()
ids=sorted((p.parent.name for p in BATCHES.glob('*/state.json')),reverse=True)
selected=st.selectbox('选择批次',ids)

@st.fragment(run_every='5s')
def live():
    folder=BATCHES/selected
    state=load_json(folder/'state.json')
    a,b,c=st.columns(3)
    a.metric('批次状态',state.get('phase','unknown'))
    b.metric('当前模型',state.get('current_model','—'))
    c.metric('已启动任务',len(state.get('jobs',[])))
    if state.get('reason'):st.error(state['reason'])
    rows=validation_rows(folder)
    if rows:
        st.subheader('官方一致配置下的最佳验证表现')
        df=pd.DataFrame(rows)
        st.dataframe(df[['model','status','best_val_mse','best_epoch','last_epoch','duration_sec','run_id']],
                     use_container_width=True,hide_index=True)
    series=[]
    for info in rows:
        for x in read_loss(info['run_id']):
            series.append({'epoch':x['epoch'],'model':info['model'],
                           'train_loss':x['train_loss'],'val_loss':x['val_loss']})
    if series:
        st.subheader('Validation Loss · 所有模型')
        df=pd.DataFrame(series)
        st.line_chart(df.pivot(index='epoch',columns='model',values='val_loss').sort_index(),height=300)
        choose=st.selectbox('查看单个模型的 Train/Validation 曲线',sorted(df['model'].unique()))
        curve=df[df.model==choose][['epoch','train_loss','val_loss']].set_index('epoch')
        st.line_chart(curve,height=250)
    if state.get('current_run_id'):
        active=RUNS/state['current_run_id']
        status=active/'status.json'
        if status.exists():
            q=load_json(status)
            st.caption(f"当前实验 Epoch {q.get('epoch',0)}/{q.get('expected_epochs','?')}, 状态：{q.get('state','unknown')}")
            if q.get('state')=='failed':
                st.error('训练失败；请查看下方日志中的 Traceback。')
        log=active/'console.log'
        if log.is_file():
            st.subheader('当前训练输出')
            st.code(log.read_text(encoding='utf-8',errors='replace')[-7000:],language='text')
    orch=folder/'orchestrator.log'
    if orch.exists():
        st.subheader('批次执行日志')
        st.code(orch.read_text(encoding='utf-8',errors='replace')[-3000:],language='text')

live()
st.divider()
st.subheader('验证集解释文件（由 explain 命令生成）')
choices=[]
state=load_json(BATCHES/selected/'state.json')
for job in state.get('jobs',[]):
    if job['model'].startswith('GTRPhase'):
        choices.append((job['model'],job['run_id']))
if choices:
    label=st.selectbox('选择已训练模型', [f'{x[0]} / {x[1]}' for x in choices])
    rid=choices[[f'{x[0]} / {x[1]}' for x in choices].index(label)][1]
    p=RUNS/rid/'phase2_validation_windows.csv'
    if p.exists():
        explain=pd.read_csv(p)
        st.dataframe(explain.head(30),hide_index=True,use_container_width=True)
        st.caption('delta_mse = proposed_mse - internal GTR branch MSE；负值表示该模型的创新分支在该窗口减少误差，不能当作独立原始 GTR 的因果贡献。')
        if len(explain):
            bins=pd.cut(explain['phase_fit_r2'],bins=[-0.001,.2,.4,.6,.8,1.001])
            avg=explain.groupby(bins,observed=True)['delta_mse'].agg(['mean','count']).reset_index()
            avg['phase_fit_r2']=avg['phase_fit_r2'].astype(str)
            st.subheader('不同相位参考拟合区间的修正效果（仅验证集）')
            st.dataframe(avg,hide_index=True,use_container_width=True)
    else:
        st.info(f'解释文件尚未生成。终端运行：./.venv/bin/python -m phase2.explain {rid} --limit 64')
st.caption('GTR 的 Q 是学习得到的潜在周期参数。拟合度仅描述局部历史与校准后的参考相似性，不是置信概率，也不是因果解释。')
