"""XLinear 本地只读 Streamlit 看板：所有图表均来自当前训练的真实文件。"""
from __future__ import annotations
from pathlib import Path
import json
import subprocess
import pandas as pd
import streamlit as st
import yaml

ROOT=Path(__file__).resolve().parents[2]
RUNS=ROOT/'runs'
try:
    REFRESH_SECONDS=max(2,int(yaml.safe_load((ROOT/'configs'/'experiment.yaml').read_text(encoding='utf-8')).get('monitoring',{}).get('dashboard_refresh_seconds',3)))
except (OSError,ValueError,TypeError):
    REFRESH_SECONDS=3
st.set_page_config(page_title='XLinear Research Lab',page_icon='📈',layout='wide')
st.title('XLinear · 训练与模块研究工作台')
st.caption('只读监控｜官方训练协议｜指标来自当前运行日志，不会因为打开网页而启动或修改训练。')

def read_json(path):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError):return {}

def load_csv(path):
    try:
        if path.is_file() and path.stat().st_size>0:
            return pd.read_csv(path)
    except (OSError,pd.errors.EmptyDataError,pd.errors.ParserError): pass
    return pd.DataFrame()

def gpu_now():
    try:
        p=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
        return p.stdout.strip() if p.returncode==0 else 'GPU 状态不可用：'+p.stderr.strip()
    except Exception as exc:return f'GPU 状态不可用：{exc}'

@st.fragment(run_every=f'{REFRESH_SECONDS}s')
def live():
    runs=sorted(RUNS.glob('*/manifest.json'),reverse=True)
    if not runs:
        st.info('暂无新版本实验。先执行 `bash run.sh setup`、`bash run.sh pilot` 或 `bash run.sh full`。')
        st.code(gpu_now());return
    names=[p.parent.name for p in runs]
    choice=st.selectbox('实验批次（自动刷新）',names,key='selected_run')
    run=RUNS/choice
    info=read_json(run/'progress.json');manifest=read_json(run/'manifest.json')
    jobs=sorted((run/'jobs').glob('*')) if (run/'jobs').exists() else []
    current=info.get('current')
    names_job=[j.name for j in jobs if j.is_dir()]
    a,b,c,d=st.columns(4)
    a.metric('任务状态',info.get('status','unknown'))
    b.metric('已完成模型',f"{info.get('completed',0)}/{info.get('total',0)}")
    c.metric('Epoch',f"{info.get('epoch',0)}/{info.get('epoch_max','—')}")
    d.metric('Batch',f"{info.get('batch',0)}/{info.get('steps_per_epoch') or '—'}")
    done=float(info.get('completed',0));total=max(int(info.get('total',1)),1)
    within=(float(info.get('epoch',0))+float(info.get('batch',0))/max(int(info.get('steps_per_epoch',1)),1))/max(float(info.get('epoch_max',1)),1)
    st.progress(min(1.0,(done+min(1.0,within))/total),text=f"预计工作完成度（按任务/Epoch计）：{(done+min(1.0,within))/total*100:.1f}%｜早停可能使估计不精确")
    st.caption(f"当前作业：{current or '—'} ｜ GPU：{manifest.get('device','—')} ｜ 源码版本：{manifest.get('source_commit','—')}")
    if not names_job:
        st.info('训练任务正在准备中，请查看上方状态。');return
    selected=st.selectbox('选择模型 / 数据集 / 预测长度 / 随机种子',names_job,
        index=names_job.index(current) if current in names_job and 'selected_job' not in st.session_state else 0,
        key='selected_job')
    job=run/'jobs'/selected
    metrics=job/'metrics';diags=job/'diagnostics';results=run/'results'
    tab1,tab2,tab3,tab4,tab5,tab6=st.tabs(['训练与验证曲线','Batch 与 GPU','模型对比','预测与门控诊断','模型结构说明','日志与下载'])
    with tab1:
        epochs=load_csv(metrics/'epochs.csv')
        if epochs.empty:
            st.info('等待第一个 Epoch 结束后生成 Train/Validation Loss。')
        else:
            chart=epochs.set_index('epoch')[['train_loss','val_loss']].rename(columns={
                'train_loss':'Train MSE','val_loss':'Validation MSE'})
            st.line_chart(chart,y_label='MSE',x_label='Epoch',height=340)
            best=epochs.loc[epochs['val_loss'].idxmin()]
            k1,k2,k3=st.columns(3)
            k1.metric('最佳验证 Epoch',int(best['epoch']))
            k2.metric('最低验证 MSE',f"{best['val_loss']:.6f}")
            k3.metric('当前 Train - Val 差距',f"{float(epochs.iloc[-1]['val_loss']-epochs.iloc[-1]['train_loss']):+.6f}")
            gap=epochs.assign(gap=epochs['val_loss']-epochs['train_loss']).set_index('epoch')[['gap']]
            st.line_chart(gap,y_label='Val MSE - Train MSE',x_label='Epoch',height=180)
            with st.expander('每个 Epoch 的真实数值'):
                st.dataframe(epochs.drop(columns=['test_loss'],errors='ignore'),hide_index=True,use_container_width=True)
            st.caption('Train 与 Validation 曲线实时更新；作者训练器的 Test Loss 不用于选模，本页面默认隐藏，最终测试结果只在训练完成后报告。')
    with tab2:
        batches=load_csv(metrics/'batches.csv')
        if not batches.empty:
            st.subheader('Batch 级训练 Loss（实时采样）')
            st.line_chart(batches.set_index('global_batch')[['train_batch_loss']],height=290,
                          y_label='单个 Batch MSE',x_label='Global Batch')
            st.caption('Batch MSE 本身波动较大，主要用于检查发散和异常；验证 MSE 仅在每个 Epoch 完成后计算。')
        else:st.info('等待首条 Batch 采样；也可在配置中调小 monitoring.batch_interval。')
        st.subheader('当前显卡')
        st.code(gpu_now())
        gpu=load_csv(metrics/'gpu.csv')
        if not gpu.empty:
            c1,c2=st.columns(2)
            with c1: st.line_chart(gpu.set_index('time')[['gpu_util_pct']],height=230,y_label='GPU 利用率 %')
            with c2: st.line_chart(gpu.set_index('time')[['vram_used_mib']],height=230,y_label='显存 MiB')
            st.caption('GPU 监测是整个设备的外部采样，不能将利用率变化直接归因于某个网络层。')
    with tab3:
        for title,filename in [('与原始 XLinear 的逐种子配对比较','paired_comparison.csv'),
                               ('多随机种子汇总','summary.csv'),('完成的全部实验','all_results.csv')]:
            data=load_csv(results/filename)
            if not data.empty:
                st.subheader(title)
                st.dataframe(data,use_container_width=True,hide_index=True)
        paired=load_csv(results/'paired_comparison.csv')
        if not paired.empty:
            st.bar_chart(paired.set_index('variant')[['improvement_pct_vs_official']],y_label='MSE 改善百分比')
        st.caption('正式报告以独立完整训练与配对随机种子为依据；pilot 仅用于工程排错。')
    with tab4:
        pred=load_csv(diags/'prediction_sample.csv')
        if pred.empty:
            st.info('作业完成后，在远程终端执行下方命令生成预测、门控与梯度诊断：')
            st.code(f'bash run.sh diagnose --run {choice} --job {selected}',language='bash')
        else:
            st.subheader('最后一个变量（OT）的单个测试窗口预测与真实值')
            st.line_chart(pred.set_index('horizon_step')[['actual','prediction']],height=290,y_label='训练集标准化单位')
            horizon=load_csv(diags/'horizon_errors.csv')
            if not horizon.empty:
                st.subheader('不同预测步长的误差变化')
                st.line_chart(horizon.set_index('horizon_step')[['mse','mae']],height=240)
            channel=load_csv(diags/'channel_errors.csv')
            if not channel.empty:
                with st.expander('逐变量 MSE/MAE'):
                    st.dataframe(channel,hide_index=True,use_container_width=True)
            diagnostic=read_json(diags/'module_diagnostics.json')
            if diagnostic:
                st.subheader('TGM / VGM Sigmoid 门控分布')
                gates=diagnostic.get('gate_stats',{})
                for name,stat in gates.items():
                    st.markdown(f'**{name}**：mean={stat["mean"]:.3f}，std={stat["std"]:.3f}；低于0.1占比 {stat["fraction_below_0_1"]:.1%}，高于0.9占比 {stat["fraction_above_0_9"]:.1%}')
                    hist=pd.DataFrame({'gate_range':[f'{i/10:.1f}-{(i+1)/10:.1f}' for i in range(10)],
                                       'count':stat['histogram_counts']}).set_index('gate_range')
                    st.bar_chart(hist,height=160)
                if diagnostic.get('token'):
                    st.markdown('**Token 新增模块状态：**')
                    st.json(diagnostic['token'])
                st.subheader('单批验证梯度 L2 范数')
                st.dataframe(pd.DataFrame([dict(module=k,**v) for k,v in diagnostic.get('gradient_norms',{}).items()]),
                             hide_index=True,use_container_width=True)
                st.caption('门控值≠因果重要性，梯度仅为单个验证 Batch 的局部敏感性；这些诊断不影响权重。')
    with tab5:
        st.markdown('''### 官方主干保持不变
`X [B,L,C] → 标准化 → Embedding [B,C,D] → Global Token → TGM → VGM → Linear Head → Y [B,H,C]`

| 模块 | 作用 | 改造位置 |
|:--|:--|:--|
| Embedding | 历史窗口压缩至 D 维 | 官方源代码只读，扩展写入 `src/models/extensions/` |
| Global Token | 全局共享信息入口 | `global_token.py` 的 `dyn_token` / `stat_token` |
| TGM | 沿特征维门控融合时间上下文 | `core.en_attention`（保留原版） |
| VGM | 变量间门控与共享信息选择 | `core.ex_attention`（保留原版） |
| Prediction Head | 产生 H 步预测 | 官方原版，不参与这轮改造 |

**研究建议：** 每次只改变一个功能算子；同数据、同训练参数、同种子配对验证。Gate 与梯度快照只用于诊断，不直接代表预测贡献或因果机制。''')
        st.caption('详细模块输入输出、改造接口与图示参见 docs/01_model_flow.md、docs/05_visual_monitoring.md。')
        st.json({'selected_job':selected,'job_config':read_json(job/'command.json').get('job',{})})
    with tab6:
        logfile=job/'train.log'
        if logfile.is_file():
            with logfile.open(encoding='utf-8',errors='replace') as handle:
                from collections import deque
                lines=''.join(deque(handle,maxlen=110))
            st.code(lines,language='text')
        for title,path in [('Epoch CSV',metrics/'epochs.csv'),('Batch CSV',metrics/'batches.csv'),
                            ('GPU CSV',metrics/'gpu.csv'),('对照表',results/'paired_comparison.csv')]:
            if path.exists():
                st.download_button(f'下载 {title}',path.read_bytes(),file_name=f'{selected}_{path.name}',key=f'dl_{choice}_{selected}_{title}')
        st.caption('关闭浏览器不会停止后台训练。Train/Val 图表与 CPU 分析不会占用 GPU 训练的前向/反向计算。')

live()
