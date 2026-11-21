"""所有模型统一调用官方 run_longExp.py。

仅在内存中给上游训练器加只读的 Batch 日志；不修改官方磁盘源码、前向、
反向传播、优化器、早停、数据切分或计算图。设 --monitor-batch-interval 0 可禁用。
监控输出 LAB_BATCH 可供外层 runner 与 Streamlit 实时解析。
"""
import argparse
import inspect
from pathlib import Path
import runpy
import sys
import textwrap

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def instrument_train_function(original, interval:int):
    """对已校验作者源码的 train 方法仅增加 print；测试可检查结果不变。"""
    if interval<=0:return original
    text=textwrap.dedent(inspect.getsource(original))
    needle='                train_loss.append(loss.item())'
    # dedent 后，训练循环中的缩进比源文件减少4个空格。
    needle='            train_loss.append(loss.item())'
    if text.count(needle)!=1:
        raise RuntimeError('上游 train() 源码锚点不匹配，拒绝注入监控；不更改作者训练逻辑。')
    injection=(needle+'\n'
       '            # XLinear Lab: 只读日志；不增加 forward/backward/optimizer 操作。\n'
       f'            if ((i + 1) % {int(interval)} == 0) or (i + 1 == train_steps):\n'
       '                print(f"LAB_BATCH epoch={epoch+1} batch={i+1} steps={train_steps} loss={train_loss[-1]:.9f}", flush=True)')
    patched=text.replace(needle,injection,1)
    namespace={}
    exec(compile(patched,inspect.getsourcefile(original) or '<XLinear-instrumented>', 'exec'),
         original.__globals__, namespace)
    return namespace['train']

def attach_batch_logger(interval:int):
    if interval<=0:return
    import exp.exp_main as upstream_exp
    upstream_exp.Exp_Main.train=instrument_train_function(upstream_exp.Exp_Main.train,interval)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',required=True)
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--context-ratio',type=float,required=True)
    parser.add_argument('--monitor-batch-interval',type=int,default=0)
    options,original_args=parser.parse_known_args()
    upstream=options.upstream.resolve()
    if not (upstream/'run_longExp.py').is_file():
        raise FileNotFoundError('官方训练器不存在: '+str(upstream))
    sys.path.insert(0,str(upstream))
    from models import XLinear
    if options.variant!='official':
        from src.models.registry import build_variant_model
        XLinear.Model=build_variant_model(XLinear.Model,options.variant,options.context_ratio)
    attach_batch_logger(options.monitor_batch_interval)
    sys.argv=[str(upstream/'run_longExp.py')]+original_args
    runpy.run_path(str(upstream/'run_longExp.py'),run_name='__main__')

if __name__=='__main__':main()
