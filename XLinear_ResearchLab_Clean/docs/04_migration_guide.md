# 备份、迁移和常见报错

**1. 新项目与旧项目的关系**：新 ZIP 提供全新根目录，不是再向旧文件夹叠加脚本。因此在 `/root/autodl-tmp` 下先把旧的 `XLinear_ResearchLab_Clean` 重命名，再解压新项目。重命名不会删除原先结果。

**2. 旧实验权重在哪里**：旧 `runs/20261115_001645_b94e7f/` 和旧 `XLinear_Paper_Reference_Patch_ASCII/paper_reference/runs/` 仍在备份目录。不要把旧结果复制到新 `runs/`，否则新摘要可能混入不同协议。

**3. 避免重新下载**：`bash run.sh setup --reuse-upstream <旧版 upstream_XLinear 的完整路径>`。复制后会逐一校验模型、训练器、数据加载器和工具文件哈希。若不匹配不自动修改。

**4. GPU 检查**：`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`。不能看到 GPU 就直接修复当前 PyTorch 环境，别随意执行会覆盖 GPU torch 的依赖安装命令。

**5. 安装可视化**：`python -m pip install -r requirements-ui.txt`，然后在第二个终端启动 `bash run.sh dashboard`，VS Code Ports 转发 `8501`。只开网页不会开始训练。

**6. 终止前台训练**：`Ctrl+C`。后台运行进程使用 `bash run.sh status` 查 PID，避免无差别 `pkill python`。停止后查看对应日志并新建一次独立实验。

**7. 目录中只有 CSV，没有模型权重**：检查每个 `runs/<run>/jobs/<job>/checkpoints/**/checkpoint.pth`；官方训练器按验证损失保存最佳 checkpoint。

**8. 原论文值和本项目不同**：先确认采用 `full` 而非 `pilot`，且配置是作者对应数据集和预测期的 `profiles`，并检查固定源码哈希、torch 版本、种子和数据哈希。

**9. 如果需要恢复旧项目**：停止新训练后把新文件夹改名，然后将备份目录重命名回 `XLinear_ResearchLab_Clean` 即可；两者可以共存但不要同时启动 GPU 任务。
