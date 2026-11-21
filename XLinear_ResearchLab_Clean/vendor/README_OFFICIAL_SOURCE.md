# 官方源码目录

正式运行前执行 `bash run.sh setup`。它会从 GitHub 下载**指定提交**的作者完整 XLinear 仓库到 `vendor/XLinear/`，并对作者模型、训练器、数据加载器等关键源码进行 Git blob 哈希校验。

如果你已经在旧项目成功下载并训练，建议使用 `bash run.sh setup --reuse-upstream <旧项目中 upstream_XLinear 的路径>`，这样直接复制已校验的代码，避免重复网络下载。

不要手动修改 `vendor/XLinear/`。新方法写入 `src/models/extensions/`，通过 `src/models/registry.py` 注册。上游文件是有许可证约束的第三方源码，项目不把它当作自行创作的代码。
