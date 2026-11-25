# 修复一个失败的用例

`totals.py` 中的 `total_even` 是待修复实现。随附测试覆盖空输入和过滤奇数。先把示例复制到独立目录，从项目根运行下面的命令观察初始失败。

```sh
python -c "import shutil; shutil.copytree('examples/bugfix', '.mca-work/bugfix')"
python -m pytest -q .mca-work/bugfix
```

如果目标目录已存在，请换一个新目录，并更新测试命令里的路径。

修复目标是对偶数求和。可手动修改工作副本，或按 [CLI 说明](../../README.md#cli) 设置模型后让 Agent 修复。项目的集成测试使用 Fake SDK 验证该修复流程。
