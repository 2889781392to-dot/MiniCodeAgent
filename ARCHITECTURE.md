# 架构

## 轮次执行

[CLI](src/minicodeagent/cli.py) 构造 Workspace、配置与确认函数，将输入交给 [AgentLoop](src/minicodeagent/agent.py)。启用上下文管理后，[ContextWindowManager](src/minicodeagent/context/manager.py) 可在主请求前调整模型可见视图。持久会话先登记主请求，再由 [LLMClient](src/minicodeagent/llm.py) 发送。普通运行在进程内维护对话历史。已接纳的工具批次经 [Registry](src/minicodeagent/tools/registry.py) 校验，结果按调用 ID 写入历史。下一次请求可消费这些结果。

模型尝试、工具批次和批内调用分别限额。瞬态失败可按预算重试，启用上下文管理时，明确的窗口拒绝可触发一次上下文收紧。文件工具由 [Workspace](src/minicodeagent/workspace.py) 校验路径。命令采用实际 argv 和 cwd，等待单次批准，再按宿主账户权限执行。

## 会话和副作用

[JsonlEventStore](src/minicodeagent/context/store.py) 在会话写租约内追加结构化事件。[SessionReducer](src/minicodeagent/context/replay.py) 从事件恢复轮次、请求计数与压缩视图。回放本身只重建状态，外部动作的执行情况要以回执和现场证据核对。

[SessionService](src/minicodeagent/session/service.py) 管理创建、打开、Resume 和 Fork。缺少工具回执时，先核对未知副作用，再确认恢复记录。原调用 ID 保留。Fork 写入独立子日志，可绑定原工作区或另一目录。

## 上下文

[ContextBuilder](src/minicodeagent/context/builder.py) 构造模型投影。[预算与压缩实现](src/minicodeagent/context/) 根据显式窗口选择清理旧文本、提取已消费工具结果、归档历史前缀或生成交接摘要。候选须通过保护边界及相应证据校验。新用户目标和关键工具事务保持可见。有效压缩以事件提交，原始事件用于回放及追溯。
