# langchain-mcp-adapters Issue #417 设计文档

## Analysis

### 1. 当前代码位置与问题

`langchain_mcp_adapters/tools.py:426` 当前将 MCP `Tool` 转成 LangChain `StructuredTool` 时，只传递了：

- `name`
- `description`
- `args_schema=tool.inputSchema`
- `coroutine`
- `response_format="content_and_artifact"`
- `metadata`

这里没有传递 MCP 工具定义中的 `outputSchema`，因此 MCP 服务器已经声明的结构化输出契约，在 LangChain 侧会丢失。

### 2. `StructuredTool` 是否支持 `output_schema`

结论：不能直接假设支持。

依据：

- LangChain Reference 的 [`StructuredTool`](https://reference.langchain.com/python/langchain-core/tools/structured/StructuredTool) 页面只列出了 `description`、`args_schema`、`func`、`coroutine` 等属性，没有列出 `output_schema` 作为 `StructuredTool` 的声明字段。
- LangChain Reference 的 [`Runnable.output_schema`](https://reference.langchain.com/python/langchain-core/runnables/base/Runnable/output_schema) 页面表明，`output_schema` 是 `Runnable` 暴露出来的输出模式属性，语义更接近“运行时/派生属性”，而不是 `StructuredTool(...)` 明确声明的构造参数。

因此，`output_schema` 在 LangChain 里很可能是：

- 只读或派生属性，而不是 `StructuredTool` 的显式模型字段；
- 即使某些版本可接受额外 `kwargs`，也不能保证所有受支持版本都会接受并正确使用它。

在当前依赖范围 `langchain-core>=1.0.0,<2.0.0` 下，不能无保护地写死 `output_schema=tool.outputSchema`。

### 3. MCP 规范里 `tool.outputSchema` 是否存在

结论：存在，而且是正式规范字段。

依据：

- MCP 文档 [`Tools`](https://modelcontextprotocol.io/docs/concepts/tools) 明确列出 `outputSchema` 为工具定义字段。
- MCP 规范页 [`server/tools`](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) 明确说明：
  - `outputSchema` 是可选字段；
  - 如果提供，服务器返回的 `structuredContent` 必须符合该 schema；
  - 客户端应该对 `structuredContent` 做校验。
- MCP Schema Reference [`Tool`](https://modelcontextprotocol.io/specification/2025-11-25/schema) 中也定义了 `outputSchema?: {...}`，并明确它描述的是 `CallToolResult.structuredContent` 的结构。

所以 Issue #417 的核心判断是成立的：MCP 侧确实有 `outputSchema`，而适配器当前没有把它映射到 LangChain 可消费的位置。

### 4. 直接新增 `output_schema=tool.outputSchema` 是否是 breaking change

结论：无保护地直接新增，有破坏兼容性的风险。

原因分两层：

- 对 `langchain-mcp-adapters` 的 API 语义来说，暴露更多 schema 信息本身是增量能力，不算概念上的 breaking change。
- 但对运行时兼容性来说，如果某些 `langchain-core` 版本不接受 `output_schema` 这个构造参数，`convert_mcp_tool_to_langchain_tool()` 将在工具加载阶段直接失败，这会变成真实的 breaking change。

还要考虑一个次级风险：

- 一旦 LangChain 某版本真正开始消费或校验该 `output_schema`，此前“结构化结果不严格匹配 schema”的 MCP 服务可能开始暴露错误。
- 这更像是“让已有协议违约显性化”，不是适配器 API 破坏，但会带来行为变化。

因此推荐方案不是“无条件透传”，而是“能力探测 + 条件透传 + 兼容回退”。

### 5. 推荐方案

推荐实现目标：

- 在 `StructuredTool` 明确支持 `output_schema` 时透传 `tool.outputSchema`；
- 在不支持时，不让工具构造失败；
- 尽量保留这份 schema，便于上层未来使用。

推荐实现方式：

1. 新增一个小型能力探测函数，例如 `_supports_structured_tool_output_schema() -> bool`。
2. 通过稳定、低成本的方式判断当前 `StructuredTool` 是否声明了 `output_schema`：
   - 优先检查 `StructuredTool.model_fields` / 等价字段中是否存在 `output_schema`；
   - 或检查 `inspect.signature(StructuredTool)` / `inspect.signature(StructuredTool.__init__)` 是否可安全接受该字段；
   - 不建议用“先传再捕获异常”作为主逻辑，因为这会把正常控制流建立在异常上。
3. 构造 `StructuredTool` 参数字典：
   - 基础字段维持现状；
   - 如果 `tool.outputSchema is not None` 且能力探测为真，则加入 `output_schema=tool.outputSchema`。
4. 如果 `tool.outputSchema is not None` 但当前 LangChain 不支持，推荐把 schema 放入 `metadata` 中保底暴露，例如：
   - `metadata["mcp_output_schema"] = tool.outputSchema`

推荐使用 `mcp_output_schema` 而不是直接复用 `_meta`：

- `_meta` 是 MCP 原始字段，适配器不应混入自有兼容信息；
- 单独的 `mcp_output_schema` 更清晰，也避免污染原始协议语义。

### 6. 为什么不建议只改一行

如果只把当前代码改成：

```python
return StructuredTool(
    ...,
    output_schema=tool.outputSchema,
)
```

问题在于：

- 这隐含假设所有受支持的 `langchain-core` 版本都接受该字段；
- 但现有公开文档并不能证明这一点；
- 仓库当前的依赖范围又覆盖整个 1.x，因此不能把实现绑定到某个更高版本的行为。

因此最稳妥的修复不是“一行透传”，而是“前向增强但对低版本安全退化”。

## Implementation Steps

1. 在 `langchain_mcp_adapters/tools.py` 中新增一个内部辅助函数，用于检测 `StructuredTool` 是否声明 `output_schema`。
2. 在 `convert_mcp_tool_to_langchain_tool()` 中先构造 `metadata`，再在 `tool.outputSchema` 存在时分支处理：
   - 支持则传 `output_schema`；
   - 不支持则把 schema 放入 `metadata["mcp_output_schema"]`。
3. 保持 `response_format="content_and_artifact"` 不变，因为当前适配器已经用 `artifact.structured_content` 承载 MCP 的 `structuredContent`。
4. 不在这一变更里新增强制校验逻辑。
   - MCP 规范说客户端“SHOULD validate”，不是“MUST validate”；
   - 当前 issue 的目标是“保留并传递 schema”，不是引入新的验证失败路径。
5. 如果后续维护者希望启用校验，应作为单独 feature/PR 处理，并显式讨论错误处理、开关策略、性能影响和兼容性。

## Test Strategy

### 1. 单元测试：支持 `output_schema` 的路径

新增测试，覆盖：

- 构造带 `outputSchema` 的 `MCPTool`；
- monkeypatch 能力探测函数返回 `True`；
- 调用 `convert_mcp_tool_to_langchain_tool()`；
- 断言返回的 `StructuredTool` 带有 `output_schema`，且值等于 `tool.outputSchema`。

说明：

- 这类测试不应强依赖当前 CI 环境里安装的 `langchain-core` 版本是否真的支持该字段；
- 最稳妥的方式是 monkeypatch 一个包装层或工厂函数，而不是把测试绑死在第三方版本行为上。

### 2. 单元测试：不支持 `output_schema` 的兼容回退路径

新增测试，覆盖：

- 构造带 `outputSchema` 的 `MCPTool`；
- monkeypatch 能力探测函数返回 `False`；
- 调用转换函数；
- 断言：
  - 工具构造成功；
  - `metadata["mcp_output_schema"] == tool.outputSchema`；
  - 现有 metadata 合并逻辑不被破坏。

### 3. 单元测试：与现有 metadata 合并规则兼容

补充一个组合测试：

- `annotations`、`_meta`、`outputSchema` 同时存在；
- 在 fallback 路径下断言 metadata 同时包含：
  - annotations 展平字段；
  - `_meta`；
  - `mcp_output_schema`。

这能避免新增输出 schema 后覆盖现有 metadata。

### 4. 单元测试：无 `outputSchema` 时行为不变

新增或扩展现有测试，确认：

- 未提供 `outputSchema` 的 `MCPTool` 在两条路径下都保持现有行为；
- 不新增多余 metadata；
- 现有 `load_mcp_tools()`、工具调用、`structuredContent -> artifact` 行为不回归。

### 5. 集成测试建议

如果测试环境中的 MCP 服务器实现支持 `outputSchema`，可增加一个轻量集成测试：

- server 端注册一个带 `outputSchema` 的工具；
- client 端 `load_mcp_tools()` 后检查 schema 是否被保留；
- 执行工具并确认原有 `structuredContent` 转换逻辑不受影响。

这不是首要阻塞项，但有助于验证“声明 schema”和“实际返回 structuredContent”在真实链路里能同时工作。

## Risk Assessment

### 低风险

- MCP 规范层面：`outputSchema` 是正式字段，读取它本身没有协议风险。
- 适配器语义层面：把更多工具描述信息暴露出去属于增量能力。

### 中风险

- LangChain 兼容性：`StructuredTool` 是否接受 `output_schema` 不能仅凭当前 issue 假设，必须做能力探测。
- Metadata 扩展：新增 `mcp_output_schema` 会改变 `tool.metadata` 内容，不过这是向后兼容的新增字段，风险可控。

### 较高风险

- 若未来启用严格 schema 校验，可能会让一些当前“可运行但不完全符合 schema”的 MCP 服务开始失败。
- 因此本次不建议顺手加入验证逻辑，否则 issue 范围会从“传递 schema”膨胀成“改变执行语义”。

## 最终建议

建议接受 Issue #417，但不要直接无条件增加 `output_schema=tool.outputSchema`。

推荐落地策略是：

- 先做 `StructuredTool` 能力探测；
- 支持时透传 `output_schema`；
- 不支持时把 schema 保存在 `metadata["mcp_output_schema"]`；
- 本次不引入 `structuredContent` 校验，只做 schema 保留和前向兼容铺垫。

这样可以同时满足：

- 对 MCP 规范的正确映射；
- 对 `langchain-core` 版本范围的兼容；
- 对未来 LangChain/MCP 结构化输出能力的平滑演进。
