# Provider 参数兼容性：后续事项

当前真实参数探测结果：

- GPT-5.6 Terra：不传 reasoning_effort 或显式 medium 时，已测的非默认 temperature 被拒绝；none 时，已测的多个非默认值被接受。不据此推断所有 effort 的行为。
- Claude Sonnet 5：已观察到 adaptive thinking 下 temperature=0.7 被拒绝，错误说明该模式只允许 1。关闭 thinking 后的行为应以对应实测结果为准。

当前处理：provider smoke test 显式使用 temperature=1.0；不修改 QueryRequest 的默认值或业务 provider。Anthropic provider 当前不传 temperature，所以 smoke test 不验证其 temperature 透传。

后续再决定是否将 temperature 改为可选且默认 None（provider 应省略该字段，而不是直接发送 null），以及是否引入 thinking/effort 参数并分别映射 OpenAI 和 Anthropic 的接口。届时同时检查普通/流式请求与缓存 key，避免不同生成参数复用同一缓存。

模型名称已统一为 Anthropic 官方 API ID `claude-sonnet-5`，包括运行配置、默认配置、setup 模板与 routing rule 引用。
