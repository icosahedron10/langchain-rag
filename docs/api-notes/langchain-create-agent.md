# langchain 1.3.14 / langchain_core 1.5.3 / langgraph 1.2.10 — verified from installed source

## 1. create_agent

Import: `from langchain.agents import create_agent` (defined in `langchain/agents/factory.py:808`; `langchain.agents.__init__` exports only `create_agent` and `AgentState`).

```python
def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | dict[str, Any] | None = None,
    state_schema: type[AgentState[ResponseT]] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache[Any] | None = None,
    transformers: Sequence[TransformerFactory] | None = None,
) -> CompiledStateGraph[AgentState[ResponseT], ContextT, InputAgentState, OutputAgentState[ResponseT]]
```

## 2. Structured output

Import: `from langchain.agents.structured_output import ToolStrategy, ProviderStrategy, AutoStrategy` (NOT re-exported from `langchain.agents`).

```python
ToolStrategy(schema, *, tool_message_content: str | None = None,
             handle_errors: bool | str | type[Exception] | tuple[type[Exception], ...] | Callable[[Exception], str] = True)
ProviderStrategy(schema, *, strict: bool | None = None)
AutoStrategy(schema)   # what a bare type passed as response_format is wrapped into
```

- **Synthetic tool name** = the schema's `__name__` for Pydantic/dataclass/TypedDict (dict schemas: the `title` key; fallback `response_format_<4-hex>`). So `ToolStrategy(SomePydanticModel)` → the fake model must emit `AIMessage(tool_calls=[{"name": "SomePydanticModel", "args": {...}, "id": "call_1"}])`. Tool choice is forced to `"any"` when structured tools exist (factory.py:1388).
- **Result key**: `result["structured_response"]` (top-level state key, `NotRequired`, omitted from input). On success the agent appends a `ToolMessage` (content = `tool_message_content` or `f"Returning structured response: {structured_response}"`) answering the synthetic tool call.
- **Errors/retry** (`handle_errors`): `True` = catch all + retry with default error `ToolMessage`; `str` = retry with that message; exception type/tuple = only retry on those; callable = `Callable[[Exception], str]`; `False` = raise. Exceptions raised: `MultipleStructuredOutputsError`, `StructuredOutputValidationError` (both in `langchain.agents.structured_output`, subclass `StructuredOutputError`, carry `.ai_message`). Retry = error `ToolMessage` appended, loop returns to model.

## 3. Return value

`CompiledStateGraph` (`langgraph.graph.state.CompiledStateGraph`) — the factory ends with `graph.compile(checkpointer=checkpointer, store=store, interrupt_before=..., interrupt_after=..., debug=..., name=..., cache=..., transformers=[...])` (factory.py:1787). It is a standard LangGraph Pregel runnable: `.invoke/.ainvoke(input, config)`, `.stream/.astream(input, config, stream_mode=["updates", "messages", ...])` all supported; `checkpointer=` accepted at `create_agent`.

Custom state keys: all middleware `state_schema`s plus the base state are merged into one TypedDict (`_resolve_schemas`, factory.py:424); a key appears **top-level in the invoke result** unless annotated `OmitFromSchema(output=True)` / `PrivateStateAttr` (from `langchain.agents.middleware.types`). Input keys marked `OmitFromInput` (like `structured_response`) are excluded from the input schema.

```python
agent = create_agent(model, tools=[my_tool], response_format=ToolStrategy(MyModel), checkpointer=InMemorySaver())
out = agent.invoke({"messages": [HumanMessage("hi")]}, config={"configurable": {"thread_id": "1"}})
out["structured_response"]; out["my_custom_key"]
```

## 4. Fake chat models

Import: `from langchain_core.language_models import GenericFakeChatModel` (module: `langchain_core/language_models/fake_chat_models.py:227`).

- Constructor (pydantic field): `messages: Iterator[AIMessage | str]` — pass `GenericFakeChatModel(messages=iter([AIMessage(content="", tool_calls=[{"name": "SomePydanticModel", "args": {...}, "id": "1"}]), AIMessage(content="done")]))`. Each `_generate` call consumes `next(self.messages)`; strings are wrapped in `AIMessage(content=...)`. The returned `AIMessage` is passed through verbatim, so scripted `tool_calls` work with `.invoke`.
- `_stream`: splits string content on whitespace via `re.split(r"(\s)", content)` yielding one `AIMessageChunk` per token (last chunk gets `chunk_position="last"`); then chunks `additional_kwargs` (special-casing `function_call`, split on commas). **`tool_calls` set directly on the AIMessage are dropped in `_stream`** — only content/additional_kwargs are streamed. Use invoke paths (or `stream_mode="updates"`) for tool-call scripting.
- **`bind_tools`: BaseChatModel provides NO generic implementation** — `langchain_core/language_models/chat_models.py:2338`:
  ```python
  def bind_tools(self, tools: Sequence[builtins.dict[str, Any] | type | Callable[..., Any] | BaseTool],
                 *, tool_choice: str | None = None, **kwargs: Any) -> Runnable[LanguageModelInput, AIMessage]:
      raise NotImplementedError
  ```
  `GenericFakeChatModel` does not override it, and `create_agent` calls `model.bind_tools(...)` whenever tools (or a ToolStrategy) are present — so you MUST subclass:
  ```python
  class ToolFake(GenericFakeChatModel):
      def bind_tools(self, tools, *, tool_choice=None, **kwargs):
          return self  # scripted output ignores bound tools
  ```
  Other fakes in the same module (`FakeMessagesListChatModel(responses=[...])`, `FakeListChatModel(responses=[...])`, `ParrotFakeChatModel`) also lack `bind_tools`.

## 5. AgentMiddleware

Import: `from langchain.agents.middleware import AgentMiddleware` (class: `langchain/agents/middleware/types.py:383`, `AgentMiddleware(Generic[StateT, ContextT, ResponseT])`).

- Custom state: set class attr `state_schema: type[StateT]` to a TypedDict extending `AgentState` (import `AgentState` from `langchain.agents`). Also class attrs `tools: Sequence[BaseTool]`, `transformers: Sequence[TransformerFactory] = ()`, property `name` (defaults to class name).
- Hooks (sync/async pairs): `before_agent/abefore_agent`, `before_model/abefore_model`, `after_model/aafter_model`, `after_agent/aafter_agent` — all `(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None`; plus interceptors `wrap_model_call/awrap_model_call(request, handler)` and `wrap_tool_call/awrap_tool_call(request, handler)`. Function decorators with the same names (`before_model`, `after_model`, `before_agent`, `after_agent`, `wrap_model_call`, `wrap_tool_call`, all taking `state_schema=`, `tools=`, `name=` kwargs) are exported from `langchain.agents.middleware`.

## 6. recursion_limit and runtime access in tools

- Per-run: `agent.invoke(input, config={"recursion_limit": 50, "configurable": {...}})` — `recursion_limit` is a top-level `RunnableConfig` key (default 25, `langchain_core/runnables/config.py:171`; langgraph raises `GraphRecursionError` mentioning "Recursion limit of N reached"). Also settable via `agent.with_config(recursion_limit=50)`.
- `ToolRuntime`: `from langchain.tools import ToolRuntime` (re-export of `langgraph.prebuilt.ToolRuntime`, defined `langgraph/prebuilt/tool_node.py:1663`). A parameter annotated `runtime: ToolRuntime` (no `Annotated` needed) is auto-injected and hidden from the model. Fields: `state`, `context`, `config: RunnableConfig`, `stream_writer`, `tool_call_id: str | None`, `store: BaseStore | None`, `tools: list[BaseTool]`, `execution_info`, `server_info`; method `emit_output_delta(delta)`.
- `RunnableConfig` direct injection: any tool param annotated exactly `RunnableConfig` (any name) receives the config (`langchain_core/tools/base.py:1472` `_get_runnable_config_param`).
- `InjectedToolArg`: `from langchain_core.tools import InjectedToolArg` (also `InjectedToolCallId`; `InjectedState`, `InjectedStore` from `langchain.tools`) — `Annotated[T, InjectedToolArg]` hides an arg from the model; caller supplies it.
- `get_runtime`: `from langgraph.runtime import get_runtime`; `def get_runtime(context_schema: type[ContextT] | None = None) -> Runtime[ContextT]` — callable inside a tool/node body.

```python
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain.tools import ToolRuntime

@tool
def lookup(query: str, runtime: ToolRuntime, config: RunnableConfig) -> str:
    """Look things up."""
    user_id = config["configurable"].get("user_id")          # via injected config
    same = runtime.config["configurable"].get("user_id")     # via ToolRuntime
    return f"{query} for {user_id}"
```