# langgraph 1.2.10 API reference (verified against installed source)

## 1. get_stream_writer
Import: `from langgraph.config import get_stream_writer`
Source `langgraph/config.py:126` (implementation, docstring elided):
```python
def get_stream_writer() -> StreamWriter:
    runtime = get_config()[CONF][CONFIG_KEY_RUNTIME]
    return runtime.stream_writer

def get_config() -> RunnableConfig:
    ...
    if var_config := var_child_runnable_config.get():
        return var_config
    raise RuntimeError("Called get_config outside of a runnable context")
```
Mechanism: reads the **`var_child_runnable_config` contextvar** (`langchain_core.runnables.config`), then `config["configurable"]["__pregel_runtime"]` (`CONFIG_KEY_RUNTIME = "__pregel_runtime"`, `langgraph/_internal/_constants.py:70`) → `Runtime.stream_writer`. There is no separate `CONFIG_KEY_STREAM_WRITER` key in 1.2.10; the writer lives on the `Runtime` object. `CONFIG_KEY_STREAM = "__pregel_stream"` holds a `StreamProtocol` for parent→child forwarding.
Works inside a `@tool` under `create_agent`: **yes**. Task configs carry `CONFIG_KEY_RUNTIME` (`pregel/_algo.py:924`), and `BaseTool._arun` wraps the tool coroutine in `set_config_context(child_config)` + `coro_with_context` (`langchain_core/tools/base.py:1185-1196`), so the contextvar is set. Requires Python ≥3.11 for async. Alternative in tools: `runtime: ToolRuntime` param → `runtime.stream_writer` (injected at `langgraph/prebuilt/tool_node.py:812`).

## 2. Nested inner-graph invoked inside an outer tool, outer consumed via `outer.astream(x, stream_mode=["custom","messages"])`
Config propagation into `inner.ainvoke()` with no config arg works (see #5). What matters is `subgraphs=True` on the **outer** astream:

**(a) custom — YES only with `subgraphs=True` on outer astream.** Trace (`pregel/main.py:3283-3325`):
- Outer with `"custom"` in modes builds a real `stream_writer` closure putting `(ns, "custom", c)` into the outer queue; stored in `Runtime` under `CONFIG_KEY_RUNTIME`.
- Outer sets `loop.config[CONF][CONFIG_KEY_STREAM] = StreamProtocol(stream_put, stream_modes)` **only `if subgraphs:`** (`main.py:3388-3391`).
- Inner astream (via `ainvoke` → `stream_mode="values"` because `CONFIG_KEY_TASK_ID` present, `main.py:3145-3152`): `"custom"` not in its modes, so it hits `elif CONFIG_KEY_STREAM in config[CONF]: stream_writer = config[CONF][CONFIG_KEY_RUNTIME].stream_writer` — i.e. it reuses the **outer** writer → inner tools' `get_stream_writer()` write straight into the outer queue.
- Without `subgraphs=True`, `CONFIG_KEY_STREAM` is absent and the inner astream falls to the `else` branch defining a **local no-op** `def stream_writer(c): pass`; `parent_runtime.merge(runtime)` keeps it (merge only restores the parent's writer when the child's is the module-level `_no_op_stream_writer` sentinel — `langgraph/runtime.py:248-250` — which this local function is not). Inner custom events are silently dropped.

**(b) messages — YES only with `subgraphs=True`.** Outer registers `StreamMessagesHandler(stream_put, subgraphs, parent_ns=...)` on `run_manager.inheritable_handlers` (`main.py:3249-3255`). Inheritable callbacks flow into the inner graph's LLM runs via the tool's `child_config` + contextvar. But the handler filters in `on_chat_model_start` (`pregel/_messages.py:141-149`):
```python
ns = tuple(metadata["langgraph_checkpoint_ns"].split(NS_SEP))[:-1]
if not self.subgraphs and len(ns) > 0 and ns != self.parent_ns:
    return
```
The inner graph is nested (`CONFIG_KEY_TASK_ID` in config ⇒ `is_nested`, `_loop.py:314`), so its LLM runs have a deep `langgraph_checkpoint_ns` (`tools:<task_id>|model:<task_id>`) → `len(ns) > 0` → dropped unless `subgraphs=True`. Tokens marked with tag `"nostream"` (`TAG_NOSTREAM`) are always dropped.

Other requirements/caveats:
- With `subgraphs=True` + list stream_mode, yields become 3-tuples `(namespace, mode, payload)` (see #3).
- Checkpointer inheritance: task config injects `CONFIG_KEY_CHECKPOINTER: checkpointer or configurable.get(CONFIG_KEY_CHECKPOINTER)` (`_algo.py:914`), and `_defaults` prefers it (`main.py:2581`). So an inner graph compiled **without** a checkpointer inherits the outer's and writes under the outer thread's namespaced checkpoints; compile inner with `.compile(checkpointer=False)` to opt out (`if self.checkpointer is False: checkpointer = None`, `main.py:2579`). No error occurs either way (thread_id is inherited).
- Also fine: inner loop duplexes its own stream into the parent's when `CONFIG_KEY_STREAM` present: `self.stream = DuplexStream(self.stream, config[CONF][CONFIG_KEY_STREAM])` (`_loop.py:323-324`), forwarding only modes in the parent's `StreamProtocol.modes`.

Minimal working pattern:
```python
async for ns, mode, payload in outer.astream(inp, cfg, stream_mode=["custom", "messages"], subgraphs=True):
    ...
# inside the outer tool: await inner.ainvoke({"messages": [...]})   # no config arg needed (py>=3.11)
```
(Alternative without `subgraphs=True`: the tool itself iterates `inner.astream(..., stream_mode="custom")` and re-emits chunks through the outer `get_stream_writer()`; for messages there is no non-subgraphs alternative except the handler's `parent_ns` special case, which only applies when the *explicit inner astream* runs in the same namespace the handler was created in, i.e. root-level nodes.)

## 3. Multi-mode yield shape and messages metadata
`_output` (`pregel/main.py:4236-4243`, default `version="v1"`):
- `stream_mode=["messages","custom"]`, `subgraphs=False` → yields `(mode, payload)` 2-tuples.
- with `subgraphs=True` → `(ns, mode, payload)` 3-tuples, `ns: tuple[str, ...]` e.g. `("tools:<task_id>",)`.
- `"messages"` payload is a 2-tuple `(message_chunk, metadata_dict)` — `self.stream((meta[0], "messages", (message, meta[1])))` (`_messages.py:104`). Token chunks are `AIMessageChunk` (from `ChatGenerationChunk.message`); full messages emitted at `on_llm_end`/node output are deduped by `message.id`.
- metadata dict = LangChain run metadata for the LLM run, containing: `langgraph_step`, `langgraph_node`, `langgraph_triggers`, `langgraph_path`, `langgraph_checkpoint_ns` (`_algo.py:847-853`); `ls_provider`, `ls_model_name`, `ls_model_type`, `ls_temperature`, ... (from `BaseChatModel._get_ls_params`); `ls_integration: "langgraph"`; `checkpoint_ns` (copied from configurable by `ensure_config`, `langchain_core/runnables/config.py:297-308`); plus any user `config["metadata"]` (`thread_id` only if you put it there). **`tags`**: the handler injects `metadata["tags"] = filter_to_user_tags(tags)` (`_messages.py:147-148`) — tags from `model.with_config(tags=["inner_llm"])` appear there (internal `seq:step:N` tags stripped). There is no top-level `checkpoint_ns` key beyond the two above; use `langgraph_checkpoint_ns` for namespace filtering.
```python
async for mode, payload in outer.astream(inp, cfg, stream_mode=["messages", "custom"]):
    if mode == "messages":
        chunk, meta = payload
        if "inner_llm" in (meta.get("tags") or []): ...
```

## 4. InMemorySaver
Import: `from langgraph.checkpoint.memory import InMemorySaver` (also re-exported as `MemorySaver = InMemorySaver  # Kept for backwards compatibility`, `checkpoint/memory/__init__.py:625`).
```python
def __init__(self, *, serde: SerializerProtocol | None = None,
             factory: type[defaultdict] = defaultdict) -> None
def delete_thread(self, thread_id: str) -> None
async def adelete_thread(self, thread_id: str) -> None          # calls delete_thread
def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None
async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None
```
Reading state in tests:
```python
saver = InMemorySaver()
graph = builder.compile(checkpointer=saver)
cfg = {"configurable": {"thread_id": "t1"}}
await graph.ainvoke(inp, cfg)
snap = await graph.aget_state(cfg)          # StateSnapshot; sig: aget_state(self, config: RunnableConfig, *, subgraphs: bool = False)
snap.values["messages"]
tup = await saver.aget_tuple(cfg)           # raw CheckpointTuple (config, checkpoint, metadata, parent_config, pending_writes)
tup.checkpoint["channel_values"]
await saver.adelete_thread("t1")
```

## 5. Auto-propagation of parent config into `inner.ainvoke()` without a config arg — YES (Python async ≥3.11)
Proof, `langchain_core/tools/base.py:1185-1196` (tool execution sets the contextvar):
```python
child_config = patch_config(config, callbacks=run_manager.get_child())
with set_config_context(child_config) as context:
    ...
    coro = self._arun(*tool_args, **tool_kwargs)
    response = await coro_with_context(coro, context)
```
and `langchain_core/runnables/config.py:264-292` — `ensure_config(config=None)` (called at the top of `Pregel.astream`, `main.py:3163`) merges the contextvar:
```python
if var_config := var_child_runnable_config.get():
    empty.update({k: v.copy() if k in COPIABLE_KEYS else v
                  for k, v in var_config.items() if v is not None})
```
`COPIABLE_KEYS = ["tags", "metadata", "callbacks", "configurable"]` — so `callbacks` (incl. the outer's inheritable `StreamMessagesHandler`) and `configurable` (incl. `CONFIG_KEY_RUNTIME`, `CONFIG_KEY_STREAM`, `CONFIG_KEY_TASK_ID`, `CONFIG_KEY_CHECKPOINTER`, `thread_id`, checkpoint ns) all flow into the inner run automatically. On Python <3.11 async, contextvars don't propagate across tasks — pass `runtime.config` / the injected `config` explicitly there.

## 6. Runtime context
`from langgraph.runtime import get_runtime, Runtime` — `get_runtime(context_schema: type[ContextT] | None = None) -> Runtime[ContextT]` (`runtime.py:296`), reads `get_config()[CONF][CONFIG_KEY_RUNTIME]`.
`create_agent` (import: `from langchain.agents import create_agent`, langchain 1.3.14 `agents/factory.py:786`) accepts `context_schema: type[ContextT] | None = None`; the compiled graph's `ainvoke`/`astream` accept a keyword-only `context: ContextT | None = None` (`pregel/main.py:3068`), coerced by `_coerce_context` (dict → dataclass/pydantic) and stored on `Runtime.context` (`main.py:3315-3325`). Nested graphs inherit parent context unless they pass their own (`Runtime.merge`: `context=other.context or self.context`).
```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import ToolRuntime   # = langgraph.prebuilt.ToolRuntime

@dataclass
class Ctx: user_id: str

def my_tool(x: int, runtime: ToolRuntime[Ctx, dict]) -> str:
    """Doc."""
    return runtime.context.user_id        # also: runtime.config, runtime.state, runtime.tool_call_id, runtime.stream_writer

agent = create_agent(model, tools=[my_tool], context_schema=Ctx)
await agent.ainvoke({"messages": [...]}, context=Ctx(user_id="u1"))
# or inside any callable: from langgraph.runtime import get_runtime; get_runtime(Ctx).context.user_id
```
So `context=` is fully supported; `config["configurable"]` is NOT needed for user context (still used for `thread_id` etc.). `ToolRuntime` fields (`langgraph/prebuilt/tool_node.py:1663`): `state, context, config, stream_writer, tool_call_id, store, tools, execution_info, server_info`, plus `emit_output_delta(delta)` for `stream_mode="tools"`.
