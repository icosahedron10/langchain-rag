# deepagents 0.7.5 — API reference (from installed source)

## 1. create_deep_agent

Import: `from deepagents import create_deep_agent` (defined in `deepagents.graph`).

```python
def create_deep_agent(
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    subagents: Sequence[SubAgent | CompiledSubAgent | AsyncSubAgent] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    permissions: list[FilesystemPermission] | None = None,
    backend: BackendProtocol | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | dict[str, Any] | None = None,
    state_schema: type[DeepAgentState] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph[AgentState[ResponseT], ContextT, InputAgentState, OutputAgentState[ResponseT]]
```

**No async variant exists** (no `acreate_deep_agent` anywhere in the package). The returned `CompiledStateGraph` is sync+async: use `.invoke/.ainvoke/.stream/.astream`.

## 2. Default middleware / built-in tools

Assembled stack (graph.py:816-893), in order:
- `SkillsMiddleware(backend=..., sources=skills)` — only if `skills=` passed.
- `FilesystemMiddleware(backend=..., custom_tool_descriptions=..., _permissions=...)` — **always**. Tools: `ls, read_file, write_file, edit_file, delete, glob, grep, execute`.
- `SubAgentMiddleware(...)` — if any inline subagents exist; a default `general-purpose` subagent is auto-added, so in practice always, providing the `task` tool.
- `create_summarization_middleware(model, backend)` (`deepagents.middleware.summarization`) — always.
- `PatchToolCallsMiddleware()` — always.
- `AsyncSubAgentMiddleware(async_subagents=...)` — only if `subagents` contains `AsyncSubAgent` (has `graph_id`).
- *your `middleware=` entries inserted here* — an entry whose `.name` matches an existing one **replaces it in place** (`_apply_custom_middleware`, graph.py:201).
- Harness-profile `extra_middleware`, `_ToolExclusionMiddleware` (if profile has `excluded_tools`), `AnthropicPromptCachingMiddleware` (unconditional; no-op for non-Anthropic; Bedrock/Fireworks variants if those packages are installed), `MemoryMiddleware` (if `memory=`), `HumanInTheLoopMiddleware` (if `interrupt_on=` or any `mode="interrupt"` permission).

**There is NO planning/todo middleware in 0.7.5.** `TodoListMiddleware` appears only in a stale comment (graph.py:634); no `write_todos` tool is attached.

Omitting/disabling:
- `FilesystemMiddleware` and `SubAgentMiddleware` are protected (`_REQUIRED_MIDDLEWARE`, graph.py:238): excluding them via a profile raises `ValueError`. They cannot be removed through `create_deep_agent`; compose manually with `langchain.agents.create_agent` instead.
- Restrict/replace filesystem tools: pass your own instance in `middleware=` — name-match replaces the default: `FilesystemMiddleware(tools=["read_file","grep"])` (allowlist of `FsToolName = Literal["ls","read_file","write_file","edit_file","delete","glob","grep","execute"]`; `"read_file"` is mandatory).
- Remove `task`: register a `HarnessProfile` with `general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)` (via `register_harness_profile`) and pass no sync subagents.
- Drop other middleware (e.g. summarization) or built-in tool names: `HarnessProfile.excluded_middleware` (class or `.name` string) / `HarnessProfile.excluded_tools`. `tools=` param is purely additive.

## 3. Backends (`deepagents.backends`)

Exports (`backends/__init__.py`): `BackendProtocol`, `CompositeBackend`, `ContextHubBackend`, `FilesystemBackend`, `LangSmithSandbox`, `LocalShellBackend`, `NamespaceFactory`, `StateBackend`, `StoreBackend`, `DEFAULT_EXECUTE_TIMEOUT`. Plus `SandboxBackendProtocol` and `BaseSandbox` in `deepagents.backends.protocol` / `deepagents.backends.sandbox`.

`BackendProtocol` (abc.ABC, `deepagents.backends.protocol`) — methods a custom backend implements (all default to `raise NotImplementedError`; async `a*` versions default to `asyncio.to_thread` wrappers):

```python
def ls(self, path: str) -> LsResult
def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult
def grep(self, pattern: str, path: str | None = None, glob: str | None = None, *, max_count: int | None = None) -> GrepResult
def glob(self, pattern: str, path: str | None = None) -> GlobResult
def write(self, file_path: str, content: str) -> WriteResult
def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult
def delete(self, file_path: str) -> DeleteResult            # optional; gated by _supports_delete()
def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]
def download_files(self, paths: list[str]) -> list[FileDownloadResponse]
```

`SandboxBackendProtocol(BackendProtocol)` adds:

```python
@property
def id(self) -> str
def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse
async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse
```

`ExecuteResponse` (dataclass): `output: str`, `exit_code: int | None = None`, `truncated: bool = False`.

`BaseSandbox(SandboxBackendProtocol, ABC)` (`deepagents.backends.sandbox:952`): implements all file ops via shell; subclasses implement only `execute()`, `upload_files()`, `download_files()`, and `id`. Also has `execute_with_offload(command, capture_path, *, max_inline_bytes, max_capture_bytes=None, timeout=None) -> ExecuteOffloadResult` and class attr `enable_capture_offload: bool = False`.

Constructors:
- `StateBackend()` — no args; files live in graph state (`invoke(files={...})`). Default backend.
- `FilesystemBackend(root_dir: str | Path | None = None, virtual_mode: bool = True, max_file_size_mb: int = 10)`
- `LocalShellBackend(root_dir=None, *, virtual_mode=True, timeout=DEFAULT_EXECUTE_TIMEOUT, max_output_bytes=100_000, env=None, inherit_env=False)` — subclasses `FilesystemBackend, SandboxBackendProtocol`; the shipped way to get a working `execute` locally (unsandboxed).
- `CompositeBackend(default: BackendProtocol, routes: dict[str, BackendProtocol], *, artifacts_root: str = "/")` — prefix routing, e.g. `{"/memories/": store_backend}`.
- `StoreBackend(*, namespace: NamespaceFactory, store: BaseStore | None = None)` where `NamespaceFactory = Callable[[Runtime[Any]], tuple[str, ...]]`.
- `LangSmithSandbox(sandbox: Sandbox)` — wraps a LangSmith sandbox; subclass of `BaseSandbox`.

**How `execute` is exposed:** `FilesystemMiddleware.__init__` always builds the `execute` tool (filesystem.py:1713) unless excluded via `tools=`. In `wrap_model_call`, `supports_execution(backend)` (filesystem.py:1421 — `isinstance(backend, SandboxBackendProtocol)`, or for `CompositeBackend` an isinstance check on `backend.default`) hides `execute` from the model's tool list when False; the tool body re-checks at call time and returns a `status="error"` ToolMessage ("Execution not available...") if unsupported. So: pass a `SandboxBackendProtocol` backend → the agent gets a live `execute(command, timeout=None)` tool.

## 4. Per-run backend resolution

**Not supported.** Backend factories were removed in 0.7: `FilesystemMiddleware.__init__` raises `TypeError("backend must be an initialized backend instance. Backend factories were removed in deepagents 0.7; pass StateBackend(), CompositeBackend(...), or another BackendProtocol instance instead.")` if given a callable (filesystem.py:1660). `create_deep_agent` likewise takes only `backend: BackendProtocol | None`. The only per-run hook is `StoreBackend`'s `namespace` factory, which receives the LangGraph `Runtime` at call time (e.g. `lambda rt: (rt.server_info.user.identity, "filesystem")`).

`FilesystemMiddleware` full signature (`deepagents.middleware.filesystem`):

```python
FilesystemMiddleware(*, backend: BackendProtocol | None = None, system_prompt: str | None = None,
    custom_tool_descriptions: Mapping[str, str] | None = None,
    tool_token_limit_before_evict: int | None = 20000,
    human_message_token_limit_before_evict: int | None = 50000,
    max_execute_timeout: int = 3600, grep_max_count: int | None = 1000,
    tools: list[FsToolName] | Literal["all"] | None = None,
    _permissions: list[FilesystemPermission] | None = None)
```

## 5. model / tools / system_prompt / checkpointer / invocation

- `model`: `"provider:model"` string (resolved via `init_chat_model` + provider profiles) or a `BaseChatModel` instance. `model=None` is deprecated (falls back to `ChatAnthropic(model_name="claude-sonnet-4-6")`).
- `tools`: plain callables, `BaseTool`s, or dict schemas; merged additively with built-ins.
- `system_prompt`: `str` or `SystemMessage` (preserves `cache_control` blocks); placed before any harness-profile base/suffix. No authored default prompt in 0.7 (`BASE_AGENT_PROMPT` deprecated).
- `checkpointer`/`store`/`cache`/`debug`/`name`: passed straight through to `langchain.agents.create_agent`.
- Returns a `langgraph` `CompiledStateGraph`, pre-configured with `recursion_limit=9_999` via `.with_config`. State schema defaults to `DeepAgentState` (messages use a `DeltaChannel` reducer).

```python
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langgraph.checkpoint.memory import InMemorySaver

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[my_tool],
    system_prompt="You are a helpful research agent.",
    backend=LocalShellBackend(root_dir="/tmp/work"),   # enables the `execute` tool
    checkpointer=InMemorySaver(),
)
cfg = {"configurable": {"thread_id": "t1"}}
result = agent.invoke({"messages": [{"role": "user", "content": "hi"}]}, config=cfg)
for chunk in agent.stream({"messages": [...]}, config=cfg, stream_mode="values"):
    ...
# StateBackend file seeding: agent.invoke({"messages": [...], "files": {"/notes.txt": "..."}})
```

## 6. Middleware exported for manual composition

From `deepagents` (top level): `FilesystemMiddleware`, `SubAgentMiddleware`, `AsyncSubAgentMiddleware`, `MemoryMiddleware`, `RubricMiddleware` (+ types `SubAgent`, `CompiledSubAgent`, `AsyncSubAgent`, `FilesystemPermission`, `FsToolName`, `DeepAgentState`).

From `deepagents.middleware`: all of the above plus `SkillsMiddleware`, `SummarizationMiddleware`, `SummarizationToolMiddleware`, `create_summarization_tool_middleware`, `DEEPAGENTS_DEFAULT_SUMMARY_PROMPT`, rubric types. Not in `__all__` but importable: `deepagents.middleware.patch_tool_calls.PatchToolCallsMiddleware`, `deepagents.middleware.summarization.create_summarization_middleware`.

Note: `SubAgentMiddleware(*, backend, subagents, system_prompt=None, task_description=None, private_state_keys=None, state_schema=None)` requires at least one subagent spec. Example manual composition: `create_agent(model, middleware=[FilesystemMiddleware(backend=StateBackend())])`.

Key files: `.../deepagents/graph.py`, `.../deepagents/backends/protocol.py`, `.../deepagents/backends/sandbox.py`, `.../deepagents/middleware/filesystem.py`, `.../deepagents/middleware/subagents.py` (all under `C:/Users/madse/Documents/langchain-rag/.venv/Lib/site-packages/`).