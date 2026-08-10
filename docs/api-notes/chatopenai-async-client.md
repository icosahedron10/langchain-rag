# API Reference — langchain_openai 1.4.2 / openai 2.53.0 (verified from installed source)

Source files: `.venv/Lib/site-packages/langchain_openai/chat_models/base.py`, `.venv/Lib/site-packages/openai/_client.py`

## 1. ChatOpenAI client fields and injection

Import: `from langchain_openai import ChatOpenAI` (re-export of `langchain_openai.chat_models.base.ChatOpenAI`, subclass of `BaseChatOpenAI`).

Fields (base.py:634-640) — names unchanged, all typed `Any`:
```python
client: Any = Field(default=None, exclude=True)              # OpenAI().chat.completions
async_client: Any = Field(default=None, exclude=True)        # AsyncOpenAI().chat.completions
root_client: Any = Field(default=None, exclude=True)         # openai.OpenAI instance
root_async_client: Any = Field(default=None, exclude=True)   # openai.AsyncOpenAI instance
```
`model_config = ConfigDict(populate_by_name=True)` (line 1118) — pass by field name as constructor kwargs.

`validate_environment` is a `@model_validator(mode="after")` (base.py:1199). Client-construction part (base.py:1296-1341, verbatim, trimmed to the async half):
```python
if not self.async_client:
    if self.openai_proxy and not self.http_async_client:
        self.http_async_client = _build_proxied_async_httpx_client(...)
    async_specific = {
        "http_client": self.http_async_client
        or _get_default_async_httpx_client(
            self.openai_api_base, self.request_timeout, resolved_socket_options,
        ),
        "api_key": async_api_key_value,
    }
    self.root_async_client = openai.AsyncOpenAI(
        **client_params,
        **async_specific,  # type: ignore[arg-type]
    )
    self.async_client = self.root_async_client.chat.completions
```
The guard is `if not self.async_client:` — **only `async_client` being set skips creation**. Passing `root_async_client` alone does NOT skip: `self.async_client` is still None, so the block runs and **overwrites your `root_async_client`** with a fresh default `AsyncOpenAI`. A caller-passed `AsyncOpenAI` is honored only via this pattern — pass BOTH:
```python
import openai
from langchain_openai import ChatOpenAI

ac = openai.AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
llm = ChatOpenAI(
    model="my-model",
    api_key="EMPTY",                          # still set; avoids env fallback for the sync client
    root_async_client=ac,                     # used by responses API / structured-output paths
    async_client=ac.chat.completions,         # skips default AsyncOpenAI construction
)
```
Both are needed: `_astream`/`_agenerate` normally call `self.async_client.create(...)`, but the `response_format`/Responses-API branches call `self.root_async_client.beta.chat.completions.stream(...)` / `self.root_async_client.responses...` (lines 1916, 1984, 1992-1998). Sync analog: pass `client=` (guard `if not self.client:` at 1296) and `root_client=`.

## 2. openai.AsyncOpenAI constructor (openai/_client.py:734-764)

```python
class AsyncOpenAI(AsyncAPIClient):
    def __init__(self, *,
        api_key: str | Callable[[], Awaitable[str]] | None = None,
        admin_api_key: str | None = None,
        workload_identity: WorkloadIdentity | None = None,
        organization: str | None = None,
        project: str | None = None,
        webhook_secret: str | None = None,
        provider: _Provider | None = None,
        base_url: str | httpx.URL | None = None,
        websocket_base_url: str | httpx.URL | None = None,
        timeout: float | Timeout | None | NotGiven = not_given,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        http_client: httpx.AsyncClient | None = None,
        _strict_response_validation: bool = False,
        _enforce_credentials: bool = True,
    ) -> None:
```
`base_url` and `api_key` are unchanged. New in 2.x but optional: `admin_api_key`, `workload_identity`, `provider` (`provider` is mutually exclusive with `api_key`/`base_url`, raises `OpenAIError`). If no `api_key`/env/`admin_api_key`/`workload_identity` at all, `__init__` raises `OpenAIError("Missing credentials. ...")` (lines 829-839). `AsyncOpenAI(base_url="http://host:8000/v1", api_key="EMPTY")` is valid.

## 3. Async path client usage

Confirmed: `_agenerate` (base.py:1971) awaits `self.async_client.with_raw_response.create(**payload)` (line 2012) or `self.root_async_client.chat.completions.with_raw_response.parse` / `self.root_async_client.responses.*` for structured/Responses branches. `_astream` (base.py:1891) awaits `self.async_client.create(**payload)` (line 1928) or `self.root_async_client.beta.chat.completions.stream` when `response_format` is set (line 1916). `ChatOpenAI._astream` (line 3522) only routes between `super()._astream` and `super()._astream_responses` (which uses `root_async_client`, line 1557-1564). **No sync `client`/`root_client` reference appears anywhere in the async methods** — sync client is untouched during `ainvoke`/`astream`; sync `_generate`/`_stream` use `self.client`/`self.root_client` exclusively.

## 4. vLLM-relevant kwargs (all on BaseChatOpenAI)

- `model_name: str = Field(default="gpt-3.5-turbo", alias="model")` — pass `model="..."`.
- `openai_api_key: SecretStr | None | Callable[[], str] | Callable[[], Awaitable[str]] = Field(alias="api_key", default=None)` — pass `api_key=`; a plain `str` is accepted and coerced to `SecretStr` by pydantic; sync/async callables also accepted.
- `openai_api_base: str | None = Field(default=None, alias="base_url")` — pass `base_url=`. Resolution: explicit kwarg > `OPENAI_API_BASE` env > `OPENAI_BASE_URL` env (read by the openai SDK).
- `temperature: float | None = None`.
- `streaming: bool = False` (line 775; forces streaming inside invoke). `stream_usage: bool | None = None` (line 733) — auto-set to `True` only when base_url/proxy/clients are all unset (lines 1230-1246), so **with a vLLM base_url it stays off by default**; pass `stream_usage=True` explicitly (vLLM supports `stream_options.include_usage`).
- `extra_body: Mapping[str, Any] | None = None` (line 966) — documented as the correct channel for vLLM params, e.g. `extra_body={"use_beam_search": True, "best_of": 4}`; do not use `model_kwargs` for non-standard params.
- `use_responses_api: bool | None = None` (line 1090) — leave unset/False for vLLM so Chat Completions is used.
- **Passing `root_async_client` alone does NOT skip default async client creation** (see §1 — the guard checks `async_client` only, and your `root_async_client` gets overwritten). Pass `async_client=` (and `root_async_client=`) together.

```python
llm = ChatOpenAI(
    model="meta-llama/Llama-3.1-8B-Instruct",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    temperature=0.2,
    stream_usage=True,
    extra_body={"repetition_penalty": 1.05},
)
```

## 5. bind_tools (base.py:2221)

```python
def bind_tools(
    self,
    tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
    *,
    tool_choice: dict | str | bool | None = None,
    strict: bool | None = None,
    parallel_tool_calls: bool | None = None,
    response_format: _DictOrPydanticClass | None = None,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, AIMessage]:
```
`tool_choice` options: tool name str; `"auto"`; `"none"`; `"any"` / `"required"` / `True` (force a tool); `{"type": "function", "function": {"name": ...}}`; `False`/`None` = provider default. `parallel_tool_calls=False` disables parallel tool use (`None` = unspecified).

Streaming chunks: `_convert_delta_to_message_chunk` (base.py:470-506) returns `AIMessageChunk(content=..., additional_kwargs=..., id=..., tool_call_chunks=[tool_call_chunk(name=..., args=..., id=..., index=...), ...])` built from `delta.tool_calls`. Confirmed: `astream` yields `ChatGenerationChunk`s whose `.message` is `AIMessageChunk` carrying both `content` and `tool_call_chunks`.

```python
bound = llm.bind_tools([my_tool], tool_choice="auto", parallel_tool_calls=False)
async for chunk in bound.astream("..."):   # chunk: AIMessageChunk
    chunk.content; chunk.tool_call_chunks
```

## 6. OPENAI_API_KEY env requirement

There is no `__init__` override or `model_post_init`; init-time logic is the two `@model_validator(mode="after")` hooks (`set_langchain_version`, `validate_environment`). When `api_key=` is given, `_resolve_gateway_config` (langchain_core/utils/_gateway.py:149-150) returns the explicit key unchanged — **env is never consulted; nothing requires `OPENAI_API_KEY` when `base_url` and `api_key` kwargs are provided**, and no init-time validation error exists for a missing env var (missing sync key just leaves `client=None` with an error deferred to invocation, base.py:1297-1301). One indirect path can still hit env: with NO api_key resolvable anywhere and no injected `async_client`, langchain constructs `openai.AsyncOpenAI(api_key=None)`, which falls back to `os.environ["OPENAI_API_KEY"]` and raises `OpenAIError("Missing credentials. ...")` at ChatOpenAI construction. Any non-empty `api_key` string (e.g. `"EMPTY"`) avoids this.

Note: ChatOpenAI's docstring (line 2633-2641) warns it targets official OpenAI spec only — third-party fields like vLLM `reasoning_content` are not extracted; core chat/tool-calling/extra_body flows above work regardless.