# litestar 2.24.0 — verified against installed source

## 1. SSE

Imports: `from litestar.response import ServerSentEvent, ServerSentEventMessage` (defined in `litestar/response/sse.py`; also importable as `from litestar.response.sse import ...`).

```python
class ServerSentEvent(Stream):
    def __init__(self, content: str | bytes | StreamType[SSEData], *,
        background: BackgroundTask | BackgroundTasks | None = None,
        cookies: ResponseCookies | None = None, encoding: str = "utf-8",
        headers: ResponseHeaders | None = None, event_type: str | None = None,
        event_id: int | str | None = None, retry_duration: int | None = None,
        comment_message: str | None = None, status_code: int | None = None) -> None
```

```python
@dataclass
class ServerSentEventMessage:
    data: str | int | bytes | None = ""
    event: str | None = None
    id: int | str | None = None
    retry: int | None = None
    comment: str | None = None
    sep: str = "\r\n"   # DEFAULT_SEPARATOR
```

`SSEData` (from `litestar.types`) = `Union[int, str, bytes, Dict[str, Any], ServerSentEventMessage]`. Yielded dicts are turned into `ServerSentEventMessage(**d)`; bare str/int/bytes yields inherit `event_type`/`event_id`/`retry_duration` from the `ServerSentEvent` constructor. Per-message event name vs data: set `event=` and `data=` on `ServerSentEventMessage`.

```python
from typing import AsyncGenerator
from litestar import get
from litestar.response import ServerSentEvent, ServerSentEventMessage

@get("/stream")
async def stream() -> ServerSentEvent:
    async def gen() -> AsyncGenerator[ServerSentEventMessage, None]:
        yield ServerSentEventMessage(data="hello", event="greeting", id=1)
        yield "plain data line"          # data-only message
    return ServerSentEvent(gen())        # pass the generator OBJECT
```

Media type is hardcoded `media_type="text/event-stream"`; headers set in `__init__`: `Cache-Control: no-cache` (setdefault), `Connection: keep-alive`, `X-Accel-Buffering: no`. Status defaults to 200.

## 2. Controller

`from litestar import Controller, get, post, delete` (decorators live in `litestar.handlers.http_handlers.decorators`; also re-exported from `litestar.handlers`). Decorators are classes: `class delete(HTTPRouteHandler)`, `class get(...)`, `class post(...)`, first param `path: str | None | Sequence[str] = None`, keyword-only rest including `dependencies`, `exception_handlers: ExceptionHandlersMap | None = None`, `status_code: int | None = None`, `guards`, `media_type`, `middleware`, etc.

Controller class attributes (from `litestar/controller.py` — all set as plain class attrs on your subclass): `path: str`, `dependencies: Dependencies | None`, `exception_handlers: ExceptionHandlersMap | None` (yes, controller-level), plus `after_request`, `after_response`, `before_request`, `guards`, `middleware`, `opt`, `parameters`, `response_class`, `response_cookies`, `response_headers`, `dto`, `return_dto`, `tags`, `security`, `signature_namespace`, `signature_types`, `type_encoders`, `type_decoders`, `request_class`, `websocket_class`, `include_in_schema`, `request_max_body_size`, `cache_control`, `etag`.

Path parameter syntax REQUIRES a type: `{name:type}` where type ∈ `str, int, float, uuid, decimal, date, datetime, time, timedelta, path` (`param_type_map` in `litestar/routes/base.py`; `{id}` without `:type` raises `ImproperlyConfiguredException`).

Default status codes (`get_default_status_code` in `handlers/http_handlers/_utils.py`): POST → 201, DELETE → 204, else 200. So `@post` is already 201 and `@delete` already 204 — pass `status_code=` only to override.

```python
from litestar import Controller, get, post, delete

class ItemController(Controller):
    path = "/items"
    dependencies = {}          # {"name": Provide(...)}
    exception_handlers = {}    # {SomeError: handler_fn}

    @post()                                    # 201 by default
    async def create(self, data: dict) -> dict: ...

    @get("/{item_id:str}")                     # 200
    async def retrieve(self, item_id: str) -> dict: ...

    @delete("/{item_id:str}")                  # 204; return type MUST be None
    async def remove(self, item_id: str) -> None: ...
```

Note: 204 handlers must be annotated `-> None` or Litestar raises `ImproperlyConfiguredException` at startup.

## 3. Lifespan and state

`Litestar(...)` parameter (from `litestar/app.py` `__init__`):
`lifespan: Sequence[Callable[[Litestar], AbstractAsyncContextManager] | AbstractAsyncContextManager] | None = None`
Callables are invoked with the app instance (`manager = manager(self)`) and entered on an `AsyncExitStack`.

`app.state` is `litestar.datastructures.State` (`from litestar.datastructures import State`), a `MutableMapping` with attribute access; set via `Litestar(state=State({...}))` or mutate inside lifespan.

```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from litestar import Litestar, get
from litestar.datastructures import State
from litestar.di import Provide

@asynccontextmanager
async def db_lifespan(app: Litestar) -> AsyncGenerator[None, None]:
    app.state.db = await connect()      # set on app.state
    try:
        yield
    finally:
        await app.state.db.close()

# (a) inject whole State via reserved kwarg — name must be exactly `state`
@get("/a")
async def handler_a(state: State) -> str:
    return str(state.db)

# (b) dependency that pulls from state
async def provide_db(state: State):    # dependencies may also use reserved kwargs
    return state.db

@get("/b")
async def handler_b(db) -> str: ...

app = Litestar(route_handlers=[handler_a, handler_b],
               lifespan=[db_lifespan],
               dependencies={"db": Provide(provide_db)})
```

`Provide` (from `litestar.di`): `Provide(dependency: AnyCallable | type[Any], use_cache: bool = False, sync_to_thread: bool | None = None)`.

## 4. exception_handlers

Type (from `litestar/types/callable_types.py`): `ExceptionHandler: TypeAlias = "Callable[[Request, ExceptionT], Response]"` — handler takes `(request, exc)` and returns a `Response`. Map: `ExceptionHandlersMap = Mapping[int | type[Exception], ExceptionHandler]` (keys may be status codes or exception classes).

```python
from litestar import Litestar, MediaType, Request, Response

class NotFoundError(Exception): ...

def not_found_handler(request: Request, exc: NotFoundError) -> Response:
    return Response(
        content={"error": "not_found", "detail": str(exc)},
        status_code=404,
        media_type=MediaType.JSON,   # dict content is JSON-serialized by default anyway
    )

app = Litestar(route_handlers=[...],
               exception_handlers={NotFoundError: not_found_handler})
```

`Response.__init__` (from `litestar/response/base.py`): `Response(content, *, background=None, cookies=None, encoding="utf-8", headers=None, media_type=None, status_code=None, type_encoders=None)`.

## 5. Testing

`from litestar.testing import TestClient, create_test_client` (also `AsyncTestClient`, `create_async_test_client`).

`TestClient(app, base_url="http://testserver.local", raise_server_exceptions=True, root_path="", backend="asyncio", backend_options=None, session_config=None, timeout=None, cookies=None)` — subclasses `httpx.Client`, so all httpx methods work.

Lifespan: NOT run on construction. `TestClient.__enter__` creates a `LifeSpanHandler` which sends `lifespan.startup` — so lifespan (and your `lifespan=[...]` managers) run **only when used as a context manager** (`with TestClient(app) as client:` / `with create_test_client(...) as client:`). Plain `client.get(...)` without `with` skips startup/shutdown.

`create_test_client(route_handlers=None, *, ..., lifespan: list[...] | None = None, on_startup=None, on_shutdown=None, state: State | None = None, debug: bool = True, raise_server_exceptions: bool = True, timeout=None, backend="asyncio", ...) -> TestClient[Litestar]` — builds a `Litestar` app from the kwargs and wraps it.

Streaming SSE — use httpx streaming API:

```python
with create_test_client([stream]) as client:
    with client.stream("GET", "/stream") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in response.iter_lines()]
        # e.g. ["id: 1", "event: greeting", "data: hello", "", "data: plain data line", ""]
```

(`client.get(...)` would also work but buffers the whole body; `stream` + `iter_lines()`/`iter_bytes()` is the streaming path. Note messages are separated by `\r\n`.)

## 6. Uvicorn + factory

```python
# app.py
from litestar import Litestar

def create_app() -> Litestar:
    return Litestar(route_handlers=[...], lifespan=[db_lifespan])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:create_app", factory=True, host="127.0.0.1", port=8000)
```

CLI equivalents: `uvicorn app:create_app --factory` or `litestar --app app:create_app run` (Litestar CLI auto-detects factories named `create_app`).