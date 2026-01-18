# FastAPI Advanced Patterns (December 2025)

Advanced patterns for production FastAPI applications.

## Table of Contents
- [Python 3.12+ Type Aliases](#python-312-type-aliases)
- [Middleware Patterns](#middleware-patterns)
- [Background Tasks](#background-tasks)
- [Streaming Responses](#streaming-responses)
- [WebSocket Patterns](#websocket-patterns)
- [Database Patterns with SQLAlchemy 2.0](#database-patterns-with-sqlalchemy-20)
- [Dependency Injection Patterns](#dependency-injection-patterns)

---

## Python 3.12+ Type Aliases

Use the `type` statement for cleaner type aliases (Python 3.12+):

```python
# ✅ Python 3.12+ (modern)
from typing import Annotated
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

type DB = Annotated[AsyncSession, Depends(get_db)]
type CurrentUser = Annotated[User, Depends(get_current_user)]

@router.get("/items")
async def list_items(db: DB, user: CurrentUser):
    ...

# Also works for complex types
type UserList = list[UserResponse]
type UserOrNone = User | None
```

---

## Middleware Patterns

### Timing Middleware

```python
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        return response

app.add_middleware(TimingMiddleware)
```

### Request ID Middleware

```python
import uuid
from contextvars import ContextVar

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request_id_ctx.set(request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
```

### CORS (use built-in)

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],  # Or ["*"] for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Background Tasks

### Simple Background Task

```python
from fastapi import BackgroundTasks

async def send_email(email: str, subject: str, body: str):
    """Send email asynchronously."""
    # Email sending logic
    ...

@router.post("/register")
async def register_user(
    user_data: UserCreate,
    background_tasks: BackgroundTasks,
    db: DB,
):
    user = await UserService.create(db, user_data)

    # Add task to run after response is sent
    background_tasks.add_task(
        send_email,
        email=user.email,
        subject="Welcome!",
        body="Thanks for signing up.",
    )

    return user
```

### Multiple Background Tasks

```python
@router.post("/orders")
async def create_order(
    order_data: OrderCreate,
    background_tasks: BackgroundTasks,
    db: DB,
):
    order = await OrderService.create(db, order_data)

    # Queue multiple tasks
    background_tasks.add_task(send_order_confirmation, order.id)
    background_tasks.add_task(notify_warehouse, order.id)
    background_tasks.add_task(update_analytics, "order_created")

    return order
```

---

## Streaming Responses

### Server-Sent Events (SSE)

```python
from fastapi.responses import StreamingResponse
import asyncio

async def event_generator():
    """Generate SSE events."""
    while True:
        data = await get_latest_data()
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(1)

@router.get("/events")
async def stream_events():
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
```

### Streaming File Downloads

```python
from pathlib import Path

async def file_chunk_generator(file_path: Path, chunk_size: int = 8192):
    """Yield file in chunks."""
    async with aiofiles.open(file_path, "rb") as f:
        while chunk := await f.read(chunk_size):
            yield chunk

@router.get("/download/{filename}")
async def download_file(filename: str):
    file_path = Path("files") / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found")

    return StreamingResponse(
        file_chunk_generator(file_path),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
```

---

## WebSocket Patterns

### Basic WebSocket

```python
from fastapi import WebSocket, WebSocketDisconnect

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected")
```

### WebSocket with Connection Manager

```python
from dataclasses import dataclass, field

@dataclass
class ConnectionManager:
    connections: dict[str, WebSocket] = field(default_factory=dict)

    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        self.connections[client_id] = websocket

    def disconnect(self, client_id: str):
        self.connections.pop(client_id, None)

    async def broadcast(self, message: str):
        for ws in self.connections.values():
            await ws.send_text(message)

    async def send_to(self, client_id: str, message: str):
        if ws := self.connections.get(client_id):
            await ws.send_text(message)

manager = ConnectionManager()

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(client_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(f"{client_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(client_id)
```

---

## Database Patterns with SQLAlchemy 2.0

### Async Engine Setup

```python
# app/db/database.py
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

# Example: async database engine (adapt to your DB)
engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/db",  # or your DB URL
    echo=False,
    pool_size=5,
    max_overflow=10,
)

async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def close_db():
    await engine.dispose()
```

### Repository Pattern

```python
# app/db/repositories/base.py
from typing import Generic, TypeVar
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar("T")

class BaseRepository(Generic[T]):
    def __init__(self, session: AsyncSession, model: type[T]):
        self.session = session
        self.model = model

    async def get(self, id: int) -> T | None:
        return await self.session.get(self.model, id)

    async def get_all(self, skip: int = 0, limit: int = 100) -> list[T]:
        result = await self.session.execute(
            select(self.model).offset(skip).limit(limit)
        )
        return list(result.scalars().all())

    async def create(self, obj: T) -> T:
        self.session.add(obj)
        await self.session.commit()
        await self.session.refresh(obj)
        return obj

    async def delete(self, obj: T) -> None:
        await self.session.delete(obj)
        await self.session.commit()

# Usage
class UserRepository(BaseRepository[User]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, User)

    async def get_by_email(self, email: str) -> User | None:
        result = await self.session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()
```

---

## Dependency Injection Patterns

### Cached Dependencies

```python
from functools import lru_cache

@lru_cache
def get_settings() -> Settings:
    return Settings()

# Use in routes
@router.get("/info")
async def get_info(settings: Annotated[Settings, Depends(get_settings)]):
    return {"app_name": settings.app_name}
```

### Class-based Dependencies

```python
class Paginator:
    def __init__(
        self,
        skip: int = Query(0, ge=0),
        limit: int = Query(20, ge=1, le=100),
    ):
        self.skip = skip
        self.limit = limit

@router.get("/items")
async def list_items(pagination: Paginator = Depends()):
    return await ItemService.list(
        skip=pagination.skip,
        limit=pagination.limit,
    )
```

### Dependency with Yield (cleanup)

```python
async def get_db_transaction() -> AsyncGenerator[AsyncSession, None]:
    """Database session with automatic transaction handling."""
    async with async_session() as session:
        async with session.begin():
            yield session
            # Commit happens automatically if no exception
            # Rollback happens automatically on exception

@router.post("/transfer")
async def transfer_money(
    data: TransferRequest,
    db: Annotated[AsyncSession, Depends(get_db_transaction)],
):
    # All operations in this endpoint are in a single transaction
    await debit_account(db, data.from_account, data.amount)
    await credit_account(db, data.to_account, data.amount)
    return {"status": "success"}
```

### Sub-dependencies

```python
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session

async def get_user_repo(
    db: Annotated[AsyncSession, Depends(get_db)]
) -> UserRepository:
    return UserRepository(db)

async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    user_repo: Annotated[UserRepository, Depends(get_user_repo)],
) -> User:
    user_id = decode_token(token)
    user = await user_repo.get(user_id)
    if not user:
        raise HTTPException(401, "Invalid token")
    return user

# Clean route with chained dependencies
@router.get("/me")
async def get_me(user: Annotated[User, Depends(get_current_user)]):
    return user
```

---

## Rate Limiting Pattern

```python
from collections import defaultdict
from datetime import datetime, timedelta
import asyncio

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window = timedelta(seconds=window_seconds)
        self.requests: dict[str, list[datetime]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def is_allowed(self, key: str) -> bool:
        async with self._lock:
            now = datetime.now()
            cutoff = now - self.window

            # Clean old requests
            self.requests[key] = [
                t for t in self.requests[key] if t > cutoff
            ]

            if len(self.requests[key]) >= self.max_requests:
                return False

            self.requests[key].append(now)
            return True

rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

async def check_rate_limit(request: Request):
    client_ip = request.client.host
    if not await rate_limiter.is_allowed(client_ip):
        raise HTTPException(429, "Too many requests")

@router.get("/api/data", dependencies=[Depends(check_rate_limit)])
async def get_data():
    return {"data": "..."}
```
