---
name: fastapi-developing
description: FastAPI 0.115+ async API developing with Pydantic v2 validation, dependency injection, lifespan managers, and async database patterns. Use when creating endpoints, models, services, middleware, or debugging backend errors. Apply for .py files in api/, routes/, services/, or main.py. (project)
---

# FastAPI Backend Development

Modern patterns for FastAPI with Pydantic v2 and Python 3.13+ (December 2025 best practices).

**Package Manager:** Use `uv` for all Python projects (`uv init`, `uv add`, `uv run`).

## When To Apply
- Creating or editing API endpoints
- Building Pydantic models/schemas
- Setting up middleware or dependencies
- Debugging backend errors
- Working with async database operations

## Quick Reference

| Pattern | ✅ Current (Dec 2025) | ❌ Deprecated |
|---------|----------------------|---------------|
| Startup/shutdown | `lifespan` context manager | `@app.on_event()` |
| Pydantic config | `model_config = ConfigDict(...)` | `class Config:` |
| Parse dict → model | `Model.model_validate(data)` | `Model.parse_obj(data)` |
| Model → dict | `model.model_dump()` | `model.dict()` |
| JSON schema | `Model.model_json_schema()` | `Model.schema()` |
| Type hints | `str \| None` | `Optional[str]` |
| Dependencies | `Annotated[T, Depends(fn)]` | `param: T = Depends(fn)` |
| Validators | `@field_validator` + `@classmethod` | `@validator` |
| Serializers | `@field_serializer` | `@validator(pre=False)` |
| DB driver | async driver (project-specific) | sync driver |
| Python version | 3.13+ (use `type` statement) | 3.11 and below |
| Package manager | `uv` (uv.lock) | pip, poetry |

---

## Project Structure

```
app/
├── main.py              # FastAPI app, lifespan, include routers
├── config.py            # Settings with pydantic-settings
├── dependencies.py      # Shared dependencies (auth, db session)
├── models/              # Pydantic models (schemas)
│   ├── __init__.py
│   ├── user.py
│   └── item.py
├── routers/             # APIRouter modules
│   ├── __init__.py
│   ├── users.py
│   └── items.py
├── services/            # Business logic
│   └── user_service.py
└── db/                  # Database (if applicable)
    ├── database.py
    └── repositories/
```

---

## App Setup with Lifespan

```python
# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.routers import users, items
from app.db.database import init_db, close_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: runs before app accepts requests
    await init_db()
    yield
    # Shutdown: runs when app is stopping
    await close_db()

app = FastAPI(
    title="My API",
    version="1.0.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(users.router)
app.include_router(items.router)
```

---

## Pydantic v2 Models

```python
# app/models/user.py
from pydantic import BaseModel, ConfigDict, Field, EmailStr
from datetime import datetime

class UserBase(BaseModel):
    """Shared fields for user models."""
    email: EmailStr
    name: str = Field(min_length=1, max_length=100)

class UserCreate(UserBase):
    """Fields required to create a user."""
    password: str = Field(min_length=8)

class UserResponse(UserBase):
    """Fields returned to client (no password)."""
    id: int
    created_at: datetime
    
    # Pydantic v2 config (replaces class Config)
    model_config = ConfigDict(
        from_attributes=True,  # Was: orm_mode = True
        json_schema_extra={
            "examples": [
                {
                    "id": 1,
                    "email": "user@example.com",
                    "name": "John Doe",
                    "created_at": "2024-01-15T10:30:00Z"
                }
            ]
        }
    )

class UserUpdate(BaseModel):
    """Optional fields for partial update."""
    email: EmailStr | None = None
    name: str | None = Field(default=None, min_length=1, max_length=100)
    
    model_config = ConfigDict(extra="forbid")  # Reject unknown fields
```

---

## Routers with Typed Dependencies

```python
# app/routers/users.py
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status

from app.models.user import UserCreate, UserResponse, UserUpdate
from app.dependencies import get_current_user, get_db
from app.services.user_service import UserService

router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)

# Reusable dependency type aliases
CurrentUser = Annotated[UserResponse, Depends(get_current_user)]
DB = Annotated[AsyncSession, Depends(get_db)]

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(user: CurrentUser):
    """Get current authenticated user."""
    return user

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: int, db: DB):
    """Get user by ID."""
    user = await UserService.get_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )
    return user

@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user_data: UserCreate, db: DB):
    """Create a new user."""
    return await UserService.create(db, user_data)

@router.patch("/{user_id}", response_model=UserResponse)
async def update_user(user_id: int, user_data: UserUpdate, db: DB, user: CurrentUser):
    """Update user (partial)."""
    # Use model_dump(exclude_unset=True) to get only provided fields
    update_data = user_data.model_dump(exclude_unset=True)
    return await UserService.update(db, user_id, update_data)
```

---

## Dependencies

```python
# app/dependencies.py
from typing import Annotated, AsyncGenerator
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.db.database import async_session
from app.services.auth_service import verify_token

security = HTTPBearer()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield database session with automatic cleanup."""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()

async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Validate token and return current user."""
    token = credentials.credentials
    user = await verify_token(db, token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
```

---

## Settings with pydantic-settings

```python
# app/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    app_name: str = "My API"
    debug: bool = False
    
    # Database
    database_url: str
    
    # Auth
    secret_key: str
    access_token_expire_minutes: int = 30
    
    # Load from .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra env vars
    )

@lru_cache
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
```

---

## Error Handling

```python
# app/main.py (add to existing)
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )

# Custom exception
class NotFoundError(Exception):
    def __init__(self, resource: str, id: int | str):
        self.resource = resource
        self.id = id

@app.exception_handler(NotFoundError)
async def not_found_handler(request: Request, exc: NotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": f"{exc.resource} with id {exc.id} not found"},
    )
```

---

## Query Parameters with Models

```python
from typing import Literal
from fastapi import Query
from pydantic import BaseModel, Field

class PaginationParams(BaseModel):
    """Reusable pagination parameters."""
    skip: int = Field(0, ge=0)
    limit: int = Field(20, ge=1, le=100)
    order_by: Literal["created_at", "updated_at", "name"] = "created_at"
    order: Literal["asc", "desc"] = "desc"
    
    model_config = ConfigDict(extra="forbid")

@router.get("/")
async def list_items(
    pagination: PaginationParams = Query(),
    search: str | None = None,
):
    """List items with pagination and optional search."""
    return await ItemService.list(
        skip=pagination.skip,
        limit=pagination.limit,
        order_by=pagination.order_by,
        order=pagination.order,
        search=search,
    )
```

---

## Async Best Practices

```python
# ✅ DO: Use async for I/O operations
@router.get("/items")
async def get_items(db: DB):
    return await db.execute(select(Item))

# ✅ DO: Use sync def for CPU-bound operations
# FastAPI runs these in a thread pool automatically
@router.get("/compute")
def compute_heavy():
    return {"result": some_cpu_intensive_calculation()}

# ❌ DON'T: Block the event loop
@router.get("/bad")
async def bad_endpoint():
    time.sleep(5)  # Blocks entire server!
    
# ✅ DO: Use asyncio.sleep or run_in_executor
import asyncio
@router.get("/good")
async def good_endpoint():
    await asyncio.sleep(5)  # Non-blocking
```

---

## Testing

```python
# tests/test_users.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_create_user(client):
    response = client.post(
        "/users/",
        json={
            "email": "test@example.com",
            "name": "Test User",
            "password": "securepassword123"
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test@example.com"
    assert "password" not in data  # Should not be in response

def test_get_user_not_found(client):
    response = client.get("/users/99999")
    assert response.status_code == 404
```

---

## Key Reminders

1. **Always use `lifespan`** for startup/shutdown, not `@app.on_event()`
2. **Use `Annotated[T, Depends()]`** for cleaner dependency injection
3. **Use `model_config = ConfigDict(...)`** not `class Config:`
4. **Use `model_dump()` and `model_validate()`** not `dict()` and `parse_obj()`
5. **Use `str | None`** not `Optional[str]` (Python 3.10+)
6. **Use `status.HTTP_*` constants** for status codes
7. **Separate schemas**: Create, Response, Update models should differ
8. **Use `exclude_unset=True`** in `model_dump()` for PATCH endpoints

---

## Resource Files

For detailed information, see these resource files:

- [patterns.md](resources/patterns.md) - Advanced patterns: middleware, background tasks, WebSockets, streaming
- [anti-patterns.md](resources/anti-patterns.md) - Common mistakes to avoid, deprecated patterns, async pitfalls
