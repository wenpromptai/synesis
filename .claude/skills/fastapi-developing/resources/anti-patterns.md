# FastAPI Anti-Patterns to Avoid (December 2025)

Common mistakes and outdated patterns that should be avoided in modern FastAPI development.

## Table of Contents
- [Deprecated Patterns](#deprecated-patterns)
- [Async Mistakes](#async-mistakes)
- [Pydantic v2 Migration Issues](#pydantic-v2-migration-issues)
- [Architecture Anti-Patterns](#architecture-anti-patterns)
- [Python 3.12+ Considerations](#python-312-considerations)

---

## Deprecated Patterns

### ❌ Using @app.on_event() for lifecycle

**Don't do this (deprecated in FastAPI 0.103+):**
```python
@app.on_event("startup")
async def startup():
    await init_db()

@app.on_event("shutdown")  
async def shutdown():
    await close_db()
```

**Do this instead:**
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()

app = FastAPI(lifespan=lifespan)
```

### ❌ Using old dependency injection syntax

**Don't do this:**
```python
@router.get("/users/{user_id}")
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    ...
```

**Do this instead:**
```python
from typing import Annotated

DB = Annotated[AsyncSession, Depends(get_db)]

@router.get("/users/{user_id}")
async def get_user(user_id: int, db: DB):
    ...
```

---

## Async Mistakes

### ❌ Blocking the event loop

**This blocks the entire server:**
```python
import time

@router.get("/slow")
async def slow_endpoint():
    time.sleep(5)  # BLOCKS EVERYTHING!
    return {"status": "done"}
```

**Do this instead:**
```python
import asyncio

@router.get("/slow")
async def slow_endpoint():
    await asyncio.sleep(5)  # Non-blocking
    return {"status": "done"}
```

### ❌ Using sync database drivers in async routes

**This defeats the purpose of async:**
```python
@router.get("/users")
async def get_users():
    # psycopg2 is synchronous!
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    return cursor.fetchall()
```

**Do this instead:**
```python
from sqlalchemy.ext.asyncio import AsyncSession

@router.get("/users")
async def get_users(db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(select(User))
    return result.scalars().all()
```

### ❌ Making sync routes when you need async

**If you have async operations, the route must be async:**
```python
# ❌ This won't work properly
@router.get("/items")
def get_items(db: AsyncSession):
    return db.execute(select(Item))  # Can't await in sync function!
```

---

## Pydantic v2 Migration Issues

### ❌ Using Pydantic v1 syntax

| v1 (Don't use) | v2 (Use this) |
|----------------|---------------|
| `class Config:` | `model_config = ConfigDict(...)` |
| `orm_mode = True` | `from_attributes = True` |
| `.dict()` | `.model_dump()` |
| `.parse_obj()` | `.model_validate()` |
| `.schema()` | `.model_json_schema()` |
| `Optional[str]` | `str \| None` |

### ❌ Using @validator instead of @field_validator

**Don't do this:**
```python
from pydantic import validator

class User(BaseModel):
    name: str
    
    @validator("name")
    def name_must_be_capitalized(cls, v):
        return v.capitalize()
```

**Do this instead:**
```python
from pydantic import field_validator

class User(BaseModel):
    name: str
    
    @field_validator("name")
    @classmethod
    def name_must_be_capitalized(cls, v: str) -> str:
        return v.capitalize()
```

---

## Architecture Anti-Patterns

### ❌ Business logic in routes

**Don't do this:**
```python
@router.post("/users/")
async def create_user(user_data: UserCreate, db: DB):
    # Business logic directly in route
    existing = await db.execute(select(User).where(User.email == user_data.email))
    if existing.scalar():
        raise HTTPException(400, "Email already exists")
    
    hashed_password = hash_password(user_data.password)
    new_user = User(
        email=user_data.email,
        name=user_data.name,
        hashed_password=hashed_password
    )
    db.add(new_user)
    await db.commit()
    return new_user
```

**Do this instead:**
```python
# services/user_service.py
class UserService:
    @staticmethod
    async def create(db: AsyncSession, user_data: UserCreate) -> User:
        existing = await db.execute(select(User).where(User.email == user_data.email))
        if existing.scalar():
            raise UserAlreadyExistsError(user_data.email)
        
        hashed_password = hash_password(user_data.password)
        new_user = User(
            email=user_data.email,
            name=user_data.name,
            hashed_password=hashed_password
        )
        db.add(new_user)
        await db.commit()
        return new_user

# routers/users.py
@router.post("/users/")
async def create_user(user_data: UserCreate, db: DB):
    return await UserService.create(db, user_data)
```

### ❌ Hardcoding configuration

**Don't do this:**
```python
DATABASE_URL = "postgresql://user:pass@localhost/db"
SECRET_KEY = "my-secret-key"
```

**Do this instead:**
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    secret_key: str
    
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
```

### ❌ Using one model for everything

**Don't do this:**
```python
class User(BaseModel):
    id: int
    email: str
    password: str
    created_at: datetime
```

**Do this instead (separate schemas):**
```python
class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    created_at: datetime
    # Note: no password field

class UserUpdate(BaseModel):
    email: EmailStr | None = None
```

---

## Python 3.12+ Considerations

### ❌ Using Optional instead of union syntax

**Don't do this (Python 3.9 style):**
```python
from typing import Optional, List, Dict

def get_user(user_id: int) -> Optional[User]:
    ...

def get_items() -> List[Dict[str, Any]]:
    ...
```

**Do this instead (Python 3.10+):**
```python
def get_user(user_id: int) -> User | None:
    ...

def get_items() -> list[dict[str, Any]]:
    ...
```

### ❌ Not using the type statement (Python 3.12+)

**Old style:**
```python
from typing import TypeAlias, Annotated
from fastapi import Depends

DB: TypeAlias = Annotated[AsyncSession, Depends(get_db)]
```

**Modern style (Python 3.12+):**
```python
type DB = Annotated[AsyncSession, Depends(get_db)]
type UserOrNone = User | None
```

### ❌ Using root_validator instead of model_validator

**Don't do this:**
```python
from pydantic import root_validator

class DateRange(BaseModel):
    start: date
    end: date

    @root_validator
    def check_dates(cls, values):
        if values.get("end") < values.get("start"):
            raise ValueError("end must be after start")
        return values
```

**Do this instead:**
```python
from pydantic import model_validator

class DateRange(BaseModel):
    start: date
    end: date

    @model_validator(mode="after")
    def check_dates(self) -> "DateRange":
        if self.end < self.start:
            raise ValueError("end must be after start")
        return self
```

### ❌ Using sync HTTP clients in async routes

**Don't do this:**
```python
import requests

@router.get("/external")
async def get_external_data():
    response = requests.get("https://api.example.com/data")  # Blocks!
    return response.json()
```

**Do this instead:**
```python
import httpx

@router.get("/external")
async def get_external_data():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
        return response.json()
```

### ❌ Not using pydantic-settings for configuration

**Don't do this:**
```python
import os

DATABASE_URL = os.getenv("DATABASE_URL")
SECRET_KEY = os.getenv("SECRET_KEY", "default-secret")
```

**Do this instead:**
```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str
    secret_key: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )
```
