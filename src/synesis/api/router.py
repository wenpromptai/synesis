"""Top-level API router — mounts all domain routers under /api/v1."""

from fastapi import APIRouter

from synesis.api.routes import factset, mkt_intel, system, watchlist

api_router = APIRouter()
api_router.include_router(factset.router, prefix="/factset", tags=["factset"])
api_router.include_router(watchlist.router, prefix="/watchlist", tags=["watchlist"])
api_router.include_router(system.router, prefix="/system", tags=["system"])
api_router.include_router(mkt_intel.router, prefix="/mkt_intel", tags=["mkt_intel"])
