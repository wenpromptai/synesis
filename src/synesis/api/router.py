"""Top-level API router — mounts all domain routers under /api/v1."""

from fastapi import APIRouter

from synesis.api.routes import earnings, fh_prices, sec_edgar, system, watchlist

api_router = APIRouter()
api_router.include_router(fh_prices.router, prefix="/fh_prices", tags=["fh_prices"])
api_router.include_router(watchlist.router, prefix="/watchlist", tags=["watchlist"])
api_router.include_router(system.router, prefix="/system", tags=["system"])
api_router.include_router(sec_edgar.router, prefix="/sec_edgar", tags=["sec_edgar"])
api_router.include_router(earnings.router, prefix="/earnings", tags=["earnings"])
