"""Top-level API router — mounts all domain routers under /api/v1."""

from fastapi import APIRouter

from synesis.api.routes import earnings, fh, fred, sec_edgar, system, watchlist, yf

api_router = APIRouter()
api_router.include_router(fh.router, prefix="/fh", tags=["fh"])
api_router.include_router(watchlist.router, prefix="/watchlist", tags=["watchlist"])
api_router.include_router(system.router, prefix="/system", tags=["system"])
api_router.include_router(sec_edgar.router, prefix="/sec_edgar", tags=["sec_edgar"])
api_router.include_router(earnings.router, prefix="/earnings", tags=["earnings"])
api_router.include_router(yf.router, prefix="/yf", tags=["yfinance"])
api_router.include_router(fred.router, prefix="/fred", tags=["fred"])
