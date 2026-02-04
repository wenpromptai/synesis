"""CLI entry point for Synesis."""

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Synesis")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    args = parser.parse_args()

    uvicorn.run(
        "synesis.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
