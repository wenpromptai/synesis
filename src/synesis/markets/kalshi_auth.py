"""Kalshi RSA-PSS authentication utilities.

Generates the required headers for Kalshi API/WebSocket authentication:
- KALSHI-ACCESS-KEY
- KALSHI-ACCESS-TIMESTAMP
- KALSHI-ACCESS-SIGNATURE (RSA-PSS SHA256, salt_length=DIGEST_LENGTH)

Signing payload: timestamp_ms + method + path
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from synesis.core.logging import get_logger

logger = get_logger(__name__)


def load_private_key(path: str) -> rsa.RSAPrivateKey:
    """Load an RSA private key from a PEM file.

    Args:
        path: Filesystem path to the PEM key file.

    Returns:
        Loaded RSA private key.

    Raises:
        FileNotFoundError: If the key file doesn't exist.
        ValueError: If the file isn't a valid RSA private key.
    """
    pem_data = Path(path).read_bytes()
    key = serialization.load_pem_private_key(pem_data, password=None)
    if not isinstance(key, rsa.RSAPrivateKey):
        raise ValueError(f"Expected RSA private key, got {type(key).__name__}")
    return key


def make_kalshi_headers(
    api_key: str,
    private_key: rsa.RSAPrivateKey,
    method: str,
    path: str,
) -> dict[str, str]:
    """Generate Kalshi authentication headers.

    Args:
        api_key: Kalshi API key ID.
        private_key: Loaded RSA private key.
        method: HTTP method (GET, POST) — uppercased in signature.
        path: Request path (e.g. /trade-api/ws/v2).

    Returns:
        Dict with KALSHI-ACCESS-KEY, KALSHI-ACCESS-TIMESTAMP, KALSHI-ACCESS-SIGNATURE.
    """
    timestamp_ms = str(int(time.time() * 1000))
    payload = (timestamp_ms + method.upper() + path).encode()

    signature = private_key.sign(
        payload,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=hashes.SHA256().digest_size,
        ),
        hashes.SHA256(),
    )

    return {
        "KALSHI-ACCESS-KEY": api_key,
        "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
    }
