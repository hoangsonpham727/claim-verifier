#!/usr/bin/env python3
"""
HTTPS static server for the Word add-in task pane.

Mac Word's webview rejects plain http://localhost, so the add-in files must
be served over HTTPS using the trusted Office dev certificates created by:

    npx office-addin-dev-certs install

Run:
    python3 addin/serve_https.py          # serves ./addin on https://localhost:3001
"""
from __future__ import annotations

import http.server
import os
import ssl
from pathlib import Path

PORT = 3002
ADDIN_DIR = Path(__file__).resolve().parent
CERT_DIR = Path.home() / ".office-addin-dev-certs"
CERT = CERT_DIR / "localhost.crt"
KEY = CERT_DIR / "localhost.key"


def main() -> None:
    if not CERT.exists() or not KEY.exists():
        raise SystemExit(
            f"Dev certs not found in {CERT_DIR}.\n"
            "Run: npx office-addin-dev-certs install"
        )

    os.chdir(ADDIN_DIR)
    handler = http.server.SimpleHTTPRequestHandler
    httpd = http.server.HTTPServer(("localhost", PORT), handler)

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(CERT), keyfile=str(KEY))
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

    print(f"Serving {ADDIN_DIR} at https://localhost:{PORT}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
