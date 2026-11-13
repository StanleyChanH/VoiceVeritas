"""Container entrypoint: serve the model at $VV_MODEL (default /models/model.joblib)."""

from __future__ import annotations

import os

import uvicorn

from voiceveritas.api import create_app


def main() -> None:
    model_path = os.environ.get("VV_MODEL", "/models/model.joblib")
    host = os.environ.get("VV_HOST", "0.0.0.0")
    port = int(os.environ.get("VV_PORT", "8000"))
    app = create_app(model_path)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
