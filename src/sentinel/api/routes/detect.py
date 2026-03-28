"""Detection routes: batch anomaly detection and WebSocket streaming."""

from __future__ import annotations

import asyncio
import json as json_mod
from typing import Any

import numpy as np
import structlog
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from sentinel.api.deps import resolve_safe_path
from sentinel.api.schemas import DetectRequest, DetectResponse, ErrorResponse
from sentinel.core.registry import get_model_class, list_models
from sentinel.data.loaders import load_file
from sentinel.data.validators import get_feature_columns, validate_dataframe

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/detect", tags=["detection"])


@router.post(
    "",
    response_model=DetectResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def batch_detect(body: DetectRequest) -> DetectResponse:
    """Run batch anomaly detection on a dataset.

    Loads the saved model from ``model_path``, loads data from
    ``data_path``, scores the data, and returns scores with labels.

    Args:
        body: Request with data_path and model_path.

    Returns:
        Detection results with scores, labels, and threshold.
    """
    # Sanitize paths — reject traversal outside allowed directories.
    data_path = resolve_safe_path(body.data_path)
    model_path = resolve_safe_path(body.model_path)

    # Validate data path exists.
    if not data_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Data file not found: {body.data_path}",
        )

    # Validate model path exists.
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model path not found: {body.model_path}",
        )

    # Load data.
    try:
        df = load_file(data_path)
        df = validate_dataframe(df)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Data validation failed: {exc}"
        ) from exc

    # Separate features.
    feature_cols = get_feature_columns(df)
    X = df.select(feature_cols).to_numpy().astype(np.float64)

    # Determine the model name from the saved config.
    import json

    config_file = model_path / "config.json"
    meta_file = model_path / "meta.json"

    model_name = ""
    if config_file.exists():
        config_data = json.loads(config_file.read_text())
        model_name = config_data.get("model", config_data.get("model_name", ""))
    elif meta_file.exists():
        meta_data = json.loads(meta_file.read_text())
        model_name = meta_data.get("model_name", "")

    if not model_name:
        raise HTTPException(
            status_code=400,
            detail="Cannot determine model name from saved artifacts. "
            "Expected config.json or meta.json with a 'model' or "
            "'model_name' field.",
        )

    # Verify model is registered.
    registry = list_models()
    if model_name not in registry:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' is not registered. "
            f"Available: {', '.join(sorted(registry.keys()))}",
        )

    # Instantiate and load the model.
    try:
        model_cls = get_model_class(model_name)
        model = model_cls()
        model.load(str(model_path))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {exc}",
        ) from exc

    # Score the data.
    try:
        scores = model.score(X)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Scoring failed: {exc}",
        ) from exc

    # Apply a default threshold (95th percentile) if not stored.
    threshold = float(np.percentile(scores, 95))
    labels = (scores > threshold).astype(np.int32)

    logger.info(
        "detect.batch_complete",
        model_name=model_name,
        n_samples=len(scores),
        n_anomalies=int(labels.sum()),
    )

    return DetectResponse(
        scores=scores.tolist(),
        labels=labels.tolist(),
        threshold=threshold,
        model_name=model_name,
    )


@router.websocket("/stream")
async def detect_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time streaming anomaly detection.

    Protocol::

        Client -> Server: {"type": "init", "model_path": "data/experiments/run-1"}
        Client -> Server: {"type": "data", "timestamp": "...",
                          "features": {"cpu": 45.2}}
        Server -> Client: {"type": "score", "timestamp": "...", "score": 0.87,
                           "label": 1, "threshold": 0.75}
        Server -> Client: {"type": "error", "message": "...", "code": "..."}
        Server -> Client: {"type": "ping"}
        Client -> Server: {"type": "pong"}
    """
    await websocket.accept()

    model: Any = None
    threshold: float = 0.0
    buffer: list[list[float]] = []
    seq_len: int = 50
    feature_names: list[str] = []

    async def _send_error(msg: str, code: str = "ERROR") -> None:
        await websocket.send_json({"type": "error", "message": msg, "code": code})

    # Heartbeat task: ping every 30s.
    async def _heartbeat() -> None:
        try:
            while True:
                await asyncio.sleep(30)
                await websocket.send_json({"type": "ping"})
        except (WebSocketDisconnect, Exception):
            pass

    heartbeat_task = asyncio.create_task(_heartbeat())

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json_mod.loads(raw)
            except json_mod.JSONDecodeError:
                await _send_error("Invalid JSON", "PARSE_ERROR")
                continue

            msg_type = msg.get("type", "")

            if msg_type == "pong":
                continue

            if msg_type == "init":
                # Load the model.
                model_path_str = msg.get("model_path", "")
                if not model_path_str:
                    await _send_error("model_path required", "MISSING_PARAM")
                    continue

                try:
                    model_path = resolve_safe_path(model_path_str)
                except Exception:
                    await _send_error(
                        f"Invalid model path: {model_path_str}", "INVALID_PATH"
                    )
                    continue

                if not model_path.exists():
                    await _send_error(
                        f"Model not found: {model_path_str}", "MODEL_NOT_FOUND"
                    )
                    continue

                # Determine model name from config.
                config_file = model_path / "config.json"
                meta_file = model_path / "meta.json"
                model_name = ""
                if config_file.exists():
                    cfg = json_mod.loads(config_file.read_text())
                    model_name = cfg.get("model", cfg.get("model_name", ""))
                    seq_len = int(cfg.get("seq_len", 50))
                elif meta_file.exists():
                    meta = json_mod.loads(meta_file.read_text())
                    model_name = meta.get("model_name", "")

                if not model_name:
                    await _send_error(
                        "Cannot determine model name", "MODEL_NAME_MISSING"
                    )
                    continue

                registry = list_models()
                if model_name not in registry:
                    await _send_error(
                        f"Model '{model_name}' not registered", "MODEL_NOT_FOUND"
                    )
                    continue

                try:
                    model_cls = get_model_class(model_name)
                    model = model_cls()
                    model.load(str(model_path))
                except Exception as exc:
                    await _send_error(f"Failed to load model: {exc}", "LOAD_ERROR")
                    model = None
                    continue

                buffer = []
                await websocket.send_json(
                    {"type": "ready", "model": model_name, "seq_len": seq_len}
                )

            elif msg_type == "data":
                if model is None:
                    await _send_error(
                        "Model not loaded. Send init first.", "MODEL_NOT_LOADED"
                    )
                    continue

                timestamp = msg.get("timestamp", "")
                features_dict = msg.get("features", {})

                if not features_dict:
                    await _send_error("No features provided", "MISSING_FEATURES")
                    continue

                # On first data point, lock feature names.
                if not feature_names:
                    feature_names = sorted(features_dict.keys())

                row = [float(features_dict.get(f, 0.0)) for f in feature_names]
                buffer.append(row)

                # Keep only the last seq_len points.
                if len(buffer) > seq_len:
                    buffer = buffer[-seq_len:]

                # Score when we have enough data.
                if len(buffer) >= seq_len:
                    try:
                        window = np.array(buffer, dtype=np.float64)
                        scores = model.score(window)
                        score_val = float(scores[-1])

                        # Use 95th percentile of current window as threshold.
                        if threshold == 0.0:
                            threshold = float(np.percentile(scores, 95))

                        label = 1 if score_val > threshold else 0

                        await websocket.send_json({
                            "type": "score",
                            "timestamp": timestamp,
                            "score": round(score_val, 6),
                            "label": label,
                            "threshold": round(threshold, 6),
                        })
                    except Exception as exc:
                        await _send_error(f"Scoring failed: {exc}", "SCORE_ERROR")
                else:
                    # Not enough data yet — acknowledge receipt.
                    await websocket.send_json({
                        "type": "buffering",
                        "buffered": len(buffer),
                        "needed": seq_len,
                    })
            else:
                await _send_error(
                    f"Unknown message type: {msg_type}", "UNKNOWN_TYPE"
                )

    except WebSocketDisconnect:
        logger.info("detect.stream_disconnected")
    except Exception as exc:
        logger.error("detect.stream_error", error=str(exc))
    finally:
        heartbeat_task.cancel()
