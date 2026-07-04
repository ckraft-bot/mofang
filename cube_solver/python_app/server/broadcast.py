from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket


class EventBroadcaster:
    """Broadcasts JSON messages to any connected dashboard clients."""

    def __init__(self) -> None:
        self.connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.connections:
            self.connections.remove(websocket)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        message = json.dumps(payload)
        dead: list[WebSocket] = []
        for websocket in list(self.connections):
            try:
                await websocket.send_text(message)
            except Exception:
                dead.append(websocket)
        for websocket in dead:
            self.disconnect(websocket)

    def publish(self, payload: dict[str, Any]) -> None:
        try:
            asyncio.run(self.broadcast(payload))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.broadcast(payload))
            finally:
                loop.close()
