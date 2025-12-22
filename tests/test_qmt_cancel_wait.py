import pytest

from bullet_trade.server.adapters.qmt import QmtBrokerAdapter
from bullet_trade.server.adapters.base import AccountRouter
from bullet_trade.server.config import AccountConfig, ServerConfig


class _FakeBroker:
    def __init__(self, snapshots):
        self._snapshots = list(snapshots)

    async def cancel_order(self, order_id):
        return True

    async def get_order_status(self, order_id):
        if self._snapshots:
            return self._snapshots.pop(0)
        return {}


@pytest.mark.asyncio
async def test_qmt_cancel_wait_final_status(monkeypatch):
    monkeypatch.setenv("TRADE_MAX_WAIT_TIME", "1")
    config = ServerConfig(
        server_type="qmt",
        listen="127.0.0.1",
        port=0,
        token="t",
        enable_data=False,
        enable_broker=True,
        accounts=[AccountConfig(key="default", account_id="demo")],
    )
    router = AccountRouter(config.accounts)
    adapter = QmtBrokerAdapter(config, router)
    ctx = router.get("default")
    adapter._brokers[ctx.config.key] = _FakeBroker(
        [
            {"order_id": "1", "status": "canceling", "raw_status": 1},
            {"order_id": "1", "status": "canceled", "raw_status": 0},
        ]
    )

    resp = await adapter.cancel_order(ctx, "1")

    assert resp["value"] is True
    assert resp["timed_out"] is False
    assert resp["status"] == "canceled"


@pytest.mark.asyncio
async def test_qmt_cancel_wait_timeout(monkeypatch):
    monkeypatch.setenv("TRADE_MAX_WAIT_TIME", "0.6")
    config = ServerConfig(
        server_type="qmt",
        listen="127.0.0.1",
        port=0,
        token="t",
        enable_data=False,
        enable_broker=True,
        accounts=[AccountConfig(key="default", account_id="demo")],
    )
    router = AccountRouter(config.accounts)
    adapter = QmtBrokerAdapter(config, router)
    ctx = router.get("default")
    adapter._brokers[ctx.config.key] = _FakeBroker(
        [
            {"order_id": "2", "status": "open", "raw_status": 50},
        ]
    )

    resp = await adapter.cancel_order(ctx, "2")

    assert resp["value"] is True
    assert resp["timed_out"] is True
    assert resp["status"] == "open"
