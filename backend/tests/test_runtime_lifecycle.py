import asyncio

from fastapi.testclient import TestClient

from app.infrastructure.db import Database
from app.main import create_app


def test_wal_reader_can_read_while_another_connection_writes(tmp_path):
    db = Database(str(tmp_path / "concurrent.db"))
    with db.connection() as writer, db.connection() as reader:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("UPDATE tariffs SET name = 'pending' WHERE tariff_id = 'T-DEFAULT'")
        assert reader.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert reader.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert reader.execute("SELECT name FROM tariffs WHERE tariff_id = 'T-DEFAULT'").fetchone()[0] != "pending"
    assert db.get_tariff("T-DEFAULT").name == "pending"


def test_inspection_task_stops_with_application(tmp_path):
    app = create_app(str(tmp_path / "lifecycle.db"))
    with TestClient(app):
        task = app.state.inspection_task
        assert not task.done()
    assert task.cancelled()


def test_inspection_failure_is_logged_and_next_cycle_retries(monkeypatch, caplog):
    import app.main as main
    from types import SimpleNamespace

    calls = 0

    class FailingDatabase:
        def list_sites(self):
            nonlocal calls
            calls += 1
            raise RuntimeError("inspection database failure")

    async def tick(_):
        if calls == 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(main.asyncio, "sleep", tick)
    try:
        asyncio.run(main._inspection_loop(SimpleNamespace(state=SimpleNamespace(db=FailingDatabase()))))
    except asyncio.CancelledError:
        pass
    assert calls == 2
    records = [r for r in caplog.records if r.message == "Background inspection failed"]
    assert len(records) == 2
    assert all(r.exc_info for r in records)


def test_unhandled_error_logs_request_id_without_exposing_exception(tmp_path, caplog):
    app = create_app(str(tmp_path / "errors.db"))

    @app.get("/broken")
    def broken():
        raise RuntimeError("internal diagnostic detail")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/broken")
    assert response.status_code == 500
    assert "internal diagnostic detail" not in response.text
    request_id = response.json()["error"]["request_id"]
    records = [r for r in caplog.records if request_id in r.message]
    assert len(records) == 1
    assert records[0].exc_info
