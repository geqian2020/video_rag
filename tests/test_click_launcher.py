"""No real browser or model is launched by these tests."""
import threading
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.config import Settings
from serve.run import choose_index, main, open_when_ready


def test_default_index_finds_sibling_data(monkeypatch, tmp_path):
    monkeypatch.delenv("RAG_STORAGE__ROOT", raising=False)
    repo = tmp_path / "code"
    repo.mkdir()
    sibling = tmp_path / "video_rag_data" / "index"
    sibling.mkdir(parents=True)
    (sibling / "CURRENT").write_text("a" * 32)
    assert choose_index(Settings(), repo) == sibling


def test_explicit_and_custom_paths_take_precedence(monkeypatch, tmp_path):
    cfg = Settings()
    monkeypatch.setenv("RAG_STORAGE__ROOT", "environment-index")
    cfg.storage.root = "environment-index"
    assert choose_index(cfg, tmp_path, "explicit-index") == Path("explicit-index")
    assert choose_index(cfg, tmp_path) == Path("environment-index")
    monkeypatch.delenv("RAG_STORAGE__ROOT", raising=False)
    cfg.storage.root = "custom-index"
    assert choose_index(cfg, tmp_path) == Path("custom-index")


def test_missing_sibling_keeps_configured_default(monkeypatch, tmp_path):
    monkeypatch.delenv("RAG_STORAGE__ROOT", raising=False)
    assert choose_index(Settings(), tmp_path) == Path("data/index")


def test_browser_opens_only_after_server_started(monkeypatch):
    server = SimpleNamespace(started=False)
    stopped = Mock()
    stopped.is_set.return_value = False
    stopped.wait.side_effect = lambda seconds: setattr(server, "started", True)
    opener = Mock(return_value=True)
    monkeypatch.setattr("serve.run.webbrowser.open", opener)
    open_when_ready(server, "http://127.0.0.1:8000/", stopped)
    stopped.wait.assert_called_once_with(.25)
    opener.assert_called_once_with("http://127.0.0.1:8000/")


def test_failed_startup_does_not_open_browser(monkeypatch):
    stopped = threading.Event()
    stopped.set()
    opener = Mock()
    monkeypatch.setattr("serve.run.webbrowser.open", opener)
    open_when_ready(SimpleNamespace(started=False), "http://127.0.0.1:8000/", stopped)
    opener.assert_not_called()


def test_browser_failure_only_prints_manual_url(monkeypatch, capsys):
    monkeypatch.setattr("serve.run.webbrowser.open", Mock(side_effect=OSError("no browser")))
    open_when_ready(SimpleNamespace(started=True), "http://127.0.0.1:8000/", threading.Event())
    assert "visit http://127.0.0.1:8000/" in capsys.readouterr().out


def test_open_browser_mode_stops_waiter_on_server_exit(monkeypatch, tmp_path):
    (tmp_path / "CURRENT").write_text("a" * 32)
    monkeypatch.setenv("RAG_API_KEY", "test-launcher-key")
    monkeypatch.delenv("RAG_CONFIG", raising=False)
    monkeypatch.setattr("sys.argv", ["run", "--index-root", str(tmp_path), "--open-browser"])
    server = Mock()
    server.started = False
    config = Mock(return_value=object())
    monkeypatch.setattr("uvicorn.Config", config)
    monkeypatch.setattr("uvicorn.Server", Mock(return_value=server))
    worker = Mock()
    thread = Mock(return_value=worker)
    monkeypatch.setattr("serve.run.threading.Thread", thread)
    main()
    config.assert_called_once_with("serve.api:app", host="127.0.0.1", port=8000, workers=1, access_log=False)
    server.run.assert_called_once()
    worker.start.assert_called_once()
    worker.join.assert_called_once_with(timeout=1)
    assert thread.call_args.kwargs["args"][2].is_set()


def test_open_browser_mode_cleans_up_after_interrupt(monkeypatch, tmp_path):
    (tmp_path / "CURRENT").write_text("a" * 32)
    monkeypatch.setenv("RAG_API_KEY", "test-launcher-key")
    monkeypatch.delenv("RAG_CONFIG", raising=False)
    monkeypatch.setattr("sys.argv", ["run", "--index-root", str(tmp_path), "--open-browser"])
    server = Mock()
    server.run.side_effect = KeyboardInterrupt()
    monkeypatch.setattr("uvicorn.Server", Mock(return_value=server))
    worker = Mock()
    monkeypatch.setattr("serve.run.threading.Thread", Mock(return_value=worker))
    with pytest.raises(KeyboardInterrupt):
        main()
    worker.join.assert_called_once_with(timeout=1)


def test_direct_script_works_from_other_cwd_with_safe_path(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    root = tmp_path / "existing index"
    root.mkdir()
    (root / "CURRENT").write_text("a" * 32)
    environment = {**os.environ, "PYTHONSAFEPATH": "1", "RAG_API_KEY": "test-direct-launch-key"}
    for key in ("PYTHONPATH", "RAG_CONFIG", "RAG_STORAGE__ROOT"):
        environment.pop(key, None)
    code = '''
import os, runpy, sys
from types import SimpleNamespace
entry = sys.argv.pop(1)
sys.modules['uvicorn'] = SimpleNamespace(run=lambda *a, **kw: print('TEST_SERVER_CWD=' + os.getcwd()))
runpy.run_path(entry, run_name='__main__')
'''
    result = subprocess.run([sys.executable, "-P", "-c", code, str(repo / "serve" / "run.py"),
                             "--index-root", str(root)], cwd=tmp_path, env=environment,
                            text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "TEST_SERVER_CWD=" + str(repo) in result.stdout
    assert "Index: " + str(root) in result.stdout
