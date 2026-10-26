"""Local HTML page, unchanged search contract and loopback launcher."""
import os
from pathlib import Path

from fastapi.testclient import TestClient

from serve.api import create_app


class Backend:
    def search(self, query, top_k, video_ids):
        if query == "empty":
            return []
        return [{"chunk_id": "demo", "video_id": "sharepoint/demo", "start_ts": 65.5,
                 "end_ts": 95., "text": '<img src=x onerror="alert(1)"> 原文证据',
                 "summary": "原文证据", "timestamp_quality": "asr",
                 "enrichment_status": "extractive_fallback", "rrf_score": .03, "rerank_score": 2.1}]

    def close(self):
        pass


def test_page_is_self_contained_and_never_embeds_secret(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "private-server-key")
    with TestClient(create_app(Backend)) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert response.headers["cache-control"] == "no-store"
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
        assert "private-server-key" not in response.text
        assert 'id="query"' in response.text
        assert 'id="api-key" type="password"' in response.text
        assert "fetch('/search'" in response.text
        assert "X-API-Key" in response.text
        assert "localStorage" not in response.text
        assert "sessionStorage" not in response.text
        assert "innerHTML" not in response.text
        assert "<script src=" not in response.text


def test_page_does_not_bypass_search_authentication(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "test-key")
    with TestClient(create_app(Backend)) as client:
        assert client.get("/").status_code == 200
        assert client.post("/search", json={"query": "PA calibration"}).status_code == 401
        response = client.post("/search", headers={"X-API-Key": "test-key"},
                               json={"query": "PA calibration", "top_k": 5, "video_ids": ["sharepoint/demo"]})
        assert response.status_code == 200
        result = response.json()
        assert result["results"][0]["start_ts"] == 65.5
        assert result["results"][0]["text"].startswith("<img")
        assert result["context_is_untrusted"] is True


def test_empty_matches_remain_valid_response(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "test-key")
    with TestClient(create_app(Backend)) as client:
        response = client.post("/search", headers={"X-API-Key": "test-key"}, json={"query": "empty"})
        assert response.json()["results"] == []


def test_launcher_uses_given_index_and_loopback(monkeypatch, tmp_path, capsys):
    from serve.run import main
    (tmp_path / "CURRENT").write_text("a" * 32, encoding="utf-8")
    monkeypatch.setenv("RAG_API_KEY", "existing-key")
    monkeypatch.delenv("RAG_CONFIG", raising=False)
    monkeypatch.setattr("sys.argv", ["run", "--index-root", str(tmp_path), "--port", "8765"])
    calls = []
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: calls.append((args, kwargs)))
    main()
    assert os.environ["RAG_STORAGE__ROOT"] == str(tmp_path)
    assert calls[0][1] == {"host": "127.0.0.1", "port": 8765, "workers": 1, "access_log": False}
    assert "existing-key" in capsys.readouterr().out


def test_launcher_creates_temporary_key_if_missing(monkeypatch, tmp_path, capsys):
    from serve.run import main
    (tmp_path / "CURRENT").write_text("a" * 32, encoding="utf-8")
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    monkeypatch.delenv("RAG_CONFIG", raising=False)
    monkeypatch.setattr("sys.argv", ["run", "--index-root", str(tmp_path)])
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)
    main()
    assert len(os.environ["RAG_API_KEY"]) >= 24
    assert "Access key" in capsys.readouterr().out


def test_launcher_rejects_missing_index_without_starting(monkeypatch, tmp_path):
    from serve.run import main
    import pytest
    monkeypatch.setattr("sys.argv", ["run", "--index-root", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
