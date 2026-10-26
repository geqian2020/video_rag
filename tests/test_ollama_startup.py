"""Ollama lifecycle tests never start Ollama or download a model."""
import errno
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx

from core.config import Settings
from core.io import write_rows
from core.ollama import _launch, ensure_ollama
from ingest.enrich import main as enrich_main


def refused(request):
    try:
        raise ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused")
    except ConnectionRefusedError as cause:
        raise httpx.ConnectError("refused", request=request) from cause


class OllamaStartup(unittest.TestCase):
    def setUp(self):
        self.cfg = Settings()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.log = Path(self.temp.name) / "logs" / "ollama.log"
        self.messages = []

    def client(self, handler):
        client = httpx.Client(transport=httpx.MockTransport(handler), trust_env=False,
                             follow_redirects=False)
        self.addCleanup(client.close)
        return client

    def run_ensure(self, client, **kwargs):
        return ensure_ollama(client, self.cfg, self.log, report=self.messages.append, **kwargs)

    def test_existing_server_reused_without_launch(self):
        client = self.client(lambda request: httpx.Response(200, json={"version": "test"}))
        with patch("core.ollama._launch") as launch:
            self.run_ensure(client)
        launch.assert_not_called()
        self.assertIn("Reusing server", self.messages[-1])

    def test_refused_then_start_and_wait_until_ready(self):
        calls = []
        def handler(request):
            calls.append(request.url.path)
            if len(calls) < 3:
                return refused(request)
            return httpx.Response(200, json={"version": "test"})
        process = Mock()
        process.poll.return_value = None
        with patch("core.ollama.shutil.which", return_value="/existing/ollama"), \
                patch("core.ollama._launch", return_value=process) as launch, \
                patch("core.ollama.time.sleep"):
            self.run_ensure(self.client(handler))
        launch.assert_called_once_with("/existing/ollama", "127.0.0.1:11434", self.log)
        self.assertEqual(calls, ["/api/version"] * 3)
        self.assertIn("Ready", self.messages[-1])
        process.terminate.assert_not_called()

    def test_disabled_startup_does_not_launch(self):
        with patch("core.ollama._launch") as launch:
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                self.run_ensure(self.client(refused), auto_start=False)
        launch.assert_not_called()

    def test_missing_binary_is_clear_error_no_install(self):
        with patch("core.ollama.shutil.which", return_value=None), \
                patch("core.ollama._launch") as launch:
            with self.assertRaisesRegex(FileNotFoundError, "No automatic install"):
                self.run_ensure(self.client(refused))
        launch.assert_not_called()

    def test_remote_or_https_endpoint_never_started_locally(self):
        with patch("core.ollama._launch") as launch:
            for url in ("http://approved.internal:11434", "https://127.0.0.1:11434"):
                self.cfg.enrichment.url = url
                with self.assertRaisesRegex(RuntimeError, "loopback"):
                    self.run_ensure(self.client(refused))
        launch.assert_not_called()

    def test_non_refusal_connect_error_not_treated_as_stopped_server(self):
        def handler(request):
            raise httpx.ConnectError("certificate or permission error", request=request)
        with patch("core.ollama._launch") as launch:
            with self.assertRaisesRegex(RuntimeError, "not a refused"):
                self.run_ensure(self.client(handler))
        launch.assert_not_called()

    def test_existing_wrong_http_service_is_not_restarted(self):
        with patch("core.ollama._launch") as launch:
            for status, body in ((404, {}), (200, {"service": "other"}), (302, {})):
                client = self.client(lambda request: httpx.Response(status, json=body))
                with self.assertRaises((httpx.HTTPStatusError, RuntimeError)):
                    self.run_ensure(client)
        launch.assert_not_called()

    def test_startup_exit_reports_log(self):
        process = Mock()
        process.poll.return_value = 1
        with patch("core.ollama.shutil.which", return_value="ollama"), \
                patch("core.ollama._launch", return_value=process):
            with self.assertRaisesRegex(RuntimeError, "exited with code=1"):
                self.run_ensure(self.client(refused))

    def test_startup_timeout_reports_log_without_killing_server(self):
        process = Mock()
        process.poll.return_value = None
        with patch("core.ollama.shutil.which", return_value="ollama"), \
                patch("core.ollama._launch", return_value=process), \
                patch("core.ollama.time.monotonic", side_effect=[0, 0, 10]):
            with self.assertRaisesRegex(RuntimeError, "not ready within"):
                self.run_ensure(self.client(refused), startup_timeout=1)
        process.terminate.assert_not_called()

    def test_custom_local_port_and_ipv6_match_binding(self):
        for url, address in (("http://127.0.0.1:12345", "127.0.0.1:12345"),
                             ("http://[::1]:11434", "[::1]:11434")):
            self.cfg.enrichment.url = url
            calls = []
            def handler(request):
                calls.append(request)
                if len(calls) == 1:
                    return refused(request)
                return httpx.Response(200, json={"version": "test"})
            with patch("core.ollama.shutil.which", return_value="ollama"), \
                    patch("core.ollama._launch") as launch:
                self.run_ensure(self.client(handler))
            launch.assert_called_once_with("ollama", address, self.log)

    def test_launch_uses_existing_executable_local_env_and_detaches(self):
        with patch.dict(os.environ, {"OLLAMA_NO_CLOUD": "0", "OLLAMA_MODELS": "existing-models"}), \
                patch("core.ollama.sys.platform", "linux"), \
                patch("core.ollama.subprocess.Popen") as popen:
            _launch("/existing/ollama", "127.0.0.1:11434", self.log)
        args, kwargs = popen.call_args
        self.assertEqual(args, (["/existing/ollama", "serve"],))
        self.assertEqual(kwargs["env"]["OLLAMA_NO_CLOUD"], "1")
        self.assertEqual(kwargs["env"]["OLLAMA_HOST"], "127.0.0.1:11434")
        self.assertEqual(kwargs["env"]["OLLAMA_MODELS"], "existing-models")
        self.assertTrue(kwargs["start_new_session"])
        self.assertTrue(self.log.is_file())

    def test_windows_launch_is_detached(self):
        with patch("core.ollama.sys.platform", "win32"), \
                patch("core.ollama.subprocess.DETACHED_PROCESS", 8, create=True), \
                patch("core.ollama.subprocess.CREATE_NEW_PROCESS_GROUP", 512, create=True), \
                patch("core.ollama.subprocess.Popen") as popen:
            _launch("C:/existing/ollama.exe", "127.0.0.1:11434", self.log)
        kwargs = popen.call_args.kwargs
        self.assertEqual(kwargs["creationflags"], 520)
        self.assertNotIn("start_new_session", kwargs)

    def make_input(self):
        text = "设备校准需要采集测试数据。"
        source = {"video_id": "demo", "start": 0, "end": 5, "text": text}
        chunk = {"chunk_id": "demo", "video_id": "demo", "start_ts": 0,
                 "end_ts": 5, "text": text, "source_segments": [source],
                 "timestamp_quality": "asr"}
        path = Path(self.temp.name) / "chunks.jsonl"
        write_rows(path, [chunk])
        return path, text

    def test_enrich_cli_starts_service_before_show_and_chat(self):
        source, text = self.make_input()
        output = Path(self.temp.name) / "output" / "enriched.jsonl"
        calls = []
        def handler(request):
            calls.append(request.url.path)
            if len(calls) == 1:
                return refused(request)
            if request.url.path == "/api/version":
                return httpx.Response(200, json={"version": "test"})
            if request.url.path == "/api/show":
                return httpx.Response(200, json={"details": {"family": "test"}})
            questions = ["设备校准需要什么？", "什么过程需要采集数据？", "需要采集哪种数据？"]
            content = {"summary": text, "questions": [
                {"question": q, "evidence": text} for q in questions]}
            return httpx.Response(200, json={"message": {"content": json.dumps(content)}})
        client = self.client(handler)
        with patch("sys.argv", ["enrich", "--input", str(source), "--output", str(output)]), \
                patch("ingest.enrich.load_config", return_value=self.cfg), \
                patch("ingest.enrich.httpx.Client", return_value=client), \
                patch("core.ollama.shutil.which", return_value="ollama"), \
                patch("core.ollama._launch") as launch, \
                contextlib.redirect_stdout(io.StringIO()):
            enrich_main()
        self.assertEqual(calls, ["/api/version", "/api/version", "/api/show", "/api/chat"])
        launch.assert_called_once()
        saved = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(saved["enrichment_status"], "validated")

    def test_startup_failure_preserves_existing_enriched_output(self):
        source, _ = self.make_input()
        output = Path(self.temp.name) / "enriched.jsonl"
        output.write_text("previous output", encoding="utf-8")
        calls = []
        def handler(request):
            calls.append(request.url.path)
            return httpx.Response(404, json={})
        client = self.client(handler)
        with patch("sys.argv", ["enrich", "--input", str(source), "--output", str(output)]), \
                patch("ingest.enrich.load_config", return_value=self.cfg), \
                patch("ingest.enrich.httpx.Client", return_value=client):
            with self.assertRaises(httpx.HTTPStatusError):
                enrich_main()
        self.assertEqual(output.read_text(encoding="utf-8"), "previous output")
        self.assertEqual(calls, ["/api/version"])


if __name__ == "__main__":
    unittest.main()
