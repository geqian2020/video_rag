"""Reuse Ollama or start a detached local server; never install or pull models."""
import errno
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.parse import urlparse

import httpx


def _refused(error):
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, OSError) and (
                current.errno == errno.ECONNREFUSED or getattr(current, "winerror", None) == 10061):
            return True
        pending.extend(e for e in (current.__cause__, current.__context__, *current.args)
                       if isinstance(e, BaseException))
    # Some HTTP backends retain the OS error only in the message.
    return "10061" in str(error) or "connection refused" in str(error).lower()


def _probe(client, url):
    response = client.get(url.rstrip("/") + "/api/version", timeout=2)
    response.raise_for_status()
    info = response.json()
    if not isinstance(info, dict) or not isinstance(info.get("version"), str) or not info["version"]:
        raise RuntimeError(f"Endpoint did not identify as Ollama: {url}")
    return info["version"]


def _launch(executable, address, logfile):
    logfile = Path(logfile)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "OLLAMA_NO_CLOUD": "1", "OLLAMA_HOST": address}
    options = {"stdin": subprocess.DEVNULL, "env": environment, "close_fds": True}
    if sys.platform == "win32":
        options["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    with logfile.open("ab", buffering=0) as log:
        return subprocess.Popen([executable, "serve"], stdout=log, stderr=log, **options)


def ensure_ollama(client, cfg, logfile, *, auto_start=True, startup_timeout=30, report=print):
    """Only an explicitly refused loopback connection triggers server startup.

    Existing servers are not restarted; their environment remains their owner's
    responsibility. The caller still checks that the selected model is local.
    A server started here remains alive for later pipeline runs.
    """
    url = cfg.enrichment.url
    try:
        version = _probe(client, url)
    except httpx.ConnectError as error:
        if not _refused(error):
            raise RuntimeError(f"Cannot connect to Ollama; not a refused connection: {url}") from error
        if not auto_start:
            raise RuntimeError(f"Ollama is not running at {url}; automatic startup is disabled") from error
    else:
        report(f"[OLLAMA] Reusing server: url={url}, version={version}")
        return

    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("Ollama automatic startup supports local HTTP loopback URLs only; "
                           f"start the configured service separately: {url}")
    if parsed.path not in {"", "/"} or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Ollama URL must be a host root without credentials")
    executable = shutil.which("ollama")
    if not executable:
        raise FileNotFoundError("Ollama executable not found in PATH; use the existing installation "
                                "or install it through your approved channel. No automatic install was attempted.")
    host = f"[{parsed.hostname}]" if parsed.hostname == "::1" else parsed.hostname
    address = f"{host}:{parsed.port or 80}"
    report(f"[OLLAMA] Starting local server: address={address}, cloud disabled, log={logfile}")
    process = _launch(executable, address, logfile)
    deadline = time.monotonic() + startup_timeout
    last_report = time.monotonic()
    while True:
        try:
            version = _probe(client, url)
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        else:
            report(f"[OLLAMA] Ready: url={url}, version={version}")
            return
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"Ollama server exited with code={code}; inspect {logfile}")
        now = time.monotonic()
        if now >= deadline:
            raise RuntimeError(f"Ollama was not ready within {startup_timeout}s; inspect {logfile}. "
                               "The server was not stopped; it may still be starting.")
        if now - last_report >= 5:
            report("[OLLAMA] Waiting for the local server...")
            last_report = now
        time.sleep(0.5)
