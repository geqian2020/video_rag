"""Reuse Ollama or start a detached local server; never install or pull models."""
import errno
import ipaddress
import os
from pathlib import Path
import shutil
import socket
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


def _require_free_local_port(host, port):
    """Check binding only; do not listen, connect, kill or restart another process."""
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    checked = set()
    for family, kind, protocol, _, address in addresses:
        key = (family, address)
        if key in checked:
            continue
        checked.add(key)
        if not ipaddress.ip_address(address[0]).is_loopback:
            raise RuntimeError(f"Refusing to bind non-loopback address resolved from {host}")
        try:
            with socket.socket(family, kind, protocol) as probe:
                # Windows normally allows some shared binds unless this is set.
                if sys.platform == "win32":
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                probe.bind(address)
        except OSError as error:
            raise RuntimeError(
                f"Ollama connection timed out at {host}:{port}, but its local port is occupied "
                "or cannot be bound. No new server was started. Check the configured URL, "
                f"existing listener and local network controls. Bind error: {error}") from error
    if not checked:
        raise RuntimeError(f"No local address resolved for Ollama host: {host}")


def ensure_ollama(client, cfg, logfile, *, auto_start=True, startup_timeout=30, report=print):
    """Start on local refusal, or connection timeout with a verified free port.

    Existing servers are not restarted; their environment remains their owner's
    responsibility. The caller still checks that the selected model is local.
    A server started here remains alive for later pipeline runs.
    """
    url = cfg.enrichment.url
    report(f"[OLLAMA] Checking server: url={url}, model={cfg.enrichment.model}")
    connection_timed_out = False
    try:
        version = _probe(client, url)
    except httpx.ConnectTimeout as error:
        connection_timed_out = True
        report(f"[OLLAMA] Initial TCP connection timed out: url={url}, probe timeout=2s")
        if not auto_start:
            raise RuntimeError(f"Ollama connection timed out at {url}; automatic startup is disabled") from error
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
    if connection_timed_out:
        report(f"[OLLAMA] Checking local port before startup: {parsed.hostname}:{parsed.port or 80}")
        _require_free_local_port(parsed.hostname, parsed.port or 80)
        report("[OLLAMA] Local port is free; attempting automatic startup")
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
