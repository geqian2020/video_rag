"""Launch the local search page against an existing index; no rebuilding."""
import argparse
import os
from pathlib import Path
import secrets
import threading
import webbrowser

from core.config import load_config


def choose_index(cfg, repo_root, explicit=None):
    """Honor explicit paths; use the deployed sibling data for the default config."""
    if explicit:
        return Path(explicit)
    if os.environ.get("RAG_STORAGE__ROOT"):
        return Path(cfg.storage.root)
    if Path(cfg.storage.root) == Path("data/index"):
        sibling = Path(repo_root).parent / "video_rag_data" / "index"
        if (sibling / "CURRENT").is_file():
            return sibling
    return Path(cfg.storage.root)


def open_when_ready(server, url, stopped):
    """Server.started is set after lifespan/model loading and socket binding."""
    while not stopped.is_set():
        if server.started:
            try:
                if not webbrowser.open(url):
                    print(f"Browser could not be opened; visit {url}", flush=True)
            except Exception:
                print(f"Browser could not be opened; visit {url}", flush=True)
            return
        stopped.wait(0.25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", default=None, help="Existing directory containing CURRENT")
    parser.add_argument("--config", default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open-browser", action="store_true", help="Open the page after models and server are ready")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if args.config:
        os.environ["RAG_CONFIG"] = args.config
    if args.index_root:
        os.environ["RAG_STORAGE__ROOT"] = args.index_root
    cfg = load_config()
    root = choose_index(cfg, Path(__file__).resolve().parents[1], args.index_root).resolve()
    if not (root / "CURRENT").is_file():
        parser.error(f"Index CURRENT not found in {root}; use --index-root")
    cfg.storage.root = str(root)
    os.environ["RAG_STORAGE__ROOT"] = str(root)
    if not os.environ.get("RAG_API_KEY"):
        os.environ["RAG_API_KEY"] = secrets.token_urlsafe(24)
    print(f"Index: {cfg.storage.root}", flush=True)
    print(f"Open after application startup completes: http://127.0.0.1:{args.port}/", flush=True)
    print(f"Access key (paste into the page): {os.environ['RAG_API_KEY']}", flush=True)
    print("Keep this terminal open. Ctrl+C stops the search service.", flush=True)
    import uvicorn
    if not args.open_browser:
        uvicorn.run("serve.api:app", host="127.0.0.1", port=args.port, workers=1, access_log=False)
        return
    config = uvicorn.Config("serve.api:app", host="127.0.0.1", port=args.port,
                            workers=1, access_log=False)
    server = uvicorn.Server(config)
    stopped = threading.Event()
    opener = threading.Thread(target=open_when_ready,
                              args=(server, f"http://127.0.0.1:{args.port}/", stopped), daemon=True)
    opener.start()
    try:
        server.run()
    finally:
        stopped.set()
        opener.join(timeout=1)


if __name__ == "__main__":
    main()
