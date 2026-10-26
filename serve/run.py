"""Launch the local search page against an existing index; no rebuilding."""
import argparse
import os
from pathlib import Path
import secrets

from core.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", default=None, help="Existing directory containing CURRENT")
    parser.add_argument("--config", default=None)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if args.config:
        os.environ["RAG_CONFIG"] = args.config
    if args.index_root:
        os.environ["RAG_STORAGE__ROOT"] = args.index_root
    cfg = load_config()
    if not (Path(cfg.storage.root) / "CURRENT").is_file():
        parser.error(f"Index CURRENT not found in {cfg.storage.root}; use --index-root")
    if not os.environ.get("RAG_API_KEY"):
        os.environ["RAG_API_KEY"] = secrets.token_urlsafe(24)
    print(f"Index: {cfg.storage.root}", flush=True)
    print(f"Open after application startup completes: http://127.0.0.1:{args.port}/", flush=True)
    print(f"Access key (paste into the page): {os.environ['RAG_API_KEY']}", flush=True)
    print("Keep this terminal open. Ctrl+C stops the search service.", flush=True)
    import uvicorn
    uvicorn.run("serve.api:app", host="127.0.0.1", port=args.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
