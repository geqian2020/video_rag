"""Upgrade the distributed V5 Git Bash runner to persistent-registry V6.

Usage: python -m workflow.make_v6 path/to/run_video_rag_dedup_v5.sh [output.sh]
The original script is not modified.
"""
from pathlib import Path
import sys


def upgrade(source):
    text = Path(source).read_text(encoding="utf-8")
    if "cloud-directory-scan-v5-filename-dedup" not in text:
        raise ValueError("Expected the complete V5 filename-dedup runner")
    text = text.replace("2026-10-24-cloud-directory-scan-v5-filename-dedup",
                        "2026-10-24-cloud-directory-scan-v6-transcript-registry")
    text = text.replace("run_video_rag_dedup_v5.sh", "run_video_rag_registry_v6.sh")
    anchor = 'cd "$REPO"\n\n# Mandatory online-only mode'
    if anchor not in text:
        raise ValueError("Cannot locate repository entry point")
    text = text.replace(anchor, '''cd "$REPO"
# Abort before cloud hydration if Python dependencies are missing.
python -c 'import yaml, whisper, torch, qdrant_client; import workflow.transcript_registry' || die 'Python requirements/registry missing. Pull the repository first.'

# Mandatory online-only mode''')
    first = text.index("# Filename-based reuse (explicit user rule")
    last = text.index("success=0; skipped=0;", first)
    text = text[:first] + '''# Persistent registry. It only reads transcript outputs and the cached filename list.
REGISTRY="$DATA/transcript_registry.json"
log "Loading completed-transcript registry: $REGISTRY"
python -m workflow.transcript_registry rebuild \\
  "$(winpath "$REGISTRY")" "$SOURCE" "$(winpath "$CACHE_FILE")" "$(winpath "$DATA/transcripts")" \\
  || die 'Registry initialization failed; no media accessed.'

''' + text[last:]
    text = text.replace('  name_key="$(filename_key "$file")"\n  reused_duplicate=0',
                        '  reused_duplicate=0')
    first = text.index('  elif [[ -n "${completed_by_name[$name_key]:-}" ]]; then')
    last = text.index('  else\n    # OneDrive can hydrate', first)
    text = text[:first] + '''  elif python -m workflow.transcript_registry reuse \\
      "$(winpath "$REGISTRY")" "$file" "$(winpath "$out")" "$video_id"; then
    log "[$current/$total] Filename cache hit; cloud media not read: $rel"
    skipped=$((skipped+1))
    reused_duplicate=1
''' + text[last:]
    text = text.replace('''      failed=$((failed+1)); continue
    fi
    log "[$current/$total] 阶段2/2''',
                        '''      die "[$current/$total] Probe failed; stopping batch."
    fi
    log "[$current/$total] 阶段2/2''')
    text = text.replace('''        failed=$((failed+1)); continue
      fi
    else
      log "[$current/$total] 转录失败''',
                        '''        die "[$current/$total] Transcript verification failed; stopping."
      fi
    else
      log "[$current/$total] 转录失败''')
    text = text.replace('''      failed=$((failed+1)); continue
    fi
  fi

  # Register successful outputs''',
                        '''      die "[$current/$total] ASR failed; saved windows remain on disk."
    fi
  fi

  # Register successful outputs''')
    first = text.index("  # Register successful outputs for later filename duplicates")
    last = text.index("  if (( reused_duplicate )); then", first)
    text = text[:first] + '''  python -m workflow.transcript_registry register \\
    "$(winpath "$REGISTRY")" "$file" "$(winpath "$out")" \\
    || die "Cannot register completed transcript: $rel"
''' + text[last:]
    if "completed_by_name" in text or "filename_key()" in text:
        raise ValueError("Outdated dedup logic remains; aborting")
    return text


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit("Usage: python -m workflow.make_v6 input_v5.sh [output_v6.sh]")
    source = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) == 3 else source.with_name("run_video_rag_registry_v6.sh")
    if out.resolve() == source.resolve():
        raise SystemExit("Cannot overwrite V5")
    new = upgrade(source)
    out.write_text(new, encoding="utf-8", newline="\n")
    print(f"Created {out} ({len(new.splitlines())} lines). Run bash -n before using.")


if __name__ == "__main__":
    main()
