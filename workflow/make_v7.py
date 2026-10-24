"""Generate the one-slot prefetch V7 runner from the complete, tested V6.

The previous V6 script is intentionally not modified. V7 preserves the known
working PowerShell Unicode directory scanner and final RAG indexing steps.
"""
from pathlib import Path
import sys


OLD = '2026-10-24-cloud-directory-scan-v6-transcript-registry'
NEW = '2026-10-24-cloud-directory-scan-v7-audio-prefetch'
ENTRY = 'complete_transcript() {'
END = '# Create a complete snapshot'
WORKER = '''# OneDrive preparation and Whisper run concurrently across adjacent videos.
# Names and fully verified transcripts are resolved BEFORE opening cloud content.
# The Python worker saves prepared audio for retry, and deletes only its own
# staged .wav after JSONL + audit validation and registry registration.
log "开始受限流水线：单后台预取 + 前台 Whisper；完整视频不重复处理。"
log "下载/音轨与 ASR 计时：$DATA/logs/pipeline-*.log"
python -m workflow.pipelined_asr \\
  --manifest "$(winpath "$CACHE_FILE")" \\
  --source "$(winpath "$SOURCE")" \\
  --data "$(winpath "$DATA")" \\
  || die '流水线停止；已完成窗口及音轨保留。修复后重跑自动复用，不会继续下载后续视频。'

'''


def upgrade(source: str) -> str:
    if OLD not in source or ENTRY not in source or END not in source:
        raise ValueError('The source must be the complete V6 registry shell runner')
    a = source.index(ENTRY)
    b = source.index(END, a)
    source = source[:a] + WORKER + source[b:]
    source = source.replace(OLD, NEW)
    source = source.replace('if (( SCRIPT_LINES < 420 ))',
                            'if (( SCRIPT_LINES < 315 ))')
    source = source.replace('import workflow.transcript_registry\'','import workflow.pipelined_asr\'')
    source = source.replace('run_video_rag_registry_v6.sh', 'run_video_rag_prefetch_v7.sh')
    source = source.replace('recursive, resumable, single-video-at-a-time ingestion.',
                            'recursive, resumable, bounded 1-slot audio prefetch + Whisper ingestion.')
    if 'python -m workflow.pipelined_asr' not in source:
        raise ValueError('Prefetch entrypoint absent')
    if not source.endswith('# END_VIDEO_RAG_SCRIPT_V2\n'):
        raise ValueError('End marker missing; refusing to create truncated script')
    return source


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit('Usage: python -m workflow.make_v7 INPUT_V6.sh [OUTPUT_V7.sh]')
    original = Path(sys.argv[1])
    target = Path(sys.argv[2]) if len(sys.argv) == 3 else original.with_name('run_video_rag_prefetch_v7.sh')
    if original.resolve() == target.resolve():
        raise ValueError('Will not overwrite original V6')
    text = upgrade(original.read_text(encoding='utf-8'))
    target.write_text(text, encoding='utf-8', newline='\n')
    print(f'V7 written: {target} ({len(text.splitlines())} lines); run bash -n before use.')


if __name__ == '__main__':
    main()
