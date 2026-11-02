"""One input path: bootstrap, local ASR, review, chunk, embed, index and search."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import venv

ROOT = Path(__file__).resolve().parents[1]
MEDIA = {'.mp4', '.mkv', '.mov', '.avi', '.webm', '.m4v', '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma'}
ASR_MODELS = {'medium': 'Systran/faster-whisper-medium',
              'large-v3': 'Systran/faster-whisper-large-v3'}


def report(message):
    print(f'[LAB] {message}', flush=True)


@contextmanager
def heartbeat(label):
    stopped = threading.Event()
    started = time.monotonic()
    def tick():
        while not stopped.wait(20):
            report(f'{label}: still running, elapsed={time.monotonic()-started:.0f}s')
    thread = threading.Thread(target=tick, daemon=True)
    thread.start()
    report(label)
    try:
        yield
    finally:
        stopped.set()
        thread.join()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, data):
    # Used only by the single controlling process (before filelock is installed).
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def child_env():
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith('RAG_') and ('__' in k or k == 'RAG_CONFIG'))}
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
               TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_TELEMETRY='1',
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
    spec = importlib.util.find_spec('torch')
    if spec and spec.submodule_search_locations:
        torch_root = Path(next(iter(spec.submodule_search_locations)))
        libraries = [torch_root / 'lib', *sorted((torch_root.parent / 'nvidia').glob('*/lib'))]
        key = 'PATH' if os.name == 'nt' else 'LD_LIBRARY_PATH'
        env[key] = os.pathsep.join(str(p) for p in libraries if p.is_dir()) + os.pathsep + env.get(key, '')
    return env


def nvidia_info():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                                 '--format=csv,noheader'], capture_output=True, text=True,
                                timeout=10, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ''


def bootstrap():
    directory = ROOT / '.venv-lab'
    python = directory / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.is_file():
        report('Creating .venv-lab (Python 3.11/3.12)')
        venv.EnvBuilder(with_pip=True).create(directory)
    gpu = bool(nvidia_info())
    identity = sha((ROOT / 'requirements.txt').read_bytes()
                   + (ROOT / 'requirements-lab.txt').read_bytes()
                   + f'torch=2.7.1;cuda126={gpu}'.encode())
    marker = directory / 'lab-setup.json'
    ready = json.loads(marker.read_text(encoding='utf-8')) if marker.is_file() else {}
    if ready.get('signature') != identity:
        env = child_env()
        report('First setup downloads Python packages and model weights; media stays local.')
        with heartbeat('Install inference libraries'):
            index = 'https://download.pytorch.org/whl/' + ('cu126' if gpu else 'cpu')
            subprocess.run([str(python), '-m', 'pip', 'install', 'torch==2.7.1',
                            '--index-url', index], env=env, check=True, cwd=ROOT)
            subprocess.run([str(python), '-m', 'pip', 'install', '-r',
                            str(ROOT / 'requirements-lab.txt')], env=env, check=True, cwd=ROOT)
            subprocess.run([str(python), '-m', 'pip', 'check'], env=env, check=True)
        atomic_json(marker, {'signature': identity, 'torch_index': index})
    return subprocess.call([str(python), str(Path(__file__).resolve()), *sys.argv[1:]],
                           cwd=ROOT, env=child_env())


def input_path(value=None):
    text = value if value is not None else (ROOT / 'lab_input.txt').read_text(encoding='utf-8-sig')
    text = text.strip().strip('"')
    if not text or '\n' in text or '\r' in text:
        raise ValueError('lab_input.txt must contain exactly one video or folder path.')
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f'Input not found: {path}. Edit lab_input.txt.')
    if path.is_file() and path.suffix.lower() not in MEDIA:
        raise ValueError('Input must be a supported video/audio file or a folder.')
    return path


def discover(source):
    if source.is_file():
        return [source]
    videos = []
    def scan_error(error):
        raise error  # Do not silently claim a complete scan after permission failures.
    with heartbeat('Scan media filenames (no PDF/PPT/Word contents)'):
        for number, (base, dirs, names) in enumerate(os.walk(source, onerror=scan_error), 1):
            dirs[:] = sorted(d for d in dirs if not (Path(base) / d).is_symlink())
            for name in sorted(names):
                path = Path(base) / name
                if path.suffix.lower() in MEDIA and not path.is_symlink():
                    videos.append(path)
                    report(f'Found media {len(videos)}: {path.name}')
            if number % 25 == 0:
                report(f'Scanned folders={number}, media={len(videos)}')
    if not videos:
        raise ValueError('No supported media files found.')
    return videos


def workspace(source):
    key = sha(os.path.normcase(str(source)).encode())[:12]
    return ROOT / 'data' / 'lab' / key


def manifest_for(source, videos, folder):
    result = []
    for video in videos:
        key = sha(os.path.normcase(str(video)).encode())[:12]
        stem = re.sub(r'[^\w.-]+', '_', video.stem)[:60].strip('. ') or 'video'
        result.append({'video': str(video),
                       'video_id': video.relative_to(source).as_posix() if source.is_dir() else video.name,
                       'output': str(folder / 'transcripts' / f'video_{stem}-{key}.jsonl')})
    return result


def prepare_models(asr_name):
    if asr_name not in ASR_MODELS:
        raise ValueError('LAB_ASR_MODEL must be medium or large-v3.')
    # Prepare weights explicitly; workers below run offline. No media is uploaded.
    os.environ['HF_HUB_OFFLINE'] = '0'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    paths = {}
    for kind, model in [('asr', ASR_MODELS[asr_name]), ('embedding', 'BAAI/bge-m3'),
                        ('reranker', 'BAAI/bge-reranker-v2-m3')]:
        options = ({'allow_patterns': ['config.json', 'model.bin', 'tokenizer.json',
                                      'vocabulary.*', 'preprocessor_config.json']}
                   if kind == 'asr' else
                   {'ignore_patterns': ['onnx/*', '*.onnx', '*.bin', '*.h5', '*.msgpack', '*.ot']})
        with heartbeat(f'Prepare {kind}: {model} (reuse Hugging Face cache first)'):
            try:
                path = snapshot_download(model, local_files_only=True, **options)
            except LocalEntryNotFoundError:
                path = snapshot_download(model, **options)
            # Cache lookup alone can return an incomplete snapshot. Verify required files
            # before offline inference, and let Hub download only the missing files.
            path = Path(path)
            def complete_snapshot(path):
                required = (['model.bin', 'config.json', 'tokenizer.json'] if kind == 'asr'
                            else ['config.json', 'tokenizer.json', 'tokenizer_config.json'])
                if kind == 'embedding':
                    required.append('modules.json')
                    if (path / 'modules.json').is_file():
                        modules = json.loads((path / 'modules.json').read_text(encoding='utf-8'))
                        required.extend(str(Path(m['path']) / 'config.json') for m in modules if m['path'])
                if kind != 'asr':
                    if not any(path.glob('*.safetensors')):
                        return False
                    index = path / 'model.safetensors.index.json'
                    if index.is_file():
                        required.extend(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values())
                return all((path / name).is_file() for name in required)
            if not complete_snapshot(path):
                path = Path(snapshot_download(model, revision=path.name, **options))
            if not complete_snapshot(path):
                raise RuntimeError(f'Incomplete model snapshot: {model}; retry setup when the download source is available.')
            paths[kind] = str(path.resolve())
    return paths


def make_config(folder, paths):
    import torch
    from core.config import Settings
    cfg = Settings()
    cfg.asr.backend = 'faster-whisper'
    cfg.asr.model = paths['asr']
    cfg.asr.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.asr.compute_type = 'int8_float16' if cfg.asr.device == 'cuda' else 'int8'
    cfg.asr.language = os.environ.get('LAB_LANGUAGE', 'auto')
    cfg.models.embedding = paths['embedding']
    cfg.models.reranker = paths['reranker']
    cfg.models.device = 'cpu'  # 4GB GPU stays available for ASR; search models fit in 32GB RAM.
    cfg.models.batch_size = 8
    cfg.models.offline = True
    cfg.enrichment.mode = 'extractive'
    cfg.storage.root = str(folder / 'index')
    cfg.retrieval.expand_after_rerank = True
    cfg.retrieval.context_seconds = 15
    return cfg


def run_stage(label, args, folder):
    log = folder / 'logs' / (label + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    with heartbeat(label), log.open('a', encoding='utf-8') as handle:
        command = [sys.executable, '-u', '-m', *map(str, args)]
        with subprocess.Popen(command, cwd=ROOT, env=child_env(), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as proc:
            try:
                for line in proc.stdout:
                    print(line, end='', flush=True)
                    handle.write(line)
                    handle.flush()
                if proc.wait():
                    raise subprocess.CalledProcessError(proc.returncode, command)
            except BaseException:
                if proc.poll() is None:
                    proc.terminate()
                proc.wait()
                raise


def pipeline_code():
    return b''.join(p.read_bytes() for package in ('tools', 'core', 'ingest', 'transcribe')
                    for p in sorted((ROOT / package).glob('*.py')))


def cached_stage(label, args, inputs, outputs, folder, config):
    # Include input content, config and pipeline code, so edits invalidate downstream output.
    signature = sha(config.read_bytes() + pipeline_code() + b''.join(Path(p).read_bytes() for p in inputs))
    stamp = folder / 'stages' / (label + '.json')
    previous = json.loads(stamp.read_text(encoding='utf-8')) if stamp.is_file() else {}
    actual = {str(p): sha(Path(p).read_bytes()) for p in outputs if Path(p).is_file()}
    if previous.get('signature') == signature and len(actual) == len(outputs) and previous.get('outputs') == actual:
        report(f'Reuse completed stage: {label}')
        return
    run_stage(label, args, folder)
    atomic_json(stamp, {'signature': signature,
                        'outputs': {str(p): sha(Path(p).read_bytes()) for p in outputs}})


_DLL_HANDLES = []


def configure_cuda_libraries():
    # CUDA torch wheels supply cuBLAS/cuDNN. Register their directories before CTranslate2.
    spec = importlib.util.find_spec('torch')
    if spec and os.name == 'nt':
        directory = Path(next(iter(spec.submodule_search_locations))) / 'lib'
        if directory.is_dir():
            os.environ['PATH'] = str(directory) + os.pathsep + os.environ.get('PATH', '')
            _DLL_HANDLES.append(os.add_dll_directory(str(directory)))
            import torch  # Load CUDA DLLs and retain their handles for the worker lifetime.


def asr_worker(manifest, config):
    configure_cuda_libraries()
    from core.config import load_config
    from core.vocabulary import Vocabulary
    from transcribe.faster import FasterWhisperAdapter
    from transcribe.run import transcribe_video
    from transcribe.review import export_review
    cfg = load_config(str(config))
    report(f'进入转录步骤: faster-whisper, device={cfg.asr.device}, compute={cfg.asr.compute_type}')
    model = None
    entries = json.loads(manifest.read_text(encoding='utf-8'))
    for number, item in enumerate(entries, 1):
        report(f'Transcribe video {number}/{len(entries)}: {item["video_id"]}')
        video, output = Path(item['video']), Path(item['output'])
        stat = video.stat()
        identity = sha(json.dumps([str(video), item['video_id'], stat.st_size, stat.st_mtime_ns,
                                  cfg.asr.model_dump()], sort_keys=True).encode() + pipeline_code())
        marker = Path(str(output) + '.lab.json')
        audit = Path(str(output) + '.audit.json')
        saved = json.loads(marker.read_text(encoding='utf-8')) if marker.is_file() else {}
        complete = (saved.get('identity') == identity and output.is_file() and audit.is_file()
                    and saved.get('output') == sha(output.read_bytes())
                    and saved.get('audit') == sha(audit.read_bytes()))
        if complete:
            report('Reuse complete transcript; audio decoding/model loading skipped.')
        else:
            if model is None:
                model = FasterWhisperAdapter(cfg)
            transcribe_video(video, item['video_id'], Vocabulary(), cfg,
                             model, model.token_count, output)
            atomic_json(marker, {'identity': identity, 'output': sha(output.read_bytes()),
                                 'audit': sha(audit.read_bytes())})
        try:
            export_review(item['output'])
        except FileExistsError:
            report('Existing complete review Markdown preserved.')


def assemble(manifest, folder):
    from core.io import read_rows, write_rows
    from transcribe.review import import_review
    rows = []
    for item in manifest:
        source = Path(item['output'])
        review = source.with_suffix('.review.md')
        key = sha(source.read_bytes() + review.read_bytes())[:20]
        corrected = folder / 'corrected' / (source.stem + '-' + key + '.jsonl')
        if not corrected.exists():
            # Import validates source checksum and all paragraph IDs/times.
            import_review(source, review, corrected)
        rows.extend(read_rows(corrected))
    output = folder / 'effective-transcripts.jsonl'
    write_rows(output, rows)
    return output


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', help='Optional override; otherwise read lab_input.txt')
    parser.add_argument('--plan', action='store_true', help='Inspect input/plan without installing or loading models')
    parser.add_argument('--setup-only', action='store_true')
    parser.add_argument('--serve', action='store_true', help='Open local search after building')
    parser.add_argument('--search', action='store_true', help='Search existing lab index without rebuilding')
    parser.add_argument('--worker', choices=['asr'], help=argparse.SUPPRESS)
    parser.add_argument('--manifest', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--config', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    if args.worker:
        return asr_worker(args.manifest, args.config)
    if sys.version_info[:2] not in {(3, 11), (3, 12)}:
        raise RuntimeError('Python 3.11 or 3.12 required. Use the run_lab launcher.')
    source = input_path(args.input)
    folder = workspace(source)
    if args.plan:
        report(f'Input: {source}\nOutput: {folder}\nMedia: {len(discover(source))}\n'
               'ASR: faster-whisper medium INT8, one window at a time; CPU fallback\n'
               'Embedding: BGE-M3 CPU; reranker: bge-reranker-v2-m3 CPU\n'
               'Hardware budget: 32GB system RAM, 4GB dedicated VRAM; shared GPU RAM is system RAM.')
        return 0
    if Path(sys.prefix).resolve() != (ROOT / '.venv-lab').resolve():
        return bootstrap()
    from filelock import FileLock
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder / 'PIPELINE.lock'), timeout=0):
        config = folder / 'config.json'  # JSON is valid YAML, including Windows paths.
        if not args.search:
            paths = prepare_models(os.environ.get('LAB_ASR_MODEL', 'medium'))
            cfg = make_config(folder, paths)
            atomic_json(config, cfg.model_dump())
            if args.setup_only:
                report(f'Environment and models ready. Config: {config}')
                return 0
            entries = manifest_for(source, discover(source), folder)
            manifest = folder / 'videos.json'
            atomic_json(manifest, entries)
            asr_args = ['tools.lab', '--worker', 'asr', '--manifest', manifest, '--config', config]
            try:
                run_stage('asr', asr_args, folder)
            except subprocess.CalledProcessError:
                if cfg.asr.device != 'cuda':
                    raise
                report('CUDA ASR failed; retry same model on CPU INT8. See logs/asr.log for the original error.')
                cfg.asr.device, cfg.asr.compute_type = 'cpu', 'int8'
                atomic_json(config, cfg.model_dump())
                run_stage('asr-cpu', asr_args, folder)
            effective = assemble(entries, folder)
            chunks, enriched = folder / 'chunks.jsonl', folder / 'enriched.jsonl'
            cached_stage('chunk', ['ingest.chunk', '--input', effective, '--output', chunks, '--config', config],
                         [effective], [chunks], folder, config)
            cached_stage('enrich', ['ingest.enrich', '--input', chunks, '--output', enriched, '--config', config],
                         [chunks], [enriched], folder, config)
            current = folder / 'index' / 'CURRENT'
            # Index build itself verifies and reuses saved embedding batches.
            stamp = folder / 'index-input.sha256'
            signature = sha(enriched.read_bytes() + config.read_bytes() + pipeline_code())
            if not (current.is_file() and stamp.is_file() and stamp.read_text() == signature
                    and (current.parent / current.read_text().strip() / 'manifest.json').is_file()):
                run_stage('index', ['ingest.index', '--input', enriched, '--config', config], folder)
                stamp.write_text(signature, encoding='ascii')
            else:
                report('Reuse published index (input/config unchanged).')
            report(f'Complete. Full review Markdown: {folder / "transcripts"}\nIndex: {folder / "index"}')
        if args.search or args.serve:
            if not config.is_file() or not (folder / 'index' / 'CURRENT').is_file():
                raise FileNotFoundError('No lab index. Run the launcher once without --search.')
            run_stage('search', ['serve.run', '--config', config, '--open-browser'], folder)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        report('Stopped. Saved ASR windows/vectors remain available for resume.')
        sys.exit(130)
    except Exception as exc:
        report(f'FAILED: {exc}')
        sys.exit(1)
