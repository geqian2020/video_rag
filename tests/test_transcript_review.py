"""Human corrections must survive reruns without losing video/time provenance."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from core.config import Settings
from core.io import read_rows, write_rows
from core.vocabulary import Vocabulary
from transcribe.review import export_review, export_reviews, import_review, paragraphs
from transcribe import run
from ingest.chunk import build_chunks


class TranscriptReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / '中文 路径.jsonl'
        self.rows = [dict(video_id='training/一', start=0, end=12, text='PC 软件负责'),
                     dict(video_id='training/一', start=12, end=32, text='调度流程。'),
                     dict(video_id='training/一', start=32, end=42, text='下一段。'),
                     dict(video_id='training/一', start=70, end=74, text='最后一段。')]
        write_rows(self.source, self.rows)
        self.output = self.root / 'corrected.jsonl'

    def test_roundtrip_unchanged_retains_all_segment_times(self):
        review = export_review(self.source)
        self.assertIn('PC 软件负责调度流程。', review.read_text(encoding='utf-8'))
        import_review(self.source, review, self.output)
        actual = read_rows(self.output)
        self.assertEqual([{k: r[k] for k in self.rows[0]} for r in actual], self.rows)

    def test_correction_preserves_envelope_and_original_file(self):
        original = self.source.read_bytes()
        review = export_review(self.source)
        review.write_text(review.read_text(encoding='utf-8').replace(
            'PC 软件负责调度流程。', 'PC 软件负责编排整个测试流程。\n\n可以分段修正。'), encoding='utf-8')
        import_review(self.source, review, self.output)
        rows = read_rows(self.output)
        self.assertEqual((rows[0]['start'], rows[0]['end']), (0, 32))
        self.assertEqual(rows[0]['timestamp_quality'], 'estimated')
        self.assertIn('编排', rows[0]['text'])
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(rows[-1]['video_id'], 'training/一')
        self.assertEqual(rows[-1]['timestamp_quality'], 'asr')
        with self.assertRaises(FileExistsError):
            export_review(self.source)
        with self.assertRaises(FileExistsError):
            import_review(self.source, review, self.output)

    def test_changed_source_or_missing_reordered_markers_rejected(self):
        review = export_review(self.source)
        original = review.read_text(encoding='utf-8')
        for bad in (original.replace('P000001', 'P000099'),
                    original.replace('00:00:32.000', '00:00:33.000'),
                    original.replace('PC 软件负责调度流程。', '')):
            review.write_text(bad, encoding='utf-8')
            with self.assertRaises(ValueError):
                import_review(self.source, review, self.output)
            self.assertFalse(self.output.exists())
        review.write_text(original, encoding='utf-8')
        write_rows(self.source, self.rows[:1])
        with self.assertRaises(ValueError):
            import_review(self.source, review, self.output)

    def test_hard_limits_gaps_and_video_changes(self):
        rows = [dict(video_id='v', start=i * 20, end=i * 20 + 20, text='fragment')
                for i in range(5)]
        groups = paragraphs(rows)
        self.assertEqual([len(g) for g in groups], [3, 2])
        self.assertEqual(len(paragraphs(self.rows)), 3)
        self.assertEqual(len(paragraphs([self.rows[0], dict(self.rows[1], video_id='other')])), 2)
        self.assertEqual(len(paragraphs([self.rows[0], dict(self.rows[1], start=30)])), 2)
        for maximum in (-1, float('nan'), float('inf'), 5):
            with self.assertRaises(ValueError):
                paragraphs(rows, max_seconds=maximum)

    def test_cli_batch_and_import_without_models_or_media(self):
        second = self.root / 'nested' / 'second.jsonl'
        write_rows(second, self.rows)
        subprocess.run([sys.executable, '-m', 'transcribe.review', 'export', '--input',
                        str(self.root)], check=True, capture_output=True)
        self.assertTrue(second.with_suffix('.review.md').is_file())
        review = self.source.with_suffix('.review.md')
        subprocess.run([sys.executable, '-m', 'transcribe.review', 'import', '--source',
                        str(self.source), '--review', str(review), '--output',
                        str(self.output)], check=True, capture_output=True)
        self.assertEqual(len(read_rows(self.output)), 4)

    def test_mixed_input_creates_one_complete_md_per_video(self):
        other = dict(video_id='other/一', start=0, end=8, text='其他视频内容。')
        mixed = [self.rows[0], other, *self.rows[1:]]
        write_rows(self.source, mixed)
        with self.assertRaises(ValueError):
            export_review(self.source)
        outputs = export_reviews(self.source)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(len(set(p.name for p in outputs)), 2)  # 相同 basename 也不冲突。
        for i, path in enumerate(outputs):
            text = path.read_text(encoding='utf-8')
            meta = json.loads(text.splitlines()[0][len('<!-- video-rag-review '):-len(' -->')])
            expected = [r for r in mixed if r['video_id'] == meta['video_id']]
            output = self.root / f'corrected-{i}.jsonl'
            import_review(self.source, path, output)
            actual = read_rows(output)
            self.assertEqual([{k: r[k] for k in self.rows[0]} for r in actual], expected)
            if meta['video_id'] == self.rows[0]['video_id']:
                self.assertIn('PC 软件负责调度流程。', text)
                self.assertIn('最后一段。', text)
                self.assertNotIn(other['text'], text)
                path.write_text(text.replace('最后一段。', '修正后的结尾。'), encoding='utf-8')
                import_review(self.source, path, self.root / 'edited.jsonl')
                edited = read_rows(self.root / 'edited.jsonl')
                self.assertEqual(edited[-1]['text'], '修正后的结尾。')
                self.assertEqual({r['video_id'] for r in edited}, {self.rows[0]['video_id']})

    def test_multi_video_cli_preserves_existing_work_and_exports_remaining(self):
        write_rows(self.source, [*self.rows, dict(video_id='v2', start=1, end=3, text='second')])
        directory = self.root / 'all-md'
        command = [sys.executable, '-m', 'transcribe.review', 'export', '--input',
                   str(self.source), '--output', str(directory)]
        subprocess.run(command, check=True, capture_output=True)
        outputs = sorted(directory.glob('*.md'))
        self.assertEqual(len(outputs), 2)
        outputs[0].write_text('人工修正保留', encoding='utf-8')
        outputs[1].unlink()
        subprocess.run(command, check=True, capture_output=True)
        self.assertEqual(outputs[0].read_text(encoding='utf-8'), '人工修正保留')
        self.assertTrue(outputs[1].is_file())

    def test_corrected_paragraphs_can_be_chunked(self):
        import numpy as np
        review = export_review(self.source)
        review.write_text(review.read_text(encoding='utf-8').replace(
            'PC 软件负责调度流程。', 'PC 软件负责编排测试。'), encoding='utf-8')
        import_review(self.source, review, self.output)

        class Embedder:
            def token_count(self, text):
                return len(text)

            def encode(self, texts):
                return np.ones((len(texts), 2))

        chunks = build_chunks(read_rows(self.output), Embedder(), Settings())
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]['timestamp_quality'], 'estimated')
        self.assertEqual(chunks[0]['start_ts'], 0)
        self.assertIn('编排测试', chunks[0]['text'])

    def test_json_array_bom_and_english_punctuation(self):
        source = self.root / 'english.json'
        rows = [dict(video_id='en', start=0, end=2, text='Hello,'),
                dict(video_id='en', start=2, end=4, text='world.')]
        source.write_text(json.dumps(rows), encoding='utf-8-sig')
        review = export_review(source)
        text = review.read_text(encoding='utf-8')
        self.assertIn('Hello, world.', text)
        review.write_text(text, encoding='utf-8-sig')
        import_review(source, review, self.output)
        self.assertEqual([r['text'] for r in read_rows(self.output)], ['Hello,', 'world.'])
        with self.assertRaises(ValueError):
            import_review(source, review, self.root / 'wrong.json')

    def test_asr_cli_automatically_exports_and_preserves_existing_review(self):
        from types import SimpleNamespace
        model = SimpleNamespace(is_multilingual=True, num_languages=99,
                                dims=SimpleNamespace(n_text_ctx=448))
        tokenizer = SimpleNamespace(encode=lambda text: list(text))
        module = SimpleNamespace(get_tokenizer=lambda *a, **kw: tokenizer)
        args = ['transcribe.run', '--video', 'video.mp4', '--video-id', 'v',
                '--no-vocab', '--output', str(self.source)]
        with patch.object(sys, 'argv', args), patch.dict(
                sys.modules, {'whisper.tokenizer': module}), patch.object(
                run, 'load_config', return_value=Settings()), patch.object(
                run, 'local_media', return_value='video.mp4'), patch.object(
                run, 'load_local_whisper', return_value=model), patch.object(
                run, 'transcribe_video'), patch.object(run, 'log_progress'):
            run.main()
            review = self.source.with_suffix('.review.md')
            review.write_text('my manual corrections', encoding='utf-8')
            run.main()
        self.assertEqual(review.read_text(encoding='utf-8'), 'my manual corrections')

    def test_completed_asr_exports_without_loading_whisper_again(self):
        cfg = Settings()
        output = self.root / 'asr.jsonl'
        model = unittest.mock.Mock()
        model.transcribe.return_value = {'segments': [dict(start=1, end=3, text='hello')]}
        with patch.object(run, 'probe', return_value={'duration': 10}), patch.object(
                run, 'extract_audio', side_effect=lambda v, p, s, e: p.write_bytes(b'audio')):
            run.transcribe_video('source.mp4', 'v', Vocabulary(), cfg, model, len, output)
            review = export_review(output)
            run.transcribe_video('source.mp4', 'v', Vocabulary(), cfg, model, len, output)
        self.assertEqual(model.transcribe.call_count, 1)
        self.assertTrue(review.is_file())
        self.assertIn('hello', review.read_text(encoding='utf-8'))
        audit = json.loads(Path(str(output) + '.audit.json').read_text(encoding='utf-8'))
        self.assertIn('transcript_sha256', audit)


if __name__ == '__main__':
    unittest.main()
