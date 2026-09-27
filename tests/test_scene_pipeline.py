"""CPU regressions for cache validity, first-run depth, rendering and video I/O."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

import run_scene_inference as runner
from datasets.scene_depth_io import (convert_video, decode_video_depth_frame, DEPTH_ENCODING,
                                     encode_depth, decode_depth)
from datasets.utils import iter_video_chunks, iter_mask_latents
from pipeline.gpu import choose_gpu
from pipeline.inference_config import inference_settings
from pipeline.scene_cache import (condition_ready, render_fingerprint, prediction_fingerprint,
                                 prediction_ready, complete_prediction, complete_render, file_digest, video_metadata)
from pipeline.scene_schema import padded_frame_count, validate_config
from pipeline.render_scene import render_one, prepare_source, project_source
from pipeline.depth_warper import DepthWarper
from pipeline.video_writer import video_writer


def setUpModule():
    torch.set_num_threads(1)


class SceneFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='inspatio-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.scene = self.root / 'scene'
        (self.scene / 'input').mkdir(parents=True)
        (self.scene / 'depth').mkdir()
        self.output = self.root / 'output'

    def make_scene(self, views=1, resolution=(16, 16)):
        height, width = resolution
        frames = 9
        for index in range(views):
            Image.fromarray(np.full((height, width, 3), 30 + 40 * index, np.uint8)).save(self.scene / 'input' / f'view_{index:02d}.png')
            name = 'depth.png' if views == 1 else f'depth_{index:02d}.png'
            Image.fromarray(np.full((height, width), 32768, np.uint16)).save(self.scene / 'depth' / name)
        (self.scene / 'input/prompt.txt').write_text('first prompt')
        (self.scene / 'depth/metadata.txt').write_text('0 2\n')
        np.savetxt(self.scene / 'depth/source_intrinsics.txt', np.tile(np.eye(3).reshape(1, 9), (views, 1)))
        np.savetxt(self.scene / 'depth/source_tcw.txt', np.tile(np.eye(4).reshape(1, 16), (views, 1)))
        np.savetxt(self.scene / 'input/target_tcw.txt', np.tile(np.eye(4).reshape(1, 16), (frames, 1)))
        meta = dict(output_id='scene', kind='image', views=views, frames=frames, valid_frames=frames,
                    fps=24, resolution=[height, width], depth_encoding=DEPTH_ENCODING,
                    target_intrinsics='estimated_source_view_00_fixed')
        (self.scene / 'scene.json').write_text(json.dumps(meta))
        return runner.load_record(self.scene)

    def make_video(self, path, frames=9, resolution=(16, 16)):
        height, width = resolution
        with video_writer(path, 24, width=width, height=height, lossless=True) as writer:
            for i in range(frames):
                writer.write(np.full((height, width, 3), i * 10, dtype=np.uint8).tobytes())


class CacheTests(SceneFixture):
    def test_render_and_prediction_invalidation(self):
        record = self.make_scene()
        render_one(record, torch.device('cpu'), self.output)
        self.assertTrue(condition_ready(record, self.output))
        original_render = render_fingerprint(record)
        item = runner.inference_record(record, self.output, 'weights-and-config-A')
        self.make_video(item['output_path'])
        complete_prediction(item)
        self.assertTrue(prediction_ready(item))
        (self.scene / 'input/prompt.txt').write_text('changed prompt')
        self.assertEqual(original_render, render_fingerprint(record))
        changed = runner.inference_record(record, self.output, 'weights-and-config-A')
        self.assertFalse(prediction_ready(changed))
        item['prediction_fingerprint'] = prediction_fingerprint(item, original_render, 'weights-and-config-B')
        self.assertFalse(prediction_ready(item))
        trajectory = self.scene / 'input/target_tcw.txt'
        values = np.loadtxt(trajectory)
        values[0, 3] = .1
        np.savetxt(trajectory, values)
        self.assertFalse(condition_ready(record, self.output))

    def test_all_render_inputs_are_fingerprinted(self):
        record = self.make_scene()
        for name in ('input/view_00.png', 'depth/depth.png', 'depth/metadata.txt',
                     'depth/source_tcw.txt', 'depth/source_intrinsics.txt', 'input/target_tcw.txt'):
            path = self.scene / name
            before = render_fingerprint(record)
            original = path.read_bytes()
            path.write_bytes(original + b'\n')
            self.assertNotEqual(before, render_fingerprint(record), name)
            path.write_bytes(original)
        for key, value in (('frames', 21), ('fps', 15), ('resolution', [32, 16])):
            changed = {**record, key: value}
            self.assertNotEqual(render_fingerprint(record), render_fingerprint(changed))

    def test_legacy_and_corrupt_completion_records_are_not_reused(self):
        record = self.make_scene()
        render_one(record, torch.device('cpu'), self.output)
        marker = self.output / 'scene/render_complete.json'
        marker.write_text('{"frames": 9}')
        self.assertFalse(condition_ready(record, self.output))
        marker.write_text('{')
        self.assertFalse(condition_ready(record, self.output))

    def test_changed_output_bytes_are_not_reused(self):
        record = self.make_scene()
        render_one(record, torch.device('cpu'), self.output)
        path = self.output / 'scene/render.mp4'
        path.write_bytes(path.read_bytes() + b'changed')
        self.assertFalse(condition_ready(record, self.output))

    def test_checkpoint_config_and_seed_fingerprints(self):
        folder = self.root / 'model'
        folder.mkdir()
        for name in ('models_t5_umt5-xxl-enc-bf16.pth', 'Wan2.1_VAE.pth', 'checkpoint.safetensors', 'config.json'):
            (folder / name).write_bytes(b'original')
        config = OmegaConf.create({'wan_model_folder': str(folder), 'generator': {'model_path': str(folder)}})
        checkpoint = folder / 'checkpoint.safetensors'
        before = inference_settings(config, checkpoint)
        self.assertNotEqual(before, inference_settings(config, checkpoint, seed=1))
        config.extra = 'changed'
        self.assertNotEqual(before, inference_settings(config, checkpoint))
        del config.extra
        architecture = folder / 'config.json'
        architecture.write_bytes(b'changed architecture')
        self.assertNotEqual(before, inference_settings(config, checkpoint))
        architecture.write_bytes(b'original')
        # Unused base DiT shards must not affect predictions or require hashing.
        (folder / 'diffusion_pytorch_model.safetensors').write_bytes(b'unused base weights')
        self.assertEqual(before, inference_settings(config, checkpoint))
        checkpoint.write_bytes(b'replaced')
        self.assertNotEqual(before, inference_settings(config, checkpoint))

    def test_digest_detects_replacement_even_when_mtime_restored(self):
        path = self.root / 'data'
        path.write_bytes(b'first')
        before = file_digest(path)
        stat = path.stat()
        path.write_bytes(b'other')
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertNotEqual(before, file_digest(path))


class DirectVideoTests(SceneFixture):
    def test_preparation_reuses_video_and_updates_prompt_and_trajectory(self):
        source_video = self.root / 'original.mp4'
        self.make_video(source_video, frames=1)
        trajectory = self.root / 'trajectory.txt'
        np.savetxt(trajectory, np.eye(4).reshape(1, 16))
        record = runner.direct_video_record(source_video, 'first prompt', trajectory, None, self.output)
        runner.materialize_direct_video(record)
        runner.validate_video_inputs(record)
        scene = Path(record['path'])
        prepared = scene / 'input/video.mp4'
        self.assertEqual(video_metadata(prepared), (1, 24, (480, 832)))
        original_mtime = prepared.stat().st_mtime_ns
        original_digest = file_digest(prepared)

        new_trajectory = self.root / 'new_trajectory.txt'
        target = np.eye(4)
        target[0, 3] = .1
        np.savetxt(new_trajectory, target.reshape(1, 16))
        record = runner.direct_video_record(source_video, 'new prompt', new_trajectory, None, self.output)
        runner.materialize_direct_video(record)
        runner.validate_video_inputs(record)
        self.assertEqual((scene / 'input/prompt.txt').read_text(), 'new prompt\n')
        np.testing.assert_allclose(runner.matrices(scene / 'input/target_tcw.txt', 4)[0], target)
        self.assertEqual(prepared.stat().st_mtime_ns, original_mtime)
        self.assertEqual(file_digest(prepared), original_digest)

    def test_same_name_video_cannot_overwrite_another_input(self):
        source_video = self.root / 'original.mp4'
        self.make_video(source_video, frames=1)
        trajectory = self.root / 'trajectory.txt'
        np.savetxt(trajectory, np.eye(4).reshape(1, 16))
        record = runner.direct_video_record(source_video, 'first prompt', trajectory, None, self.output)
        runner.materialize_direct_video(record)
        another_video = self.root / 'another' / source_video.name
        another_video.parent.mkdir()
        self.make_video(another_video, frames=1)
        conflict = runner.direct_video_record(another_video, 'new prompt', trajectory, None, self.output)
        with self.assertRaisesRegex(ValueError, 'already refers to another video'):
            runner.materialize_direct_video(conflict)
        self.assertEqual((Path(record['path']) / 'input/prompt.txt').read_text(), 'first prompt\n')


class RunnerTests(SceneFixture):
    def test_scene_pipeline_reuses_results_and_rebuilds_changed_inputs(self):
        self.make_scene(resolution=(480, 832))
        self.output.mkdir()
        unrelated = {'id': 'another-scene', 'scene_path': '/another/scene'}
        (self.output / 'manifest.json').write_text(json.dumps([unrelated]))
        stages = []
        run_process = runner.subprocess.run

        def worker(command, **kwargs):
            if 'pipeline.render_scene' in command:
                stages.append('render')
                records = json.loads(Path(command[command.index('--manifest') + 1]).read_text())
                for record in records:
                    for name in ('source', 'render', 'mask'):
                        self.make_video(self.output / record['id'] / f'{name}.mp4',
                                        record['frames'], record['resolution'])
                    complete_render(record, self.output)
            elif 'inference.py' in command:
                stages.append('infer')
                records = json.loads(Path(command[command.index('--scene_manifest') + 1]).read_text())
                for record in records:
                    self.make_video(record['output_path'], record['valid_frames'], record['resolution'])
                    complete_prediction(record)
            else:
                return run_process(command, **kwargs)
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], 'GPU-test')
            return SimpleNamespace(returncode=0)

        with patch.object(runner, 'DEFAULT_OUTPUT', self.output), \
             patch.object(runner.sys, 'argv', ['runner', '--scene_dir', str(self.scene)]), \
             patch.object(runner, 'ensure_t5_weights'), \
             patch.object(runner, 'inference_settings', return_value='test-settings'), \
             patch.object(runner, 'choose_gpu', return_value='GPU-test') as choose, \
             patch.object(runner.subprocess, 'run', side_effect=worker):
            runner.main()
            self.assertEqual(stages, ['render', 'infer'])
            before = file_digest(self.output / 'scene/pred.mp4')
            runner.main()
            self.assertEqual(stages, ['render', 'infer'])
            self.assertEqual(choose.call_count, 1)
            self.assertEqual(file_digest(self.output / 'scene/pred.mp4'), before)
            (self.scene / 'input/prompt.txt').write_text('changed prompt')
            runner.main()
            self.assertEqual(stages, ['render', 'infer', 'infer'])
            trajectory = self.scene / 'input/target_tcw.txt'
            values = np.loadtxt(trajectory)
            values[0, 3] = .1
            np.savetxt(trajectory, values)
            runner.main()
            self.assertEqual(stages, ['render', 'infer', 'infer', 'render', 'infer'])

            alias = self.root / 'scene-alias'
            alias.symlink_to(self.scene, target_is_directory=True)
            catalog_path = self.output / 'manifest.json'
            catalog = json.loads(catalog_path.read_text())
            next(item for item in catalog if item['id'] == 'scene')['scene_path'] = str(alias)
            catalog_path.write_text(json.dumps(catalog))
            resolve = Path.resolve

            # A second mount of the same directory keeps a distinct resolved path.
            def resolve_alias(path, *args, **kwargs):
                return path if path == alias else resolve(path, *args, **kwargs)

            with patch.object(Path, 'resolve', resolve_alias):
                runner.main()
            self.assertEqual(stages, ['render', 'infer', 'infer', 'render', 'infer'])
        catalog = json.loads((self.output / 'manifest.json').read_text())
        self.assertIn(unrelated, catalog)
        self.assertEqual(next(item for item in catalog if item['id'] == 'scene')['text'], 'changed prompt')
        self.assertEqual(list(self.output.glob('.run_example_*.json')), [])


class DepthTests(SceneFixture):
    def test_first_video_creates_missing_depth_directory_and_roundtrips(self):
        raw = self.root / 'raw'
        raw.mkdir()
        depth = np.linspace(.5, 20, 16 * 16, dtype=np.float32).reshape(16, 16)
        Image.fromarray(depth.view(np.uint8).reshape(16, 16, 4)).save(raw / '0000.png')
        fresh_scene = self.root / 'fresh'
        convert_video(fresh_scene, {'views': 1, 'resolution': [16, 16], 'fps': 24}, raw)
        capture = cv2.VideoCapture(str(fresh_scene / 'depth/depth.mp4'))
        try:
            ok, frame = capture.read()
            self.assertTrue(ok)
        finally:
            capture.release()
        minimum, maximum = np.loadtxt(fresh_scene / 'depth/metadata.txt')
        decoded = decode_video_depth_frame(frame, minimum, maximum)
        np.testing.assert_allclose(decoded, depth, atol=(maximum - minimum) / 65535)

    def test_uint16_depth_roundtrip(self):
        values = np.linspace(0, 100, 64, dtype=np.float32).reshape(8, 8)
        np.testing.assert_allclose(decode_depth(encode_depth(values, 0, 100), 0, 100), values, atol=100 / 65535)

    def test_estimate_mode_runs_even_with_existing_depth(self):
        record = {'kind': 'video', 'id': 'scene', 'path': str(self.scene), 'frames': 1,
                  'video_signature': 'new-video'}
        (self.scene / 'input/video.mp4').touch()
        (self.scene / 'depth/video_signature.txt').write_text('old-video')
        with patch.object(runner, 'video_depth_ready', return_value=True), \
             patch.object(runner, 'validate_video_inputs'), \
             patch.object(runner.subprocess, 'run', side_effect=RuntimeError('estimator-started')):
            with self.assertRaisesRegex(RuntimeError, 'estimator-started'):
                runner.prepare_depth(record, self.output, 'estimate', self.root / 'weights')
        self.assertEqual((self.scene / 'depth/video_signature.txt').read_text(), 'old-video')

    def test_auto_depth_estimation_publishes_complete_scene(self):
        raw_scene = self.root / 'fresh-scene'
        (raw_scene / 'input').mkdir(parents=True)
        self.make_video(raw_scene / 'input/video.mp4', frames=1)
        (raw_scene / 'input/prompt.txt').write_text('new video')
        np.savetxt(raw_scene / 'input/target_tcw.txt', np.eye(4).reshape(1, 16))
        record = dict(kind='video', id='fresh', path=str(raw_scene), frames=1, views=1,
                      fps=24, resolution=[16, 16], video_signature=file_digest(raw_scene / 'input/video.mp4'))

        def estimator(command, **kwargs):
            raw = Path(command[command.index('--output') + 1])
            (raw / 'depth').mkdir()
            depth = np.ones((16, 16), np.float32)
            Image.fromarray(depth.view(np.uint8).reshape(16, 16, 4)).save(raw / 'depth/0000.png')
            np.savetxt(raw / 'intrinsic.txt', np.eye(3))
            np.savetxt(raw / 'extrinsic.txt', np.eye(4)[:3])

        with patch.object(runner, 'validate_video_inputs'), patch.object(runner.subprocess, 'run', side_effect=estimator) as run:
            runner.prepare_depth(record, self.output, 'auto', self.root / 'weights')
            self.assertTrue(runner.video_depth_ready(record))
            runner.prepare_depth(record, self.output, 'auto', self.root / 'weights')
            self.assertEqual(run.call_count, 1)
            runner.prepare_depth(record, self.output, 'estimate', self.root / 'weights')
            self.assertEqual(run.call_count, 2)
        (raw_scene / 'depth/.estimate_in_progress').touch()
        self.assertFalse(runner.video_depth_ready(record))

    def test_default_depth_run_omits_unused_exports_and_point_cloud(self):
        from pipeline.depth_estimation import DepthEstimator, DEFAULT_CONFIG
        image_path = self.root / 'input.png'
        Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(image_path)
        estimator = object.__new__(DepthEstimator)
        estimator.config = {**DEFAULT_CONFIG}
        estimator.device = 'cpu'
        result = SimpleNamespace(processed_images=np.zeros((1, 16, 16, 3), np.uint8),
                                 extrinsics=np.eye(4, dtype=np.float32)[None, :3],
                                 intrinsics=np.eye(3, dtype=np.float32)[None],
                                 depth=np.ones((1, 16, 16), np.float32))
        estimator.model = SimpleNamespace(inference=lambda *args, **kwargs: result)
        destination = self.root / 'estimated'
        with patch.object(estimator, 'depthmap_to_local_points', side_effect=AssertionError('unused point cloud')):
            self.assertTrue(estimator.run({'images': [str(image_path)]}, str(destination)))
        self.assertTrue((destination / 'depth/0000.png').is_file())
        self.assertFalse((destination / 'frames').exists())
        self.assertFalse((destination / 'depth_half').exists())

    def test_existing_mode_never_starts_estimator(self):
        with patch.object(runner, 'video_depth_ready', return_value=False), patch.object(runner.subprocess, 'run') as run:
            with self.assertRaises(FileNotFoundError):
                runner.prepare_depth({'kind': 'video', 'path': str(self.scene)}, self.output, 'existing', self.root)
            run.assert_not_called()


class GPUTests(unittest.TestCase):
    def query(self):
        return patch('pipeline.gpu.subprocess.run', return_value=SimpleNamespace(stdout='0, GPU-aaa, 80000\n1, GPU-bbb, 60000\n'))

    def test_visible_device_order_and_uuid(self):
        with self.query(), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1,0'}):
            self.assertEqual(choose_gpu('0'), 'GPU-bbb')
            self.assertEqual(choose_gpu('1'), 'GPU-aaa')
        with self.query(), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-bbb'}):
            self.assertEqual(choose_gpu(), 'GPU-bbb')

    def test_restricted_auto_selection(self):
        with self.query(), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1,0'}):
            self.assertEqual(choose_gpu(), 'GPU-aaa')
        with self.query(), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1'}):
            self.assertEqual(choose_gpu(), 'GPU-bbb')
            with self.assertRaises(ValueError):
                choose_gpu('1')

    def test_empty_visibility_fails(self):
        with self.query(), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            with self.assertRaises(RuntimeError):
                choose_gpu()


class VideoTests(SceneFixture):
    def test_reader_padding_and_image_source(self):
        path = self.root / 'video.mp4'
        self.make_video(path, 6)
        chunks = list(iter_video_chunks(path, 6, 9, (16, 16)))
        self.assertEqual([len(chunk) for chunk in chunks], [1, 4, 4])
        self.assertTrue(all(chunk.dtype == torch.uint8 for chunk in chunks))
        frames = torch.cat(chunks)
        self.assertTrue(torch.equal(frames[5], frames[8]))
        with patch('datasets.utils.cv2.VideoCapture', wraps=cv2.VideoCapture) as create:
            image = torch.cat(list(iter_video_chunks(path, 9, 9, (16, 16), repeat_first=True)))
            self.assertTrue(torch.equal(image[0], image[-1]))
            self.assertEqual(create.call_count, 1)

    def test_streamed_masks_match_full_conversion(self):
        from pipeline.causal_inference import convert_mask_video
        frames = torch.randint(0, 256, (9, 3, 16, 16), dtype=torch.uint8)
        reference = ((frames[:, :1].float() / 255 > .5).float() * 2 - 1).permute(1, 0, 2, 3)[None]
        streamed = torch.cat(list(iter_mask_latents([frames[:1], frames[1:5], frames[5:9]], 'cpu', torch.float32)), dim=1)
        torch.testing.assert_close(streamed, convert_mask_video(reference), rtol=0, atol=0)

    def test_incomplete_write_preserves_previous_video(self):
        path = self.root / 'video.mp4'
        self.make_video(path)
        before = path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            with video_writer(path, 24, 16, 16) as writer:
                writer.write(bytes(16 * 16 * 3))
                raise RuntimeError('interrupted')
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.root.glob('*.mp4')), [path])

    def test_frame_padding_covers_all_requested_lengths(self):
        for frames in range(1, 1001):
            padded = padded_frame_count(frames)
            self.assertGreaterEqual(padded, frames)
            self.assertLess(padded - frames, 12)
            self.assertEqual((padded + 3) % 12, 0)

    def test_unsupported_config_fails_early(self):
        for config in ({'height': 720}, {'dataset': {'video_size': [256, 256]}}, {'num_frame_per_block': 1}):
            with self.assertRaises(ValueError):
                validate_config(config)
        validate_config({'height': 480, 'width': 832, 'num_frame_per_block': 3})


class RenderTests(SceneFixture):
    def test_multiview_uses_one_projection_per_view_per_frame(self):
        record = self.make_scene(views=4)
        with patch('pipeline.render_scene.project_source', wraps=project_source) as project:
            render_one(record, torch.device('cpu'), self.output)
        self.assertEqual(project.call_count, 4 * record['frames'])
        self.assertTrue(condition_ready(record, self.output))
        meta = json.loads((self.output / 'scene/render_complete.json').read_text())
        self.assertEqual(meta['source_indices'], [0, 1, 2])
        self.assertEqual(video_metadata(self.output / 'scene/source.mp4')[0], 9)
        with patch('pipeline.render_scene.project_source') as project:
            render_one(record, torch.device('cpu'), self.output)
            project.assert_not_called()

    def test_prepared_projection_matches_reference_math(self):
        warper = DepthWarper()
        image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
        depth = np.ones((16, 16), dtype=np.float32)
        source = np.eye(4, dtype=np.float32)
        target = source.copy()
        target[0, 3] = .25
        intrinsic = np.eye(3, dtype=np.float32)
        cached = prepare_source(warper, image, depth, source, intrinsic, 'cpu')
        rgb, _, mask = project_source(warper, cached, torch.from_numpy(target)[None], torch.from_numpy(intrinsic)[None])
        points = warper.compute_transformed_points(torch.from_numpy(depth)[None, None], torch.from_numpy(source)[None],
                                                   torch.from_numpy(target)[None], torch.from_numpy(intrinsic)[None], torch.from_numpy(intrinsic)[None])
        z = points[..., 2, 0]
        flow = (points[..., :2, 0] / z.unsqueeze(-1)).permute(0, 3, 1, 2) - warper.create_grid(1, 16, 16)
        values = torch.cat([cached[0], z[:, None]], dim=1)
        reference, known = warper.bilinear_splatting(values, torch.ones_like(z[:, None]), z, flow, None)
        torch.testing.assert_close(rgb, reference[0, :3].clamp(-1, 1))
        self.assertTrue(torch.equal(mask, known[0, 0].bool()))


if __name__ == '__main__':
    unittest.main()
