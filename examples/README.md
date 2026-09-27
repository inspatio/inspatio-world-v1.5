# Examples

Run `bash run_example.sh` from the repository root to run the six scenes in
`manifest.json`. Results are saved to `output/<output_id>/`.

## Image and multi-image inputs

Copy an `image_example_*` or `multiview_example_*` directory. Replace the files
below and update its `scene.json` with a unique `output_id`, frame counts
(`frames` and `valid_frames`), and `fps`. Inputs must be 832×480. For multi-image
scenes, start with the example's `frames: 141`.

```bash
bash run_inference.sh --scene_dir /path/to/my_scene
```

| File | Single-image | Multi-image |
|---|---|---|
| RGB | `input/view_00.png` | `input/view_00.png` … `view_03.png` |
| Depth | `depth/depth.png` | `depth/depth_00.png` … `depth_03.png` |
| Prompt | `input/prompt.txt` | same |
| Source intrinsics | `depth/source_intrinsics.txt` | same |
| Source poses | `depth/source_tcw.txt` | same |
| Target poses | `input/target_tcw.txt` | same |
| Depth range | `depth/metadata.txt` | same |

Depth PNGs are single-channel uint16. `metadata.txt` contains the minimum and
maximum depth in metres: `depth = min + uint16 / 65535 × (max - min)`.
Camera matrices use row-major order: 9 numbers per source view for 3×3 intrinsics,
16 per source view or target frame for OpenCV 4×4 world-to-camera poses.

## Video inputs

```bash
bash run_inference.sh --video /path/to/input.mp4 \
  --prompt "Your prompt" --target_traj /path/to/target_tcw.txt
```

The runner resizes the video to 832×480 and estimates depth and source cameras.
Supply one target pose per frame using the matrix format above. The target
trajectory must use the same first-frame-normalized coordinate system and
displacement scale as the estimated source cameras.
