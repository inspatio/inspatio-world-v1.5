"""Scene metadata and bounded streams of rendered condition frames."""

import json
from pathlib import Path

from datasets.utils import iter_video_chunks
from pipeline.scene_schema import padded_frame_count


class SceneDataset:
    def __init__(self, manifest_path, video_size=(480, 832)):
        self.records = json.loads(Path(manifest_path).read_text())
        self.video_size = tuple(video_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def frames(self, record, name):
        valid = int(record["valid_frames"])
        return iter_video_chunks(record[f"{name}_video"], valid, padded_frame_count(valid),
                                 self.video_size, repeat_first=(name == "source" and record["source_type"] == "image"))
