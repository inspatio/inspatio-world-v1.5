"""Shared shape contract for the released inference checkpoint."""

RESOLUTION = (480, 832)
FRAMES_PER_BLOCK = 3
LATENT_CHANNELS = 16
LATENT_HEIGHT, LATENT_WIDTH = (value // 8 for value in RESOLUTION)
TOKENS_PER_FRAME = LATENT_HEIGHT * LATENT_WIDTH // 4


def validate_config(config):
    size = tuple(config.get("dataset", {}).get("video_size", RESOLUTION))
    if size != RESOLUTION or (config.get("height", 480), config.get("width", 832)) != RESOLUTION:
        raise ValueError("The released checkpoint supports resolution [480, 832] only")
    if config.get("num_frame_per_block", FRAMES_PER_BLOCK) != FRAMES_PER_BLOCK:
        raise ValueError("The released checkpoint requires num_frame_per_block=3")
    if config.get("generator", {}).get("in_dim", 36) != 36:
        raise ValueError("Scene inference requires generator.in_dim=36")


def source_type_for(meta):
    if meta["kind"] == "video":
        return "video"
    if meta["kind"] == "image" and meta["views"] in (1, 4):
        return "image" if meta["views"] == 1 else "multi-image"
    raise ValueError(f"Unsupported source type: {meta['kind']} with {meta['views']} views")


def latent_frame_groups(frames):
    """Wan VAE encodes frame zero alone, followed by groups of four."""
    yield range(1)
    for start in range(1, frames, 4):
        yield range(start, min(start + 4, frames))


def padded_frame_count(frames):
    if frames < 1:
        raise ValueError("A scene must contain at least one valid frame")
    latent_frames = (frames - 1 + 3) // 4 + 1
    latent_frames = ((latent_frames + FRAMES_PER_BLOCK - 1) // FRAMES_PER_BLOCK) * FRAMES_PER_BLOCK
    return (latent_frames - 1) * 4 + 1
