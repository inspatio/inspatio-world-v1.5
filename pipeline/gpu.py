"""Choose one GPU from the devices permitted by the caller."""

import os
import subprocess


def visible_gpus():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.free",
                             "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True)
    devices = []
    for line in result.stdout.splitlines():
        index, uuid, free = (part.strip() for part in line.split(","))
        devices.append({"index": int(index), "uuid": uuid, "free": int(free)})
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return devices
    allowed = []
    for token in visible.split(","):
        token = token.strip()
        if not token or token == "-1":
            break
        matches = [gpu for gpu in devices if str(gpu["index"]) == token or gpu["uuid"].startswith(token)]
        if len(matches) != 1 or matches[0] in allowed:
            raise ValueError(f"Unsupported or ambiguous CUDA_VISIBLE_DEVICES entry: {token}")
        allowed.append(matches[0])
    return allowed


def choose_gpu(requested="auto"):
    devices = visible_gpus()
    if not devices:
        raise RuntimeError("No GPU is visible; check CUDA_VISIBLE_DEVICES")
    if requested == "auto":
        return max(devices, key=lambda item: item["free"])["uuid"]
    try:
        index = int(requested)
    except ValueError:
        candidates = [gpu for gpu in devices if gpu["uuid"].startswith(requested)]
    else:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            candidates = [devices[index]] if 0 <= index < len(devices) else []
        else:
            candidates = [gpu for gpu in devices if gpu["index"] == index]
    if len(candidates) != 1:
        raise ValueError(f"GPU {requested} is not uniquely available within CUDA_VISIBLE_DEVICES")
    return candidates[0]["uuid"]
