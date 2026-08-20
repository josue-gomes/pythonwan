import base64
import os
import sys
import tempfile
from pathlib import Path

import runpod
from huggingface_hub import snapshot_download

WAN_ROOT = os.environ.get("WAN_ROOT", "/opt/wan")
if WAN_ROOT not in sys.path:
    sys.path.insert(0, WAN_ROOT)

VOLUME = "/runpod-volume/wan22" if os.path.isdir("/runpod-volume") else "/models/wan22"
MODEL_REPO = os.environ.get("MODEL_REPO", "Wan-AI/Wan2.2-T2V-A14B")
TASK = os.environ.get("WAN_TASK", "t2v-A14B")
SIZE = os.environ.get("WAN_SIZE", "720*1280")

_pipe = None


def _ckpt_dir() -> str:
    dest = os.path.join(VOLUME, Path(MODEL_REPO).name)
    marker = os.path.join(dest, "config.json")
    if os.path.isfile(marker):
        print(f"Using cached weights {dest}", flush=True)
        return dest
    os.makedirs(dest, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    print(f"Downloading {MODEL_REPO}", flush=True)
    return snapshot_download(repo_id=MODEL_REPO, local_dir=dest, token=token)


def _ensure_pipe():
    global _pipe
    if _pipe is not None:
        return _pipe
    import wan
    from wan.configs import WAN_CONFIGS

    ckpt = _ckpt_dir()
    cfg = WAN_CONFIGS[TASK]
    print("Loading T2V pipeline", flush=True)
    _pipe = wan.WanT2V(
        config=cfg,
        checkpoint_dir=ckpt,
        device_id=0,
        rank=0,
        t5_cpu=True,
        convert_model_dtype=True,
    )
    print("Pipeline ready", flush=True)
    return _pipe


def _video_data_url(path: str) -> str:
    raw = Path(path).read_bytes()
    if len(raw) < 1000:
        raise RuntimeError("empty video")
    return "data:video/mp4;base64," + base64.b64encode(raw).decode("ascii")


def handler(job):
    incoming = job.get("input") or {}
    prompt = str(incoming.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")
    size = str(incoming.get("size") or SIZE).strip() or SIZE
    seed = int(incoming.get("seed") if incoming.get("seed") is not None else -1)
    steps = incoming.get("sample_steps")
    try:
        pipe = _ensure_pipe()
        from wan.configs import SIZE_CONFIGS, WAN_CONFIGS

        cfg = WAN_CONFIGS[TASK]
        video = pipe.generate(
            prompt,
            size=SIZE_CONFIGS[size],
            frame_num=int(incoming.get("frame_num") or cfg.frame_num),
            shift=cfg.sample_shift,
            sample_solver="unipc",
            sampling_steps=int(steps or cfg.sample_steps),
            guide_scale=cfg.sample_guide_scale,
            seed=seed,
            offload_model=True,
        )
        from wan.utils.utils import save_video

        out = os.path.join(tempfile.gettempdir(), f"out-{os.getpid()}.mp4")
        save_video(
            tensor=video[None],
            save_file=out,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        del video
        payload = {"video": _video_data_url(out)}
        try:
            os.remove(out)
        except OSError:
            pass
        return payload
    except Exception as error:
        print(f"generate failed: {error}", flush=True)
        raise RuntimeError("video failed") from error


print("Worker online", flush=True)
runpod.serverless.start({"handler": handler})
