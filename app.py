import os
import json
import math
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

from flask import Flask, render_template, request, jsonify, Response
from openai import OpenAI


def _get_cuda_libs_dir():
    """Get the directory where downloaded CUDA DLLs are stored."""
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.dirname(sys.executable), "cuda_libs")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda_libs")


def _has_downloaded_cuda_libs():
    """Check if CUDA DLLs have been downloaded to our local directory."""
    d = _get_cuda_libs_dir()
    if not os.path.isdir(d):
        return False
    return any(f.endswith(".dll") and "cublas" in f.lower() for f in os.listdir(d))


# Register NVIDIA DLL directories on Windows (needed for CUDA libraries from pip packages)
if sys.platform == "win32":
    _dll_dirs = []
    # Check downloaded cuda_libs directory first
    _local_cuda = _get_cuda_libs_dir()
    if os.path.isdir(_local_cuda):
        _dll_dirs.append(_local_cuda)
        os.add_dll_directory(_local_cuda)
    # Then check pip package locations
    for pkg in ["cublas", "cudnn"]:
        dll_dir = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", pkg, "bin")
        if not os.path.isdir(dll_dir):
            dll_dir = os.path.join(os.path.expanduser("~"), "AppData", "Roaming",
                                   "Python", f"Python{sys.version_info.major}{sys.version_info.minor}",
                                   "site-packages", "nvidia", pkg, "bin")
        if os.path.isdir(dll_dir):
            _dll_dirs.append(dll_dir)
            os.add_dll_directory(dll_dir)
    if _dll_dirs:
        os.environ["PATH"] = os.pathsep.join(_dll_dirs) + os.pathsep + os.environ.get("PATH", "")
        print(f"  Added NVIDIA DLL dirs to PATH: {_dll_dirs}")

# Local model support (lazy loaded)
_whisper_models = {}
_model_devices = {}  # model_name -> "cuda" or "cpu"
_cuda_available = None  # None = not checked, True/False after check

# Speaker diarization support (lazy loaded)
_diarization_pipeline = None

# Log file path
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcription_log.jsonl")

# Audio playback: store extracted audio for segment playback
_AUDIO_DIR = os.path.join(tempfile.gettempdir(), "stt_audio")
os.makedirs(_AUDIO_DIR, exist_ok=True)


def write_log(entry: dict):
    """Append a log entry to the JSONL log file."""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  Log write failed: {e}")

LOCAL_MODELS = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo", "distil-large-v3.5"]

# Measured VRAM usage (MB) per model on float16. int8 uses ~60-65% of these values.
MODEL_VRAM_MB = {
    "tiny": 390, "base": 500, "small": 1000, "medium": 2500,
    "large-v3": 4500, "large-v3-turbo": 2500, "distil-large-v3.5": 2400,
}
CORRECTION_MODELS = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"]

# OpenRouter-compatible correction models (sorted cheap to quality)
OPENROUTER_CORRECTION_MODELS = [
    "qwen/qwen3-30b-a3b",
    "qwen/qwen3.5-9b",
    "openai/gpt-4.1-nano",
    "google/gemini-2.0-flash-001",
    "qwen/qwen-2.5-72b-instruct",
    "openai/gpt-4o-mini",
    "deepseek/deepseek-chat-v3-0324",
    "google/gemini-2.5-flash",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1",
]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Cache for OpenRouter models list
_openrouter_models_cache = {"models": [], "fetched_at": 0}


def _make_client(api_key, provider="openai"):
    """Create an OpenAI-compatible client for the given provider."""
    if provider == "openrouter":
        return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    return OpenAI(api_key=api_key)


def check_cuda():
    """Check if CUDA is available for CTranslate2/faster-whisper."""
    global _cuda_available
    if _cuda_available is not None:
        return _cuda_available
    try:
        import ctranslate2
        _cuda_available = bool(ctranslate2.get_supported_compute_types("cuda"))
        if _cuda_available:
            print("  CUDA is available")
    except Exception as e:
        _cuda_available = False
        print(f"  CUDA not available: {e}")
    return _cuda_available


_diarization_token = None  # track which token was used to load pipeline


def get_diarization_pipeline(hf_token):
    """Lazy-load pyannote speaker diarization pipeline."""
    global _diarization_pipeline, _diarization_token
    if _diarization_pipeline is None or hf_token != _diarization_token:
        from pyannote.audio import Pipeline
        _diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=hf_token,
        )
        _diarization_token = hf_token
        # Move to GPU if available
        if check_cuda():
            import torch
            _diarization_pipeline.to(torch.device("cuda"))
            print("  Diarization pipeline loaded on CUDA")
        else:
            print("  Diarization pipeline loaded on CPU")
    return _diarization_pipeline


def _load_audio_waveform(audio_path, sample_rate=16000):
    """Load audio as a waveform dict using ffmpeg, bypassing torchcodec/torchaudio."""
    import torch
    import numpy as np
    result = subprocess.run(
        ["ffmpeg", "-i", audio_path, "-f", "f32le", "-acodec", "pcm_f32le",
         "-ar", str(sample_rate), "-ac", "1", "-"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio decode failed: {result.stderr.decode()[:200]}")
    audio_np = np.frombuffer(result.stdout, dtype=np.float32).copy()
    waveform = torch.from_numpy(audio_np).unsqueeze(0)  # (1, samples)
    return {"waveform": waveform, "sample_rate": sample_rate}


def _extract_diarization_segments(output):
    """Extract (start, end, speaker) tuples from pyannote pipeline output."""
    segments = []
    if hasattr(output, 'exclusive_speaker_diarization'):
        for turn, speaker in output.exclusive_speaker_diarization:
            segments.append((turn.start, turn.end, speaker))
    elif hasattr(output, 'speaker_diarization'):
        for turn, speaker in output.speaker_diarization:
            segments.append((turn.start, turn.end, speaker))
    else:
        # Legacy pyannote 3.x fallback
        for turn, _, speaker in output.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))
    return segments


# Max chunk duration for diarization (seconds). Longer files are processed in chunks.
DIARIZE_CHUNK_SEC = 1200  # 20 minutes


def run_diarization(audio_path, hf_token, num_speakers=None, progress_cb=None):
    """Run speaker diarization on an audio file. Returns list of (start, end, speaker).

    For long audio (>20 min), processes in chunks to avoid memory issues.
    Supports files of any length including 24+ hours.
    progress_cb: optional callable(msg) to report progress.
    """
    import torch
    pipeline = get_diarization_pipeline(hf_token)

    # Load full audio via ffmpeg
    audio_input = _load_audio_waveform(audio_path)
    total_samples = audio_input["waveform"].shape[1]
    sr = audio_input["sample_rate"]
    total_duration = total_samples / sr

    kwargs = {}
    if num_speakers and num_speakers > 0:
        kwargs["num_speakers"] = num_speakers

    # Short audio: process in one go
    if total_duration <= DIARIZE_CHUNK_SEC:
        if progress_cb:
            progress_cb(f"Identifying speakers in {total_duration:.0f}s audio...")
        output = pipeline(audio_input, **kwargs)
        return _extract_diarization_segments(output)

    # Long audio: process in chunks
    n_chunks = math.ceil(total_samples / (DIARIZE_CHUNK_SEC * sr))
    if progress_cb:
        progress_cb(f"Identifying speakers in {total_duration/60:.0f} min audio ({n_chunks} chunks)...")
    print(f"  Diarizing {total_duration:.0f}s audio in {n_chunks} chunks of {DIARIZE_CHUNK_SEC}s")
    chunk_samples = DIARIZE_CHUNK_SEC * sr
    all_segments = []
    pos = 0
    chunk_idx = 0

    while pos < total_samples:
        end = min(pos + chunk_samples, total_samples)
        chunk_waveform = audio_input["waveform"][:, pos:end]
        chunk_input = {"waveform": chunk_waveform, "sample_rate": sr}
        offset_sec = pos / sr
        chunk_idx += 1

        if progress_cb:
            progress_cb(f"Identifying speakers: chunk {chunk_idx}/{n_chunks} ({offset_sec/60:.0f}-{end/sr/60:.0f} min)...")
        print(f"    Chunk {chunk_idx}/{n_chunks}: {offset_sec:.0f}s - {end/sr:.0f}s")
        # Don't constrain num_speakers per chunk — let pyannote auto-detect
        output = pipeline(chunk_input)
        chunk_segs = _extract_diarization_segments(output)

        # Offset timestamps to absolute positions
        for start_t, end_t, speaker in chunk_segs:
            all_segments.append((start_t + offset_sec, end_t + offset_sec, speaker))

        pos = end
        # Free memory
        del chunk_waveform, chunk_input, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_segments
    return segments


def _friendly_speaker_label(raw_label):
    """Convert pyannote labels like SPEAKER_00 to Speaker 1."""
    if raw_label and raw_label.startswith("SPEAKER_"):
        try:
            num = int(raw_label.split("_")[1]) + 1
            return f"Speaker {num}"
        except (IndexError, ValueError):
            pass
    return raw_label


def assign_speakers(transcription_segments, diarization_segments):
    """Assign speaker labels from diarization to transcription segments by time overlap."""
    for seg in transcription_segments:
        seg_mid = (seg["start"] + seg["end"]) / 2
        best_speaker = "SPEAKER_00"
        best_overlap = 0
        for d_start, d_end, d_speaker in diarization_segments:
            # Calculate overlap
            overlap_start = max(seg["start"], d_start)
            overlap_end = min(seg["end"], d_end)
            overlap = max(0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d_speaker
            # Also check if midpoint falls in this diarization segment
            if d_start <= seg_mid <= d_end and overlap >= best_overlap:
                best_speaker = d_speaker
                best_overlap = overlap
        seg["speaker"] = _friendly_speaker_label(best_speaker)
    return transcription_segments


def _find_bundled_model(model_name):
    """Check if a model is bundled with the frozen executable."""
    if not getattr(sys, "frozen", False):
        return None
    models_dir = os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)), "models")
    if not os.path.isdir(models_dir):
        return None
    # Look for directory containing the model name
    for entry in os.listdir(models_dir):
        if model_name in entry.lower():
            candidate = os.path.join(models_dir, entry)
            # Follow symlinks in snapshots directory
            snapshots = os.path.join(candidate, "snapshots")
            if os.path.isdir(snapshots):
                for snap in os.listdir(snapshots):
                    snap_dir = os.path.join(snapshots, snap)
                    if os.path.isdir(snap_dir) and os.path.isfile(os.path.join(snap_dir, "model.bin")):
                        return snap_dir
            if os.path.isfile(os.path.join(candidate, "model.bin")):
                return candidate
    return None


def _get_gpu_free_vram_mb():
    """Get free VRAM in MB using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(float(result.stdout.strip().split("\n")[0]))
    except Exception:
        pass
    return 0


def _pick_cuda_compute_type(model_name):
    """Pick best compute type for GPU based on available VRAM."""
    vram_needed_f16 = MODEL_VRAM_MB.get(model_name, 3000)
    free_vram = _get_gpu_free_vram_mb()
    if free_vram <= 0:
        return "float16"
    # If float16 fits with 500MB headroom, use it; otherwise use int8_float16
    if free_vram >= vram_needed_f16 + 500:
        return "float16"
    # int8_float16 uses ~60% of float16 VRAM
    if free_vram >= int(vram_needed_f16 * 0.6) + 500:
        return "int8_float16"
    return "int8"


def get_whisper_model(model_name, preferred_device=None):
    """Lazy-load a faster-whisper model. Auto-detect CUDA with fallback."""
    cache_key = f"{model_name}_{preferred_device or 'auto'}"
    if cache_key not in _whisper_models:
        from faster_whisper import WhisperModel
        # Use bundled model path if available
        model_path = _find_bundled_model(model_name) or model_name
        if preferred_device == "cpu":
            devices = [("cpu", "int8")]
        elif preferred_device == "cuda":
            ct = _pick_cuda_compute_type(model_name)
            devices = [("cuda", ct)]
        else:
            ct = _pick_cuda_compute_type(model_name)
            devices = [("cuda", ct), ("cpu", "int8")]
        for device, ct in devices:
            try:
                model = WhisperModel(model_path, device=device, compute_type=ct)
                _whisper_models[cache_key] = model
                _model_devices[cache_key] = device
                print(f"  Loaded {model_name} on {device} ({ct})")
                break
            except Exception as e:
                if device == devices[-1][0]:
                    raise
                print(f"  {device} failed ({e}), trying next...")
    return _whisper_models[cache_key], _model_devices[cache_key]

# Support PyInstaller frozen mode
if getattr(sys, "frozen", False):
    _base_dir = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
else:
    _base_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=os.path.join(_base_dir, "templates"))
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB upload limit

ALLOWED_EXTENSIONS = {
    "mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm",
    "mov", "avi", "mkv", "flac", "ogg", "aac", "wma",
}
MAX_CHUNK_SIZE_MB = 24  # stay under 25MB limit
MAX_CHUNK_DURATION = 1300  # stay under 1400s model limit


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_ffmpeg():
    """Find ffmpeg binary (check bundled location first for frozen builds)."""
    if getattr(sys, "frozen", False):
        exe_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
        for base in [getattr(sys, "_MEIPASS", ""), os.path.dirname(sys.executable)]:
            bundled = os.path.join(base, "ffmpeg", exe_name)
            if os.path.isfile(bundled):
                return bundled
    path = shutil.which("ffmpeg")
    if path:
        return path
    raise RuntimeError(
        "ffmpeg not found. Please install ffmpeg: https://ffmpeg.org/download.html"
    )


def get_ffprobe():
    """Find ffprobe binary."""
    if getattr(sys, "frozen", False):
        exe_name = "ffprobe.exe" if sys.platform == "win32" else "ffprobe"
        for base in [getattr(sys, "_MEIPASS", ""), os.path.dirname(sys.executable)]:
            bundled = os.path.join(base, "ffmpeg", exe_name)
            if os.path.isfile(bundled):
                return bundled
    path = shutil.which("ffprobe")
    if path:
        return path
    # Fallback: same directory as ffmpeg
    return get_ffmpeg().replace("ffmpeg", "ffprobe")


def get_duration(file_path):
    """Get audio/video duration in seconds using ffprobe."""
    ffprobe = get_ffprobe()
    result = subprocess.run(
        [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", file_path],
        capture_output=True, text=True,
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def extract_audio(input_path, output_path):
    """Extract audio from video/audio file as mp3."""
    ffmpeg = get_ffmpeg()
    subprocess.run(
        [ffmpeg, "-y", "-i", input_path, "-vn", "-acodec", "libmp3lame",
         "-ab", "128k", "-ar", "16000", "-ac", "1", output_path],
        capture_output=True, check=True,
    )


def split_audio(audio_path, tmp_dir, max_size_mb=MAX_CHUNK_SIZE_MB):
    """Split audio file into chunks under both size and duration limits."""
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    duration = get_duration(audio_path)

    if file_size_mb <= max_size_mb and duration <= MAX_CHUNK_DURATION:
        return [audio_path]

    chunks_by_size = math.ceil(file_size_mb / max_size_mb)
    chunks_by_duration = math.ceil(duration / MAX_CHUNK_DURATION)
    num_chunks = max(chunks_by_size, chunks_by_duration)
    chunk_duration = duration / num_chunks

    ffmpeg = get_ffmpeg()
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_duration
        chunk_path = os.path.join(tmp_dir, f"chunk_{i:03d}.mp3")
        subprocess.run(
            [ffmpeg, "-y", "-i", audio_path,
             "-ss", str(start), "-t", str(chunk_duration),
             "-acodec", "libmp3lame", "-ab", "128k", "-ar", "16000", "-ac", "1",
             chunk_path],
            capture_output=True, check=True,
        )
        chunks.append(chunk_path)
    return chunks


def transcribe_chunk(client, chunk_path, chunk_offset_sec):
    """Send a single chunk to OpenAI for transcription with diarization."""
    with open(chunk_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe-diarize",
            file=f,
            response_format="diarized_json",
            chunking_strategy="auto",
        )

    segments = []
    raw = transcript if isinstance(transcript, dict) else transcript.model_dump()
    for seg in raw.get("segments", []):
        segments.append({
            "speaker": seg.get("speaker", "unknown"),
            "text": seg.get("text", ""),
            "start": round(seg.get("start", 0) + chunk_offset_sec, 2),
            "end": round(seg.get("end", 0) + chunk_offset_sec, 2),
        })
    return segments


def transcribe_local_streaming(model, chunk_path, chunk_offset_sec):
    """Yield segments one by one from a local faster-whisper model (generator)."""
    segments_out, info = model.transcribe(
        chunk_path, beam_size=1, vad_filter=True, word_timestamps=False,
    )
    print(f"  Transcribing: language={info.language}, probability={info.language_probability:.2f}, duration={info.duration:.1f}s")
    for seg in segments_out:
        yield {
            "speaker": "speaker",
            "text": seg.text.strip(),
            "start": round(seg.start + chunk_offset_sec, 2),
            "end": round(seg.end + chunk_offset_sec, 2),
        }


def format_timestamp(seconds):
    """Format seconds to HH:MM:SS.ss"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def merge_segments(all_segments):
    """Merge adjacent segments from the same speaker."""
    if not all_segments:
        return all_segments

    merged = [all_segments[0].copy()]
    for seg in all_segments[1:]:
        last = merged[-1]
        if last["speaker"] == seg["speaker"] and seg["start"] - last["end"] < 0.5:
            last["text"] = last["text"].rstrip() + " " + seg["text"].lstrip()
            last["end"] = seg["end"]
        else:
            merged.append(seg.copy())
    return merged


CORRECTION_BATCH_SIZE = 30  # segments per batch


def correct_segments_batch(client, segments, model, batch_idx, total_batches):
    """Send a batch of segments to GPT for contextual correction. Returns corrected texts."""
    numbered = "\n".join(
        f"{i}: {seg['text']}" for i, seg in enumerate(segments)
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                "You are a transcription corrector. You receive numbered lines of speech-to-text output. "
                "Fix obvious transcription errors, wrong words, homophones, and punctuation based on context. "
                "Keep the original language. Do NOT add, remove, or reorder lines. "
                "Do NOT translate. Do NOT add explanations. "
                "Output ONLY the corrected numbered lines in the exact same format: 'number: corrected text'"
            )},
            {"role": "user", "content": numbered},
        ],
        temperature=0.3,
    )
    corrected = {}
    if not resp.choices or not resp.choices[0].message or not resp.choices[0].message.content:
        print(f"  Correction batch {batch_idx+1}/{total_batches}: empty response, skipping")
        return corrected
    for line in resp.choices[0].message.content.strip().split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        idx_str, text = line.split(":", 1)
        try:
            corrected[int(idx_str.strip())] = text.strip()
        except ValueError:
            continue
    return corrected


def sse_event(event, data):
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/audio/<audio_id>")
def serve_audio(audio_id):
    """Serve extracted audio file for segment playback."""
    # Sanitize: only allow alphanumeric + hyphen
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', audio_id):
        return "Invalid audio ID", 400
    audio_path = os.path.join(_AUDIO_DIR, f"{audio_id}.mp3")
    if not os.path.isfile(audio_path):
        return "Audio not found", 404
    return Response(
        open(audio_path, "rb").read(),
        mimetype="audio/mpeg",
        headers={"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=3600"},
    )


@app.route("/api/local-models")
def list_local_models():
    return jsonify({"models": LOCAL_MODELS})


@app.route("/api/verify-key", methods=["POST"])
def verify_key():
    """Quick check if an API key is valid (supports OpenAI and OpenRouter)."""
    data = request.get_json(silent=True) or {}
    api_key = data.get("api_key", "").strip()
    provider = data.get("provider", "openai").strip()
    if not api_key:
        return jsonify({"valid": False, "error": "No key provided"})
    try:
        client = _make_client(api_key, provider)
        client.models.list()
        return jsonify({"valid": True})
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)})


@app.route("/api/verify-hf-token", methods=["POST"])
def verify_hf_token():
    """Check if a HuggingFace token has access to pyannote/speaker-diarization-community-1."""
    data = request.get_json(silent=True) or {}
    hf_token = data.get("token", "").strip()
    if not hf_token:
        return jsonify({"valid": False, "error": "No token provided"})
    try:
        req = Request(
            "https://huggingface.co/api/models/pyannote/speaker-diarization-community-1",
            headers={"Authorization": f"Bearer {hf_token}"},
        )
        resp = urlopen(req, timeout=10)
        info = json.loads(resp.read().decode())
        # If we can read model info, the token is valid and has access
        return jsonify({"valid": True, "model": info.get("modelId", "pyannote/speaker-diarization-community-1")})
    except URLError as e:
        err_msg = str(e)
        if "403" in err_msg:
            return jsonify({"valid": False, "error": "Token valid but model license not accepted. Click 'Agree' on the model page."})
        if "401" in err_msg:
            return jsonify({"valid": False, "error": "Invalid token"})
        return jsonify({"valid": False, "error": f"Connection error: {err_msg}"})
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)})


@app.route("/api/test-diarization", methods=["POST"])
def test_diarization():
    """Full diagnostic: load pipeline + run on test audio. Returns step-by-step results."""
    data = request.get_json(silent=True) or {}
    hf_token = data.get("token", "").strip()
    if not hf_token:
        return jsonify({"ok": False, "step": "token", "error": "No HuggingFace token provided"})

    steps = []

    # Step 1: Generate a short test audio (2 sec of speech-like noise)
    try:
        tmp = tempfile.mktemp(suffix=".wav")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i",
             "anoisesrc=d=2:c=pink:r=16000:a=0.5",
             "-ar", "16000", "-ac", "1", tmp],
            capture_output=True, timeout=10,
        )
        steps.append({"step": "test_audio", "ok": True, "msg": "Test audio created"})
    except Exception as e:
        return jsonify({"ok": False, "step": "test_audio", "error": str(e), "steps": steps})

    # Step 2: Load audio via ffmpeg
    try:
        audio_input = _load_audio_waveform(tmp)
        shape = list(audio_input["waveform"].shape)
        steps.append({"step": "audio_load", "ok": True, "msg": f"Audio loaded: {shape}, sr={audio_input['sample_rate']}"})
    except Exception as e:
        os.unlink(tmp)
        return jsonify({"ok": False, "step": "audio_load", "error": str(e), "steps": steps})

    # Step 3: Load diarization pipeline
    try:
        pipeline = get_diarization_pipeline(hf_token)
        device = "unknown"
        try:
            device = str(next(pipeline.parameters()).device)
        except Exception:
            pass
        steps.append({"step": "pipeline_load", "ok": True, "msg": f"Pipeline loaded on {device}"})
    except Exception as e:
        os.unlink(tmp)
        err = str(e)
        hint = ""
        if "403" in err or "gated" in err.lower() or "access" in err.lower():
            hint = " — You must accept the model license at https://huggingface.co/pyannote/speaker-diarization-community-1"
        elif "401" in err or "unauthorized" in err.lower():
            hint = " — Invalid HuggingFace token"
        return jsonify({"ok": False, "step": "pipeline_load", "error": err + hint, "steps": steps})

    # Step 4: Run diarization
    try:
        output = pipeline(audio_input)
        # pyannote.audio 4.x: use exclusive_speaker_diarization (one speaker at a time)
        speakers = []
        if hasattr(output, 'exclusive_speaker_diarization'):
            for turn, speaker in output.exclusive_speaker_diarization:
                speakers.append({"start": round(turn.start, 2), "end": round(turn.end, 2), "speaker": speaker})
        elif hasattr(output, 'speaker_diarization'):
            for turn, speaker in output.speaker_diarization:
                speakers.append({"start": round(turn.start, 2), "end": round(turn.end, 2), "speaker": speaker})
        else:
            # Legacy pyannote 3.x fallback
            for turn, _, speaker in output.itertracks(yield_label=True):
                speakers.append({"start": round(turn.start, 2), "end": round(turn.end, 2), "speaker": speaker})
        n = len(set(s["speaker"] for s in speakers))
        if n == 0:
            steps.append({"step": "diarize", "ok": True, "msg": "Pipeline ran OK. 0 speakers in test audio (expected — test audio is noise, not speech)"})
        else:
            steps.append({"step": "diarize", "ok": True, "msg": f"Diarization complete: {n} speaker(s), {len(speakers)} segment(s)", "segments": speakers})
    except Exception as e:
        os.unlink(tmp)
        return jsonify({"ok": False, "step": "diarize", "error": str(e), "steps": steps})

    os.unlink(tmp)
    return jsonify({"ok": True, "steps": steps})


@app.route("/api/status")
def api_status():
    """Return loaded models and their device info, plus CUDA availability."""
    cuda = check_cuda()
    gpu = _detect_nvidia_gpu()
    vram_total = gpu["vram_mb"] if gpu else 0
    vram_free = _get_gpu_free_vram_mb() if cuda else 0
    return jsonify({
        "loaded_models": {name: _model_devices.get(name, "unknown") for name in _whisper_models},
        "available_models": LOCAL_MODELS,
        "cuda_available": cuda,
        "default_device": "cuda" if cuda else "cpu",
        "correction_models": CORRECTION_MODELS,
        "openrouter_correction_models": OPENROUTER_CORRECTION_MODELS,
        "vram_total_mb": vram_total,
        "vram_free_mb": vram_free,
        "model_vram_mb": MODEL_VRAM_MB,
    })


@app.route("/api/logs")
def get_logs():
    """Return recent transcription log entries."""
    try:
        if not os.path.exists(LOG_FILE):
            return jsonify({"logs": []})
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Return last 50 entries, newest first
        entries = []
        for line in reversed(lines[-50:]):
            line = line.strip()
            if line:
                entries.append(json.loads(line))
        return jsonify({"logs": entries})
    except Exception as e:
        return jsonify({"logs": [], "error": str(e)})


@app.route("/api/openrouter-models")
def list_openrouter_models():
    """Fetch available models from OpenRouter API (cached for 10 minutes)."""
    import time
    cache = _openrouter_models_cache
    now = time.time()
    if cache["models"] and now - cache["fetched_at"] < 600:
        return jsonify({"models": cache["models"]})
    try:
        req = Request(
            f"{OPENROUTER_BASE_URL}/models",
            headers={"User-Agent": "SpeechToText/1.0"},
        )
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        models = []
        for m in data.get("data", []):
            mid = m.get("id", "")
            # Skip meta/router models
            if mid.startswith("openrouter/"):
                continue
            name = m.get("name", mid)
            pricing = m.get("pricing", {})
            prompt_price = float(pricing.get("prompt", "0") or "0")
            # Price per million tokens
            price_per_m = max(0, prompt_price * 1_000_000)
            models.append({
                "id": mid,
                "name": name,
                "price_per_m": round(price_per_m, 3),
                "context_length": m.get("context_length", 0),
            })
        # Sort by price (cheapest first), then by name
        models.sort(key=lambda x: (x["price_per_m"], x["name"]))
        cache["models"] = models
        cache["fetched_at"] = now
        return jsonify({"models": models})
    except Exception as e:
        # Fall back to hardcoded list
        return jsonify({"models": [], "error": str(e)})


def _detect_nvidia_gpu():
    """Detect NVIDIA GPU using nvidia-smi. Returns dict with gpu info or None."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split("\n")[0].split(", ")
            return {
                "name": parts[0].strip() if len(parts) > 0 else "Unknown",
                "vram_mb": int(float(parts[1].strip())) if len(parts) > 1 else 0,
                "driver": parts[2].strip() if len(parts) > 2 else "Unknown",
            }
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return None


@app.route("/api/gpu-info")
def gpu_info():
    """Return GPU detection info and CUDA library status."""
    gpu = _detect_nvidia_gpu()
    cuda_libs_installed = _has_downloaded_cuda_libs()
    cuda = check_cuda()
    return jsonify({
        "gpu_detected": gpu is not None,
        "gpu": gpu,
        "cuda_libs_installed": cuda_libs_installed,
        "cuda_available": cuda,
        "platform": sys.platform,
    })


@app.route("/api/install-cuda", methods=["POST"])
def install_cuda():
    """Download CUDA DLLs from PyPI and extract to local cuda_libs directory."""
    if sys.platform != "win32":
        def _linux_error():
            yield sse_event("error", {"error": "Auto-install is Windows only. On Linux, CUDA should already work if nvidia drivers are installed. Run: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 in your venv if needed."})
        return Response(_linux_error(), mimetype="text/event-stream")

    def generate():
        try:
            cuda_dir = _get_cuda_libs_dir()
            os.makedirs(cuda_dir, exist_ok=True)

            # Packages to download from PyPI
            packages = [
                ("nvidia-cublas-cu12", "nvidia/cublas"),
            ]

            for pkg_name, inner_path in packages:
                yield sse_event("progress", {"pct": 5, "msg": f"Fetching {pkg_name} info from PyPI..."})

                # Get package info from PyPI JSON API
                pypi_url = f"https://pypi.org/pypi/{pkg_name}/json"
                req = Request(pypi_url, headers={"User-Agent": "SpeechToText/1.0"})
                with urlopen(req, timeout=15) as resp:
                    pkg_info = json.loads(resp.read().decode())

                # Find the latest win_amd64 wheel
                wheel_url = None
                wheel_size = 0
                for url_info in pkg_info.get("urls", []):
                    fn = url_info.get("filename", "")
                    if fn.endswith(".whl") and "win_amd64" in fn:
                        wheel_url = url_info["url"]
                        wheel_size = url_info.get("size", 0)
                        break

                if not wheel_url:
                    # Try latest version's files
                    version = pkg_info["info"]["version"]
                    for url_info in pkg_info.get("releases", {}).get(version, []):
                        fn = url_info.get("filename", "")
                        if fn.endswith(".whl") and "win_amd64" in fn:
                            wheel_url = url_info["url"]
                            wheel_size = url_info.get("size", 0)
                            break

                if not wheel_url:
                    yield sse_event("error", {"error": f"Could not find Windows wheel for {pkg_name}"})
                    return

                size_mb = wheel_size / (1024 * 1024)
                yield sse_event("progress", {"pct": 10, "msg": f"Downloading {pkg_name} ({size_mb:.0f} MB)..."})

                # Download the wheel to temp
                tmp_wheel = os.path.join(cuda_dir, "_download.whl")
                req = Request(wheel_url, headers={"User-Agent": "SpeechToText/1.0"})
                with urlopen(req, timeout=600) as resp:
                    total = int(resp.headers.get("Content-Length", wheel_size) or wheel_size)
                    downloaded = 0
                    with open(tmp_wheel, "wb") as f:
                        while True:
                            chunk = resp.read(1024 * 256)  # 256KB chunks
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total > 0:
                                pct = 10 + int((downloaded / total) * 70)
                                dl_mb = downloaded / (1024 * 1024)
                                total_mb = total / (1024 * 1024)
                                yield sse_event("progress", {
                                    "pct": pct,
                                    "msg": f"Downloading {pkg_name}... {dl_mb:.0f}/{total_mb:.0f} MB",
                                })

                yield sse_event("progress", {"pct": 85, "msg": f"Extracting DLLs from {pkg_name}..."})

                # Extract DLLs from the wheel (which is a zip file)
                dll_count = 0
                with zipfile.ZipFile(tmp_wheel, "r") as zf:
                    for name in zf.namelist():
                        if name.endswith(".dll"):
                            dll_data = zf.read(name)
                            dll_name = os.path.basename(name)
                            dest = os.path.join(cuda_dir, dll_name)
                            with open(dest, "wb") as f:
                                f.write(dll_data)
                            dll_count += 1
                            print(f"  Extracted: {dll_name} ({len(dll_data) / (1024*1024):.1f} MB)")

                # Clean up temp wheel
                os.remove(tmp_wheel)
                yield sse_event("progress", {"pct": 95, "msg": f"Extracted {dll_count} DLL(s) from {pkg_name}"})

            yield sse_event("progress", {"pct": 100, "msg": "CUDA libraries installed! Please restart the app to enable GPU."})
            yield sse_event("done", {"cuda_dir": cuda_dir, "restart_required": True})

        except Exception as e:
            yield sse_event("error", {"error": str(e)})

    return Response(generate(), mimetype="text/event-stream")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file format. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400

    model_type = request.form.get("model_type", "openai").strip()
    use_local = model_type in LOCAL_MODELS
    preferred_device = request.form.get("device", "auto").strip()
    if preferred_device not in ("cuda", "cpu", "auto"):
        preferred_device = "auto"

    # Provider settings
    provider = request.form.get("provider", "openai").strip()
    if provider not in ("openai", "openrouter"):
        provider = "openai"

    # Correction settings
    correction_enabled = request.form.get("correction", "").strip() == "1"
    correction_model = request.form.get("correction_model", "gpt-4.1-nano").strip()
    correction_provider = request.form.get("correction_provider", provider).strip()
    if correction_provider not in ("openai", "openrouter"):
        correction_provider = provider
    # Accept any model string for OpenRouter (they have thousands of models)
    if correction_provider == "openai" and correction_model not in CORRECTION_MODELS:
        correction_model = "gpt-4.1-nano"

    api_key = request.form.get("api_key", "").strip()
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "")

    correction_api_key = request.form.get("correction_api_key", "").strip() or api_key

    # Diarization settings (local mode only)
    diarize_enabled = request.form.get("diarize", "").strip() == "1"
    hf_token = request.form.get("hf_token", "").strip()
    num_speakers_str = request.form.get("num_speakers", "").strip()
    num_speakers = int(num_speakers_str) if num_speakers_str.isdigit() and int(num_speakers_str) > 0 else None
    if diarize_enabled and use_local and not hf_token:
        return jsonify({"error": "Speaker diarization requires a HuggingFace token. Get one free at huggingface.co/settings/tokens"}), 400

    # For OpenAI transcription mode, need OpenAI key
    needs_transcribe_api = not use_local
    if needs_transcribe_api and not api_key:
        return jsonify({"error": "Please provide an API key for cloud transcription"}), 400
    # For correction, need the correction provider's key
    if correction_enabled and not correction_api_key:
        return jsonify({"error": "Please provide an API key for AI correction"}), 400

    # Save file and prep before streaming
    tmp_dir = tempfile.mkdtemp(prefix="stt_")
    ext = file.filename.rsplit(".", 1)[1].lower()
    input_path = os.path.join(tmp_dir, f"input.{ext}")
    file.save(input_path)

    file_size_mb = round(os.path.getsize(input_path) / (1024 * 1024), 2)
    _job_start = __import__("time").time()

    def generate():
        audio_id = None
        try:
            client = None
            local_model = None
            if use_local:
                device_label = preferred_device if preferred_device != "auto" else "auto-detect"
                yield sse_event("progress", {"step": "load_model", "pct": 3, "msg": f"Loading local model ({model_type}) on {device_label}..."})
                try:
                    local_model, device = get_whisper_model(model_type, preferred_device if preferred_device != "auto" else None)
                except Exception as e:
                    yield sse_event("error", {"error": f"Failed to load model '{model_type}': {e}"})
                    return
                yield sse_event("status", {"device": device, "model": model_type})
            else:
                client = _make_client(api_key, provider)

            # Step 1: Extract audio
            yield sse_event("progress", {"step": "extract", "pct": 5, "msg": "Extracting audio..."})
            audio_path = os.path.join(tmp_dir, "audio.mp3")
            extract_audio(input_path, audio_path)
            # Save a copy for playback (persists after tmp_dir cleanup)
            import uuid as _uuid
            audio_id = str(_uuid.uuid4())[:12]
            os.makedirs(_AUDIO_DIR, exist_ok=True)
            playback_path = os.path.join(_AUDIO_DIR, f"{audio_id}.mp3")
            shutil.copy2(audio_path, playback_path)
            # Send audio_id immediately so frontend can enable playback even if later steps fail
            yield sse_event("audio_ready", {"audio_id": audio_id})

            # Step 1.5: Speaker diarization (local mode only, inline chunked)
            diarization_result = None
            if diarize_enabled and use_local:
                yield sse_event("progress", {"step": "diarize", "pct": 8, "msg": "Loading speaker diarization model..."})
                try:
                    import torch as _torch
                    diar_pipeline = get_diarization_pipeline(hf_token)
                    yield sse_event("progress", {"step": "diarize", "pct": 9, "msg": "Loading audio for diarization..."})
                    diar_audio = _load_audio_waveform(audio_path)
                    total_samples = diar_audio["waveform"].shape[1]
                    sr = diar_audio["sample_rate"]
                    total_dur = total_samples / sr
                    diar_kwargs = {}
                    if num_speakers and num_speakers > 0:
                        diar_kwargs["num_speakers"] = num_speakers

                    diarization_result = []
                    chunk_sec = DIARIZE_CHUNK_SEC
                    chunk_samples = chunk_sec * sr
                    n_chunks = max(1, math.ceil(total_samples / chunk_samples))

                    for ci in range(n_chunks):
                        start_s = ci * chunk_samples
                        end_s = min(start_s + chunk_samples, total_samples)
                        offset_sec = start_s / sr
                        pct = 10 + int((ci / n_chunks) * 8)
                        yield sse_event("progress", {"step": "diarize", "pct": pct, "msg": f"Identifying speakers: chunk {ci+1}/{n_chunks} ({offset_sec/60:.0f}-{end_s/sr/60:.0f} min)..."})

                        chunk_input = {"waveform": diar_audio["waveform"][:, start_s:end_s], "sample_rate": sr}
                        # Use num_speakers only for single-chunk (whole file)
                        kw = diar_kwargs if n_chunks == 1 else {}
                        output = diar_pipeline(chunk_input, **kw)
                        chunk_segs = _extract_diarization_segments(output)
                        print(f"    Diarize chunk {ci+1}/{n_chunks} ({offset_sec:.0f}s-{end_s/sr:.0f}s): {len(chunk_segs)} segments, {len(set(s for _,_,s in chunk_segs))} speakers")

                        for s_start, s_end, spk in chunk_segs:
                            diarization_result.append((s_start + offset_sec, s_end + offset_sec, spk))

                        del chunk_input, output
                        if _torch.cuda.is_available():
                            _torch.cuda.empty_cache()

                    del diar_audio
                    n_speakers = len(set(s for _, _, s in diarization_result))
                    speaker_names = [_friendly_speaker_label(s) for s in sorted(set(s for _, _, s in diarization_result))]
                    yield sse_event("progress", {"step": "diarize", "pct": 18, "msg": f"Found {n_speakers} speaker(s) in {len(diarization_result)} segments: {', '.join(speaker_names)}. Starting transcription..."})
                    print(f"  Diarization total: {len(diarization_result)} segments, {n_speakers} speakers: {speaker_names}")
                except Exception as e:
                    import traceback
                    print(f"  Diarization FAILED: {traceback.format_exc()}")
                    yield sse_event("progress", {"step": "diarize", "pct": 18, "msg": f"Diarization failed: {e}. Continuing without speaker labels..."})

            # Step 2: Split (only needed for OpenAI due to size limits)
            if use_local:
                chunks = [audio_path]
                total = 1
                yield sse_event("progress", {"step": "split_done", "pct": 20, "msg": "Starting transcription..."})
            else:
                yield sse_event("progress", {"step": "split", "pct": 15, "msg": "Splitting into chunks..."})
                chunks = split_audio(audio_path, tmp_dir)
                total = len(chunks)
                yield sse_event("progress", {"step": "split_done", "pct": 20, "msg": f"Split into {total} chunk(s). Starting transcription..."})

            # Step 3: Transcribe
            all_segments = []
            offset = 0.0

            if use_local:
                # Stream segments in real-time for local models
                audio_duration = get_duration(audio_path)
                yield sse_event("progress", {"step": "transcribe", "pct": 20, "msg": "Transcribing..."})
                seg_count = 0
                for seg in transcribe_local_streaming(local_model, audio_path, 0.0):
                    all_segments.append(seg)
                    seg_count += 1
                    # Estimate progress based on timestamp vs total duration
                    pct = min(90, 20 + int((seg["end"] / max(audio_duration, 1)) * 70))
                    ts = f"[{format_timestamp(seg['start'])} -> {format_timestamp(seg['end'])}]"
                    line = f"{ts}  {seg['speaker']}: {seg['text']}"
                    yield sse_event("segment", {
                        "segment": seg,
                        "line": line,
                        "pct": pct,
                        "elapsed": format_timestamp(seg["end"]),
                        "total": format_timestamp(audio_duration),
                    })

                # Assign speaker labels from diarization
                print(f"  Diarization result: {len(diarization_result) if diarization_result else 0} segments, Transcription: {len(all_segments)} segments")
                if diarization_result and all_segments:
                    yield sse_event("progress", {"step": "assign_speakers", "pct": 91, "msg": "Assigning speaker labels..."})
                    all_segments = assign_speakers(all_segments, diarization_result)
                    print(f"  Speakers assigned: {set(s['speaker'] for s in all_segments)}")
                    # Re-emit segments with correct speaker labels so frontend can update live preview
                    updated_lines = []
                    for seg in all_segments:
                        ts = f"[{format_timestamp(seg['start'])} -> {format_timestamp(seg['end'])}]"
                        updated_lines.append(f"{ts}  {seg['speaker']}: {seg['text']}")
                    yield sse_event("speakers_assigned", {
                        "segments": all_segments,
                        "lines": updated_lines,
                    })
            else:
                # OpenAI: chunk-based, stream segments as each chunk completes
                audio_duration = get_duration(audio_path)
                for i, chunk_path in enumerate(chunks):
                    chunk_duration = get_duration(chunk_path)
                    pct = 20 + int((i / total) * 70)
                    yield sse_event("progress", {
                        "step": "transcribe",
                        "pct": pct,
                        "msg": f"Transcribing chunk {i + 1}/{total}...",
                        "chunk": i + 1,
                        "total": total,
                    })
                    segments = transcribe_chunk(client, chunk_path, offset)
                    all_segments.extend(segments)
                    # Emit each segment from this chunk immediately
                    done_pct = 20 + int(((i + 1) / total) * 70)
                    for seg in segments:
                        ts = f"[{format_timestamp(seg['start'])} -> {format_timestamp(seg['end'])}]"
                        line = f"{ts}  {seg['speaker']}: {seg['text']}"
                        yield sse_event("segment", {
                            "segment": seg,
                            "line": line,
                            "pct": done_pct,
                            "elapsed": format_timestamp(seg["end"]),
                            "total": format_timestamp(audio_duration),
                        })
                    offset += chunk_duration

            # Step 4: Merge
            merge_pct = 85 if correction_enabled else 92
            yield sse_event("progress", {"step": "merge", "pct": merge_pct, "msg": "Merging results..."})
            merged = merge_segments(all_segments)

            # Step 5: AI Correction (optional, errors don't abort transcription)
            if correction_enabled and merged:
                try:
                    correction_client = _make_client(correction_api_key, correction_provider)
                    total_segs = len(merged)
                    num_batches = math.ceil(total_segs / CORRECTION_BATCH_SIZE)
                    yield sse_event("progress", {
                        "step": "correct", "pct": 87,
                        "msg": f"AI correcting with {correction_model} (0/{num_batches} batches)...",
                    })
                    for b in range(num_batches):
                        start_i = b * CORRECTION_BATCH_SIZE
                        end_i = min(start_i + CORRECTION_BATCH_SIZE, total_segs)
                        batch = merged[start_i:end_i]
                        corrected = correct_segments_batch(
                            correction_client, batch, correction_model, b, num_batches
                        )
                        for local_idx, new_text in corrected.items():
                            global_idx = start_i + local_idx
                            if 0 <= global_idx < total_segs:
                                merged[global_idx]["text"] = new_text
                        pct = 87 + int(((b + 1) / num_batches) * 10)
                        yield sse_event("progress", {
                            "step": "correct", "pct": pct,
                            "msg": f"AI correcting with {correction_model} ({b + 1}/{num_batches} batches)...",
                        })
                except Exception as corr_err:
                    print(f"  Correction error (non-fatal): {corr_err}")
                    yield sse_event("progress", {
                        "step": "correct", "pct": 97,
                        "msg": f"Correction failed: {corr_err}. Using uncorrected transcription.",
                    })

            # Build output
            lines = []
            full_text = []
            for seg in merged:
                ts = f"[{format_timestamp(seg['start'])} -> {format_timestamp(seg['end'])}]"
                speaker = seg["speaker"]
                text = seg["text"].strip()
                lines.append(f"{ts}  {speaker}: {text}")
                full_text.append(text)

            elapsed = round(__import__("time").time() - _job_start, 1)
            audio_dur = round(get_duration(audio_path), 1) if os.path.exists(audio_path) else 0
            write_log({
                "ts": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file": file.filename,
                "size_mb": file_size_mb,
                "audio_sec": audio_dur,
                "model": model_type,
                "provider": provider if not use_local else "local",
                "correction": correction_model if correction_enabled else None,
                "diarize": diarize_enabled and use_local,
                "elapsed_sec": elapsed,
                "status": "ok",
            })

            yield sse_event("progress", {"step": "done", "pct": 100, "msg": "Done!"})
            yield sse_event("result", {
                "success": True,
                "formatted": "\n\n".join(lines),
                "full_text": " ".join(full_text),
                "segments": merged,
                "chunk_count": total if not use_local else 1,
                "audio_id": audio_id,
            })

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"  TRANSCRIPTION ERROR: {tb}")
            write_log({
                "ts": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file": file.filename,
                "size_mb": file_size_mb,
                "audio_sec": 0,
                "model": model_type,
                "provider": provider if not use_local else "local",
                "correction": None,
                "diarize": False,
                "elapsed_sec": round(__import__("time").time() - _job_start, 1),
                "status": f"error: {e}",
            })
            yield sse_event("error", {"error": f"{e}\n\nTraceback:\n{tb}"})
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return Response(generate(), mimetype="text/event-stream")


def process_transcription(file_obj, api_key):
    """Shared transcription logic. Returns result dict or raises."""
    tmp_dir = tempfile.mkdtemp(prefix="stt_")
    try:
        ext = file_obj.filename.rsplit(".", 1)[1].lower()
        input_path = os.path.join(tmp_dir, f"input.{ext}")
        file_obj.save(input_path)

        client = OpenAI(api_key=api_key)

        audio_path = os.path.join(tmp_dir, "audio.mp3")
        extract_audio(input_path, audio_path)

        total_duration = get_duration(audio_path)
        chunks = split_audio(audio_path, tmp_dir)

        all_segments = []
        offset = 0.0
        for chunk_path in chunks:
            chunk_duration = get_duration(chunk_path)
            segments = transcribe_chunk(client, chunk_path, offset)
            all_segments.extend(segments)
            offset += chunk_duration

        merged = merge_segments(all_segments)

        lines = []
        full_text = []
        for seg in merged:
            ts = f"[{format_timestamp(seg['start'])} -> {format_timestamp(seg['end'])}]"
            speaker = seg["speaker"]
            text = seg["text"].strip()
            lines.append(f"{ts}  {speaker}: {text}")
            full_text.append(text)

        return {
            "success": True,
            "formatted_text": "\n\n".join(lines),
            "full_text": " ".join(full_text),
            "segments": merged,
            "duration_seconds": round(total_duration, 2),
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"success": False, "error": f"Unsupported format. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400

    # API key from Authorization header
    auth = request.headers.get("Authorization", "")
    api_key = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else ""
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return jsonify({"success": False, "error": "Missing Authorization header. Use: Authorization: Bearer sk-..."}), 401

    try:
        result = process_transcription(file, api_key)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def cleanup_temp_files():
    """Clean up any leftover stt_ temp directories on startup and exit."""
    import glob
    tmp_root = tempfile.gettempdir()
    for d in glob.glob(os.path.join(tmp_root, "stt_*")):
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass


def _kill_port(port):
    """Kill any process occupying the given port (Windows only)."""
    if sys.platform != "win32":
        return
    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            # Match lines like "  TCP    127.0.0.1:8080    ...    LISTENING    12345"
            parts = line.split()
            if len(parts) >= 5 and f":{port}" in parts[1] and parts[3] == "LISTENING":
                pid = int(parts[4])
                if pid == os.getpid():
                    continue
                print(f"  Killing old process on port {port} (PID {pid})...")
                subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                               capture_output=True, timeout=5)
    except Exception as e:
        print(f"  Port cleanup note: {e}")


if __name__ == "__main__":
    import signal
    import atexit

    port = int(os.environ.get("PORT", 8080))

    # Kill any old process still holding the port
    _kill_port(port)

    # Clean up leftover temp files from previous runs
    cleanup_temp_files()
    # Also clean up on exit
    atexit.register(cleanup_temp_files)

    # Handle Ctrl+C and window close gracefully
    def handle_exit(signum, frame):
        print("\n  Shutting down...")
        cleanup_temp_files()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    host = os.environ.get("HOST", "0.0.0.0")
    print(f"\n  Speech-to-Text is running at: http://{host}:{port}\n")
    app.run(host=host, port=port, debug=False)
