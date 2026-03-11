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

LOCAL_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
CORRECTION_MODELS = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"]


def check_cuda():
    """Check if CUDA is available for CTranslate2/faster-whisper."""
    global _cuda_available
    if _cuda_available is not None:
        return _cuda_available
    try:
        import ctranslate2
        _cuda_available = "cuda" in ctranslate2.get_supported_compute_types("cuda")
        if _cuda_available:
            print("  CUDA is available")
    except Exception as e:
        _cuda_available = False
        print(f"  CUDA not available: {e}")
    return _cuda_available


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
            devices = [("cuda", "float16")]
        else:
            devices = [("cuda", "float16"), ("cpu", "int8")]
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


@app.route("/api/local-models")
def list_local_models():
    return jsonify({"models": LOCAL_MODELS})


@app.route("/api/verify-key", methods=["POST"])
def verify_key():
    """Quick check if an OpenAI API key is valid."""
    data = request.get_json(silent=True) or {}
    api_key = data.get("api_key", "").strip()
    if not api_key:
        return jsonify({"valid": False, "error": "No key provided"})
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        return jsonify({"valid": True})
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)})


@app.route("/api/status")
def api_status():
    """Return loaded models and their device info, plus CUDA availability."""
    cuda = check_cuda()
    return jsonify({
        "loaded_models": {name: _model_devices.get(name, "unknown") for name in _whisper_models},
        "available_models": LOCAL_MODELS,
        "cuda_available": cuda,
        "default_device": "cuda" if cuda else "cpu",
    })


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
    })


@app.route("/api/install-cuda", methods=["POST"])
def install_cuda():
    """Download CUDA DLLs from PyPI and extract to local cuda_libs directory."""
    if sys.platform != "win32":
        return jsonify({"error": "GPU auto-install only supported on Windows"}), 400

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

    # Correction settings
    correction_enabled = request.form.get("correction", "").strip() == "1"
    correction_model = request.form.get("correction_model", "gpt-4.1-nano").strip()
    if correction_model not in CORRECTION_MODELS:
        correction_model = "gpt-4.1-nano"

    api_key = request.form.get("api_key", "").strip()
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "")

    needs_api = not use_local or correction_enabled
    if needs_api and not api_key:
        return jsonify({"error": "Please provide an OpenAI API key"}), 400

    # Save file and prep before streaming
    tmp_dir = tempfile.mkdtemp(prefix="stt_")
    ext = file.filename.rsplit(".", 1)[1].lower()
    input_path = os.path.join(tmp_dir, f"input.{ext}")
    file.save(input_path)

    def generate():
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
                client = OpenAI(api_key=api_key)

            # Step 1: Extract audio
            yield sse_event("progress", {"step": "extract", "pct": 5, "msg": "Extracting audio..."})
            audio_path = os.path.join(tmp_dir, "audio.mp3")
            extract_audio(input_path, audio_path)

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

            # Step 5: AI Correction (optional)
            if correction_enabled and merged:
                correction_client = client if client else OpenAI(api_key=api_key)
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

            # Build output
            lines = []
            full_text = []
            for seg in merged:
                ts = f"[{format_timestamp(seg['start'])} -> {format_timestamp(seg['end'])}]"
                speaker = seg["speaker"]
                text = seg["text"].strip()
                lines.append(f"{ts}  {speaker}: {text}")
                full_text.append(text)

            yield sse_event("progress", {"step": "done", "pct": 100, "msg": "Done!"})
            yield sse_event("result", {
                "success": True,
                "formatted": "\n\n".join(lines),
                "full_text": " ".join(full_text),
                "segments": merged,
                "chunk_count": total if not use_local else 1,
            })

        except Exception as e:
            yield sse_event("error", {"error": str(e)})
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


if __name__ == "__main__":
    import signal
    import atexit

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

    port = int(os.environ.get("PORT", 8080))
    print(f"\n  Speech-to-Text is running at: http://localhost:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False)
