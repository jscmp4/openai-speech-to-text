import os
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from flask import Flask, render_template, request, jsonify, Response
from openai import OpenAI

app = Flask(__name__)
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
    """Find ffmpeg binary."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    raise RuntimeError(
        "ffmpeg not found. Please install ffmpeg: https://ffmpeg.org/download.html"
    )


def get_duration(file_path):
    """Get audio/video duration in seconds using ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        ffprobe = get_ffmpeg().replace("ffmpeg", "ffprobe")
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


def sse_event(event, data):
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file format. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400

    api_key = request.form.get("api_key", "").strip()
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return jsonify({"error": "Please provide an OpenAI API key"}), 400

    # Save file and prep before streaming
    tmp_dir = tempfile.mkdtemp(prefix="stt_")
    ext = file.filename.rsplit(".", 1)[1].lower()
    input_path = os.path.join(tmp_dir, f"input.{ext}")
    file.save(input_path)

    def generate():
        try:
            client = OpenAI(api_key=api_key)

            # Step 1: Extract audio
            yield sse_event("progress", {"step": "extract", "pct": 5, "msg": "Extracting audio..."})
            audio_path = os.path.join(tmp_dir, "audio.mp3")
            extract_audio(input_path, audio_path)

            # Step 2: Split
            yield sse_event("progress", {"step": "split", "pct": 15, "msg": "Splitting into chunks..."})
            chunks = split_audio(audio_path, tmp_dir)
            total = len(chunks)
            yield sse_event("progress", {"step": "split_done", "pct": 20, "msg": f"Split into {total} chunk(s). Starting transcription..."})

            # Step 3: Transcribe each chunk
            all_segments = []
            offset = 0.0
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
                offset += chunk_duration

            # Step 4: Merge
            yield sse_event("progress", {"step": "merge", "pct": 92, "msg": "Merging results..."})
            merged = merge_segments(all_segments)

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
                "chunk_count": total,
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"\n  Speech-to-Text is running at: http://localhost:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False)
