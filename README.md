# Speech to Text

Local web app for transcribing audio/video files using **OpenAI API** or **local Whisper models**. Supports speaker diarization, real-time streaming output, AI-powered correction, batch processing, and automatic GPU acceleration.

![Main Interface](screenshot/main%20full.png)

![Speaker Diarization](screenshot/diarize%20sample.png)

## Features

- **Dual engine** — OpenAI API (`gpt-4o-transcribe-diarize`) or local models via [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- **Local models** — tiny, base, small, medium, large-v3, large-v3-turbo, distil-large-v3.5 (no API key needed)
- **GPU acceleration** — auto-detects CUDA, falls back to CPU. Manual GPU/CPU selection in UI
- **Real-time streaming** — segments appear as they're transcribed (SSE-based)
- **Speaker diarization** — identifies who is speaking using [pyannote](https://github.com/pyannote/pyannote-audio), works with local models. Cross-chunk speaker consistency via embedding matching
- **Speaker identification panel** — listen to each speaker's longest segments and rename them before viewing results
- **AI correction** — optional post-processing with OpenAI or OpenRouter LLMs to fix transcription errors
- **Batch processing** — upload and process multiple files sequentially
- **Segment playback** — click any segment to play that portion of audio
- **Timestamps** — every segment includes start/end times
- **Large file support** — auto-splits files into chunks, handles 24+ hour recordings
- **Wide format support** — MP3, MP4, MOV, M4A, WAV, WEBM, AVI, MKV, FLAC, OGG, AAC, WMA
- **Server log console** — built-in web debug console for remote troubleshooting
- **Settings persistence** — model, device, API key, correction preferences saved in localStorage

## Prerequisites

- **Python 3.9+**
- **FFmpeg** — used for audio extraction and splitting
  - macOS: `brew install ffmpeg`
  - Windows: download from https://ffmpeg.org/download.html
  - Ubuntu: `sudo apt install ffmpeg`
- **OpenAI API key** (optional) — needed for OpenAI transcription mode and AI correction. Get one at https://platform.openai.com/api-keys
- **HuggingFace token** (optional) — needed for speaker diarization. Free at https://huggingface.co/settings/tokens

### GPU Support (Optional)

For CUDA acceleration with local models:
- **NVIDIA GPU** with CUDA support
- The `nvidia-cublas-cu12` pip package is installed automatically and provides the required CUDA libraries
- No separate CUDA Toolkit installation needed

## Quick Start

### macOS

```bash
git clone https://github.com/jscmp4/openai-speech-to-text.git
cd openai-speech-to-text
```

Double-click `start.command` in Finder — it will:
1. Create a Python virtual environment
2. Install dependencies
3. Open your browser at `http://localhost:8080`

### Windows

```cmd
git clone https://github.com/jscmp4/openai-speech-to-text.git
cd openai-speech-to-text
```

Double-click `start.bat` — same automatic setup.

### Manual

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:8080` in your browser.

## Usage

1. **Select model** — choose OpenAI API or a local model (tiny -> large-v3)
2. **Select device** (local models) — Auto, GPU (CUDA), or CPU
3. **Enter API key** (if using OpenAI or AI correction) — verified automatically
4. **Enable speaker diarization** (optional) — requires HuggingFace token, first run downloads ~2GB model
5. **Enable AI correction** (optional) — select a correction model and provider (OpenAI or OpenRouter)
6. **Upload files** — drag & drop or click to browse, supports multiple files
7. **Click Start Transcription** — watch real-time progress and streaming results
8. **Identify speakers** — when diarization is enabled, a panel appears with Listen buttons to hear each speaker and rename them
9. **View results** in three formats:
   - **Timestamped** — `[00:01:23.45 -> 00:01:30.12]  Speaker 1: ...`
   - **Plain Text** — merged full text
   - **JSON** — raw segments with speaker, text, start, end
10. **Copy** to clipboard or **download** as file (named after your original file)

## REST API

Synchronous JSON endpoint for backend-to-backend integration (e.g. from Node.js).

### `POST /api/transcribe`

**Headers:**
```
Authorization: Bearer sk-your-openai-api-key
Content-Type: multipart/form-data
```

**Body:** multipart form with a `file` field.

**Example (curl):**
```bash
curl -X POST http://localhost:8080/api/transcribe \
  -H "Authorization: Bearer sk-..." \
  -F "file=@meeting.mp4"
```

**Example (Node.js):**
```js
const form = new FormData();
form.append('file', fs.createReadStream('meeting.mp4'));

const res = await fetch('http://localhost:8080/api/transcribe', {
  method: 'POST',
  headers: { 'Authorization': 'Bearer sk-...' },
  body: form,
});
const data = await res.json();
```

**Response:**
```json
{
  "success": true,
  "formatted_text": "[00:00:00.00 -> 00:00:05.20]  Speaker 1: Hello...",
  "full_text": "Hello...",
  "segments": [
    {"speaker": "Speaker 1", "text": "Hello...", "start": 0.0, "end": 5.2}
  ],
  "duration_seconds": 125.3
}
```

**Error response:**
```json
{
  "success": false,
  "error": "error message"
}
```

> Note: This is a synchronous endpoint — the request blocks until processing completes. For long files, set your HTTP client timeout to at least 30 minutes.

### Other API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/local-models` | GET | List available local models |
| `/api/status` | GET | Loaded models, device info, CUDA availability |
| `/api/verify-key` | POST | Verify an OpenAI API key |
| `/api/verify-hf-token` | POST | Verify a HuggingFace token |
| `/api/test-diarization` | POST | Test diarization pipeline |
| `/api/gpu-info` | GET | Detect NVIDIA GPU info |
| `/api/logs` | GET | Transcription history |

## How It Works

```
Upload file(s)
    -> FFmpeg extracts audio as MP3 (128kbps, 16kHz, mono)
    -> (Optional) Speaker diarization via pyannote (chunked for long audio)
    -> Split into chunks if >24MB or >1300 seconds (OpenAI mode)
    -> Transcribe via OpenAI API or local faster-whisper model
    -> Stream segments in real-time via SSE
    -> Assign speaker labels from diarization by time overlap
    -> Merge adjacent same-speaker segments
    -> (Optional) AI correction via LLM in batches of 30 segments
    -> Display results with timestamps and speaker labels
```

## Project Structure

```
├── app.py              # Flask backend
├── templates/
│   └── index.html      # Frontend (single-page)
├── screenshot/         # Screenshots for README
├── requirements.txt    # Python dependencies
├── start.command       # macOS launcher (double-click)
├── start.bat           # Windows launcher (double-click)
└── .gitignore
```

## License

MIT
