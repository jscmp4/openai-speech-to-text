# Speech to Text

Local web app for transcribing audio/video files using OpenAI's `gpt-4o-transcribe-diarize` model. Supports speaker identification, timestamps, and automatic file chunking for large files.

## Features

- **Speaker diarization** — automatically identifies and labels different speakers
- **Timestamps** — every segment includes start/end times
- **Large file support** — auto-splits files into chunks (24MB / ~21min each)
- **Real-time progress** — SSE-based progress bar shows each processing step
- **Wide format support** — MP3, MP4, MOV, M4A, WAV, WEBM, AVI, MKV, FLAC, OGG, AAC, WMA
- **API key memory** — enter once, saved in browser localStorage
- **REST API** — synchronous `POST /api/transcribe` endpoint for backend integration

## Prerequisites

- **Python 3.9+**
- **FFmpeg** — used for audio extraction and splitting
  - macOS: `brew install ffmpeg`
  - Windows: download from https://ffmpeg.org/download.html
  - Ubuntu: `sudo apt install ffmpeg`
- **OpenAI API key** — get one at https://platform.openai.com/api-keys

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

1. Enter your OpenAI API key (saved automatically for next time)
2. Drag & drop or click to upload an audio/video file
3. Click **Start Transcription**
4. View results in three formats:
   - **Timestamped** — `[00:01:23.45 -> 00:01:30.12]  Speaker 1: ...`
   - **Plain Text** — merged full text
   - **JSON** — raw segments with speaker, text, start, end
5. Copy to clipboard or download as file

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

## How It Works

```
Upload file
    → FFmpeg extracts audio as MP3 (128kbps, 16kHz, mono)
    → Split into chunks if >24MB or >1300 seconds
    → Each chunk → OpenAI gpt-4o-transcribe-diarize API
    → Adjust timestamps by chunk offset
    → Merge adjacent same-speaker segments
    → Stream results back via SSE
```

## Project Structure

```
├── app.py              # Flask backend
├── templates/
│   └── index.html      # Frontend (single-page)
├── requirements.txt    # Python dependencies
├── start.command       # macOS launcher (double-click)
├── start.bat           # Windows launcher (double-click)
└── .gitignore
```

## License

MIT
