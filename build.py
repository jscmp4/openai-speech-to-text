"""
Build script to package Speech-to-Text as a standalone executable.

Usage:
    python build.py cpu      # CPU-only build (~150MB)
    python build.py gpu      # GPU build with CUDA (~1.8GB)
    python build.py          # Defaults to GPU if CUDA available, else CPU

Models are NOT bundled — they download automatically on first use.
"""

import os
import sys
import shutil
import subprocess

VENV_PYTHON = os.path.join("venv", "Scripts", "python.exe")
APP_NAME = "SpeechToText"
DIST_DIR = os.path.join("dist", APP_NAME)


def find_ffmpeg():
    """Find ffmpeg and its DLLs."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg:
        print("  WARNING: ffmpeg not found, won't be bundled")
        return []
    ffmpeg_dir = os.path.dirname(ffmpeg)
    files = [ffmpeg]
    if ffprobe:
        files.append(ffprobe)
    # Include DLLs from same directory
    files += [os.path.join(ffmpeg_dir, f) for f in os.listdir(ffmpeg_dir) if f.endswith(".dll")]
    print(f"  Bundling ffmpeg from: {ffmpeg_dir} ({len(files)} files)")
    return files


def build(mode="gpu"):
    """Build the executable."""
    print(f"\n=== Building {APP_NAME} ({mode.upper()}) ===\n")

    if not os.path.isfile(VENV_PYTHON):
        print("Error: venv not found. Run start.bat first to create it.")
        sys.exit(1)

    # Install pyinstaller if needed
    subprocess.run([VENV_PYTHON, "-m", "pip", "install", "-q", "pyinstaller"], check=True)

    # Build PyInstaller command
    cmd = [
        VENV_PYTHON, "-m", "PyInstaller",
        "--noconfirm",
        "--name", APP_NAME,
        "--console",
        "--distpath", "dist",
        "--add-data", "templates;templates",
    ]

    # Add ffmpeg
    for f in find_ffmpeg():
        cmd += ["--add-data", f"{f};ffmpeg"]

    # GPU mode: add CUDA DLLs
    if mode == "gpu":
        site_pkgs = os.path.join("venv", "Lib", "site-packages")
        for pkg in ["cublas", "cudnn"]:
            dll_dir = os.path.join(site_pkgs, "nvidia", pkg, "bin")
            if os.path.isdir(dll_dir):
                dlls = [f for f in os.listdir(dll_dir) if f.endswith(".dll")]
                size_mb = sum(os.path.getsize(os.path.join(dll_dir, f)) for f in dlls) / (1024 * 1024)
                print(f"  Bundling {pkg}: {len(dlls)} DLLs ({size_mb:.0f} MB)")
                for dll in dlls:
                    cmd += ["--add-binary", f"{os.path.join(dll_dir, dll)};."]

        ct2_dir = os.path.join(site_pkgs, "ctranslate2")
        if os.path.isdir(ct2_dir):
            for f in os.listdir(ct2_dir):
                if f.endswith(".dll"):
                    cmd += ["--add-binary", f"{os.path.join(ct2_dir, f)};ctranslate2"]

    # Hidden imports
    cmd += [
        "--hidden-import", "faster_whisper",
        "--hidden-import", "ctranslate2",
        "--hidden-import", "huggingface_hub",
        "--hidden-import", "tokenizers",
        "--hidden-import", "openai",
        "--hidden-import", "flask",
        "--collect-all", "faster_whisper",
        "--collect-all", "ctranslate2",
    ]

    if mode == "gpu":
        cmd += [
            "--collect-all", "nvidia.cublas",
            "--collect-all", "nvidia.cudnn",
        ]

    cmd.append("app.py")

    print(f"\n  Running PyInstaller...")
    subprocess.run(cmd, check=True)

    # Create a launcher bat
    run_bat = os.path.join(DIST_DIR, "Run Speech-to-Text.bat")
    with open(run_bat, "w") as f:
        f.write('@echo off\n')
        f.write('title Speech to Text\n')
        f.write('cd /d "%~dp0"\n')
        f.write('start http://localhost:8080\n')
        f.write(f'{APP_NAME}.exe\n')
        f.write('pause\n')

    # Calculate size
    total_size = 0
    for dirpath, _, filenames in os.walk(DIST_DIR):
        for fn in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, fn))
    size_mb = total_size / (1024 * 1024)

    print(f"\n{'=' * 50}")
    print(f"  Build complete!")
    print(f"  Output: {os.path.abspath(DIST_DIR)}")
    print(f"  Size: {size_mb:.0f} MB")
    print(f"  Mode: {mode.upper()}")
    print(f"  Models will download on first use.")
    print(f"  To run: double-click 'Run Speech-to-Text.bat'")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode not in ("cpu", "gpu"):
            print(f"Unknown mode: {mode}. Use 'cpu' or 'gpu'.")
            sys.exit(1)
    else:
        try:
            import ctranslate2
            has_cuda = "cuda" in ctranslate2.get_supported_compute_types("cuda")
        except Exception:
            has_cuda = False
        mode = "gpu" if has_cuda else "cpu"
        print(f"  Auto-detected mode: {mode.upper()}")

    build(mode)
