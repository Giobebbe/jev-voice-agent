"""Settings, secrets and the allowlisted sandbox roots."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

KEYCHAIN_SERVICE = "jev-voice-agent"

DESKTOP = Path.home() / "Desktop"
# The only two folders the agent may ever create or delete things in.
ALLOWED_ROOTS = (DESKTOP / "Folder JEV", DESKTOP / "Folder JEV copy")

PROJECT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_DIR / "runs"
AUDIO_CACHE = PROJECT_DIR / "audio_cache"

# Apps the agent must never quit: they host the agent or the recording.
PROTECTED_APPS = {"Terminal", "Ghostty", "Visual Studio Code", "Code", "Claude", "OBS", "Finder"}


def secret(name: str) -> str:
    """Environment first, then the macOS Keychain entry stored for this project."""
    value = os.environ.get(name)
    if value:
        return value
    out = subprocess.run(
        ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", name, "-w"],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(f"Missing secret {name}: not in env nor in Keychain service {KEYCHAIN_SERVICE}")
    return out.stdout.strip()


@dataclass
class Settings:
    root: Path = ALLOWED_ROOTS[0]
    mic: str = "MacBook Pro Microphone"
    browser: str = "Google Chrome"
    camera: str = "MacBook Pro Camera"
    voice_id: str = "cgSgspJ2msm6clMCkdW9"  # Jessica, bright and short
    tts_model: str = "eleven_flash_v2_5"
    jev_model: str = "jev-latest"
    speak: bool = True
    dry_run: bool = False
    fast_lane: bool = True  # on-device Whisper partials for closed-vocabulary commands
    # Policy thresholds (tuned by evals/)
    tool_min: float = 0.6
    arg_min: float = 0.5
    early_tool_min: float = 0.9
    early_arg_min: float = 0.85
    item_min: float = 0.9  # existing file/folder picked by Jev: act only when clearly meant
    quiet_ms: int = 300
    scribe_lag_s: float = 0.6  # a Scribe partial covers the audio up to ~this long before it
    vad_silence_secs: float = 0.5
    fast_tool_min: float = 0.9
    fast_arg_min: float = 0.85
    keyterms: list[str] = field(
        default_factory=lambda: ["TextEdit", "Photo Booth", "Folder JEV", "x.com", "Jev", "screenshot"]
    )
