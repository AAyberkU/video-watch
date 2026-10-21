"""Make the CLI importable: it lives in the skill folder (skills/video-watch/scripts)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills" / "video-watch" / "scripts"))
