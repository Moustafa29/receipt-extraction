"""Point Hugging Face at the repo-local cache when one exists.

.hf_cache/ is gitignored, so a fresh clone falls through to the default
~/.cache/huggingface and downloads as normal. This only keeps a working copy
from re-downloading layoutlmv3-base under a different HOME.

Set before any transformers import, because the library reads these at import.
"""
from __future__ import annotations

import os
from pathlib import Path

_CACHE = Path(__file__).resolve().parent.parent / ".hf_cache"
if _CACHE.is_dir():
    os.environ.setdefault("HF_HOME", str(_CACHE))
