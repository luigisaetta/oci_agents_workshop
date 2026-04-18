"""
Author: L. Saetta
Date last modified: 2026-04-18
License: MIT
Description: Ensure project root is available on sys.path during test collection.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
