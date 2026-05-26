from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys


def run(module_name: str) -> int:
    """Run a shared hook module with OpenCode agent context."""

    os.environ["AI_POLICY_AGENT"] = "opencode"
    plugin_root = Path(__file__).resolve().parents[1]
    plugin_root_text = str(plugin_root)
    if plugin_root_text not in sys.path:
        sys.path.insert(0, plugin_root_text)
    return int(importlib.import_module(module_name).main())
