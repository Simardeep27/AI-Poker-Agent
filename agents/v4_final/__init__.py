"""Import helpers for the final modular poker agent."""

import os
import sys

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from player import MyPlayer, setup_ai

__all__ = ["MyPlayer", "setup_ai"]
