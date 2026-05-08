"""Import-path helpers for running from repo root, AI-Poker-Agent, or Colab."""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def engine_root() -> Path:
    return repo_root() / "AI-Poker-Agent"


def project_path_candidates() -> list[Path]:
    """Likely project roots when this folder is copied into Colab/Drive."""
    roots = [
        repo_root(),
        Path.cwd(),
        Path.cwd() / "AI-Poker-Agent",
        engine_root(),
    ]
    roots.extend(parent for parent in Path(__file__).resolve().parents[:4])
    roots.extend(parent / "AI-Poker-Agent" for parent in Path(__file__).resolve().parents[:4])

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        resolved = str(root.resolve()) if root.exists() else str(root)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(root)
    return unique


def ensure_project_paths() -> None:
    """Make local players and the vendored pypokerengine importable."""
    for candidate in reversed(project_path_candidates()):
        if candidate.exists():
            path = str(candidate)
            if path not in sys.path:
                sys.path.insert(0, path)


def find_project_file(*relative_paths: str) -> Path | None:
    for root in project_path_candidates():
        for relative_path in relative_paths:
            candidate = root / relative_path
            if candidate.exists():
                return candidate
    return None
