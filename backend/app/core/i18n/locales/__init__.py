"""
Init file for locales package. Enables Python imports and dynamic locale discovery.
"""

from pathlib import Path

__all__ = [
    p.name for p in Path(__file__).parent.iterdir()
    if p.is_dir() and not p.name.startswith('__')
]
