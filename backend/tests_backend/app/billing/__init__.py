"""
Spotify AI Agent - Billing Tests Package
=======================================

Test package initialization for billing system tests.
"""

import sys
import os
from pathlib import Path

# Add the app directory to the Python path for imports
app_dir = Path(__file__).parent.parent.parent.parent / "app"
sys.path.insert(0, str(app_dir))
