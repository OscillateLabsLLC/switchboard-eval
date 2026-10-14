"""Where the harness data lives: the repo root, next to this package."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Device keys in code and prediction file names -> their folders
DEVICE_DIRS = {"mac": "devices/desktop", "hub": "devices/speaker"}
