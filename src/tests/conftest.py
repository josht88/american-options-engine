# src/tests/conftest.py
import sys
from pathlib import Path

# Ensure the repository root is importable as a module location so `import app` works.
# <repo_root>/src/tests/conftest.py -> parents[2] is <repo_root>
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
