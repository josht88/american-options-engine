# src/tests/test_capture_env.py
import json, os, sys, subprocess

def test_capture_env_writes(tmp_path):
    outdir = tmp_path / "run"
    cmd = [sys.executable, "-m", "scripts.capture_env", "--outdir", str(outdir)]
    subprocess.check_call(cmd)
    p = outdir / "env.json"
    assert p.exists()
    data = json.loads(p.read_text())
    assert "python" in data and "packages" in data and "captured_at" in data
