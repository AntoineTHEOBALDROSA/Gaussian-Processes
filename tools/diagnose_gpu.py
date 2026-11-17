"""À lancer sur le serveur SSH : .venv/bin/python tools/diagnose_gpu.py"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hardware import gpu_capabilities

if __name__ == '__main__':
    print(json.dumps(gpu_capabilities(), indent=2, ensure_ascii=False))
