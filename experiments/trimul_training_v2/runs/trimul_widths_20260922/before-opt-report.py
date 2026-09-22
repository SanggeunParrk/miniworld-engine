"""Current report generator; prior generator archived in before-port-report.py."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent.parent/"trimul_cuda_widths_20260923/report.py"),run_name="__main__")
