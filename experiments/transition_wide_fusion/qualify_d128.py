"""Add unchanged D128 to the same local comparison, sequentially on one GPU."""
import subprocess
import sys
from pathlib import Path

for length in (384,768):
    subprocess.run([sys.executable,str(Path(__file__).with_name('qualify.py')),
                    '--width','128','--length',str(length)],check=True)
