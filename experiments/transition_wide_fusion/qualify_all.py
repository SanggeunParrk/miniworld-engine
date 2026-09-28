import subprocess
import sys
from pathlib import Path

for d in (256,384,512):
    for length in (384,768):
        subprocess.run([sys.executable,str(Path(__file__).with_name('qualify.py')),
                        '--width',str(d),'--length',str(length)],check=True)
