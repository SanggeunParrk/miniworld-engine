"""One allocated GPU, sequential shape/config sweeps."""
from pathlib import Path
import subprocess
import sys

here=Path(__file__).parent
for width in (256,384):
    subprocess.run([sys.executable,str(here/'sweep_ln_residual.py'),'--width',str(width),
                    '--length','384','--persistent'],check=True)
subprocess.run([sys.executable,str(here/'sweep_native_dx_ln.py'),'--width','384','--length','384'],check=True)
