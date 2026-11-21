"""Raw records of records/<card>/ -> E1_<card>.csv (columns: benchmarks/paper/README.md).   python collect.py --card h100"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paper_common import collect_exp

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--card", required=True, choices=["h100", "a100", "rtx5090", "b200"])
    collect_exp.main("speed", "E1", str(HERE), p.parse_args().card)
