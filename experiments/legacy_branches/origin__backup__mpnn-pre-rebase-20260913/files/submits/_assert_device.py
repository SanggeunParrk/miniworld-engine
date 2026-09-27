"""Fail the job if it did not land on the GPU the launcher was written for.

This guard used to be an inline `pixi run python -c "..."` in every launcher. It never
ran: pixi parses the task string with a shell parser that rejects the f-string braces in
the assertion message, and the launchers use `set -uo pipefail` without `-e`, so the
parse error printed and the job carried on. Eleven launchers were guarding nothing, and a
job that landed on the wrong card still wrote results under that card's slug.

Kept as a file so the arguments are the only thing that varies and nothing has to survive
a shell parser.

  python submits/_assert_device.py --capability 8.6 --name A5000 --name A6000
"""

from __future__ import annotations

import argparse
import sys

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capability",
        help="required compute capability as MAJOR.MINOR, e.g. 8.6 for sm_86",
    )
    parser.add_argument(
        "--major",
        type=int,
        help="required compute-capability MAJOR only, for guards that were written loose",
    )
    parser.add_argument(
        "--name",
        action="append",
        default=[],
        help="accepted substring of the device name; repeatable, any match passes",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FATAL: no CUDA device visible", file=sys.stderr)
        return 3

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    total = torch.cuda.mem_get_info()[1] / 2**30
    print(f"torch {torch.__version__}  {name}  sm_{major}{minor}  {total:.1f} GiB")

    if args.capability:
        want = tuple(int(part) for part in args.capability.split("."))
        if (major, minor) != want:
            print(
                f"FATAL: expected sm_{want[0]}{want[1]}, got sm_{major}{minor} ({name})",
                file=sys.stderr,
            )
            return 4

    if args.major is not None and major != args.major:
        print(
            f"FATAL: expected sm_{args.major}x, got sm_{major}{minor} ({name})",
            file=sys.stderr,
        )
        return 4

    if args.name and not any(pattern in name for pattern in args.name):
        print(
            f"FATAL: expected one of {args.name}, got {name}",
            file=sys.stderr,
        )
        return 5

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
