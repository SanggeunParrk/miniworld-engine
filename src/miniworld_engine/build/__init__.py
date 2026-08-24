"""Build-time policy: what the autotune-cache builder is allowed to build, per GPU.

Separate from ``autotune/`` on purpose. ``autotune`` is runtime -- it is imported by every forward
that consults the cache. This package is consulted only while BUILDING the cache, so nothing here
belongs on the hot import path.
"""

from miniworld_engine.build.matrix import (
    RULES_DIR,
    Rule,
    allows,
    decide,
    known_gpus,
    rules,
    sm_tag,
)

__all__ = ["RULES_DIR", "Rule", "allows", "decide", "known_gpus", "rules", "sm_tag"]
