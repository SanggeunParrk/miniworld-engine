"""GPU support policy shared by the cache builder and internal backend dispatch.

The CSV parser is lightweight; importing it does not import the builder or query CUDA.
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
