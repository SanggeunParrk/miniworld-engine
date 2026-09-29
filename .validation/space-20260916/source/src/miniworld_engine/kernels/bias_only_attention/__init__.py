"""Bias-only attention: softmax over a pair bias alone, no q/k projection.

The attention used by ``modules/attention_pair_bias.py``, where the logits ARE the pair bias --
there is no query-key product -- so the kernel reads ``v`` and ``bias`` and nothing else.

Layout, as in every kernel family here: ``reference.py`` is the torch definition the checkers
compare against, ``triton/`` is the backend, and ``dispatch.py`` holds the per-GPU calibrated
choice between the fused and split gate epilogues.
"""
