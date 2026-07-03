from __future__ import annotations


def get_seq_group(length: int) -> int:
    """Bucket pair-row count M=L*L for Triton autotune keys."""
    group_lengths = [32 * 32, 64 * 64, 128 * 128, 256 * 256, 384 * 384, 512 * 512, 768 * 768, 1024 * 1024]
    for group_idx, group_length in enumerate(group_lengths):
        if length <= group_length:
            return group_idx
    return len(group_lengths) - 1
