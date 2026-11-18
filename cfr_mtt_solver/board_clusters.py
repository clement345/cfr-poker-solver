"""72-bucket flop board clustering: 4 × 3 × 3 × 2."""
from __future__ import annotations
import numpy as np
from .cards import card_rank, card_suit, card_from_str, card_to_str, RANK_CHARS

# cluster_id = high_rank_tier*18 + flush_tier*6 + connect_tier*2 + is_paired
# 4 × 3 × 3 × 2 = 72

N_CLUSTERS = 72


def cluster_flop(c1: int, c2: int, c3: int) -> int:
    """Map a 3-card flop (card ints) to one of 72 cluster indices."""
    ranks = sorted([card_rank(c1), card_rank(c2), card_rank(c3)], reverse=True)
    suits = [card_suit(c1), card_suit(c2), card_suit(c3)]

    # High-rank tier: based on highest card
    high = ranks[0]
    if high <= 6:   # 2–8
        hr_tier = 0
    elif high <= 8: # 9–T  (rank 7=9, 8=T)
        hr_tier = 1
    elif high <= 10: # J–Q (rank 9=J, 10=Q)
        hr_tier = 2
    else:            # K–A (rank 11=K, 12=A)
        hr_tier = 3

    # Flush tier
    suit_counts = [suits.count(s) for s in range(4)]
    max_suit = max(suit_counts)
    if max_suit == 3:
        flush_tier = 2   # monotone
    elif max_suit == 2:
        flush_tier = 1   # two-tone
    else:
        flush_tier = 0   # rainbow

    # Connect tier: count pairs of cards within 2 ranks (gaps ≤ 1)
    gaps = [ranks[i] - ranks[i + 1] for i in range(2)]
    connected_pairs = sum(1 for g in gaps if g <= 2 and g > 0)
    # Also handle wraparound is not standard in Hold'em
    if connected_pairs >= 2:
        conn_tier = 2   # board has two connected pairs (e.g., 9-8-7)
    elif connected_pairs == 1:
        conn_tier = 1   # one connected pair (e.g., 9-8-2)
    else:
        conn_tier = 0   # disconnected

    # Paired tier
    is_paired = 1 if len(set(ranks)) < 3 else 0

    return hr_tier * 18 + flush_tier * 6 + conn_tier * 2 + is_paired


def cluster_features(cluster_id: int) -> dict[str, int]:
    """Decompose a cluster id into its four feature components."""
    is_paired = cluster_id % 2
    conn_tier = (cluster_id // 2) % 3
    flush_tier = (cluster_id // 6) % 3
    hr_tier = cluster_id // 18
    return {
        "high_rank_tier": hr_tier,
        "flush_tier": flush_tier,
        "connect_tier": conn_tier,
        "is_paired": is_paired,
    }


def _build_cluster_representatives() -> dict[int, tuple[int, int, int]]:
    """One representative flop (card ints) per reachable cluster."""
    reps: dict[int, tuple[int, int, int]] = {}
    for r1 in range(12, -1, -1):
        for r2 in range(r1, -1, -1):
            for r3 in range(r2, -1, -1):
                for s1 in range(4):
                    for s2 in range(4):
                        for s3 in range(4):
                            c1 = r1 * 4 + s1
                            c2 = r2 * 4 + s2
                            c3 = r3 * 4 + s3
                            if len({c1, c2, c3}) < 3:
                                continue
                            cid = cluster_flop(c1, c2, c3)
                            if cid not in reps:
                                reps[cid] = (c1, c2, c3)
    return reps


# Only reachable clusters (paired boards can't have conn_tier=2 → fewer than 72 entries)
CLUSTER_REPR: dict[int, tuple[int, int, int]] = _build_cluster_representatives()
REACHABLE_CLUSTERS: list[int] = sorted(CLUSTER_REPR.keys())


def cluster_repr_str(cluster_id: int) -> str:
    if cluster_id not in CLUSTER_REPR:
        return "???"
    c1, c2, c3 = CLUSTER_REPR[cluster_id]
    return f"{card_to_str(c1)}{card_to_str(c2)}{card_to_str(c3)}"
