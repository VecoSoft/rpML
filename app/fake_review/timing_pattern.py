"""
TIMING_PATTERN signal (spec §14): flags an abnormally fast cluster of
reviews landing on one business.
"""
from datetime import timedelta
from typing import List

from app.schemas import ReviewInput

# A burst is "abnormal" if this many-or-more reviews land within the window below.
_BURST_WINDOW = timedelta(hours=1)
_BURST_THRESHOLD = 5


def score_timing_pattern(review: ReviewInput, recent_reviews: List[ReviewInput]) -> tuple[int, str]:
    window_start = review.created_at - _BURST_WINDOW
    window_end = review.created_at + _BURST_WINDOW

    cluster = [
        r for r in recent_reviews
        if r.id != review.id and window_start <= r.created_at <= window_end
    ]
    cluster_size = len(cluster) + 1  # include the review itself

    if cluster_size < _BURST_THRESHOLD:
        return 0, f"no abnormal clustering ({cluster_size} review(s) within +/-1h)"

    # Scale linearly past the threshold, capped at 100 (10+ reviews/hour = max score).
    over = cluster_size - _BURST_THRESHOLD
    score = min(100, 50 + over * 10)
    return score, f"{cluster_size} reviews landed within a 2-hour window (threshold {_BURST_THRESHOLD})"
