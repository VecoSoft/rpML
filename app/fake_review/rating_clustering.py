"""
RATING_CLUSTERING signal (spec §14): flags a sudden concentration of all
5-star or all 1-star reviews.
"""
from typing import List

from app.schemas import ReviewInput

# Look at the N most recent reviews (including the one under evaluation) for the business.
_LOOKBACK_COUNT = 10
_CONCENTRATION_THRESHOLD = 0.8  # 80%+ at one extreme is "sudden concentration"


def score_rating_clustering(review: ReviewInput, recent_reviews: List[ReviewInput]) -> tuple[int, str]:
    same_business = sorted(
        [r for r in recent_reviews if r.id != review.id] + [review],
        key=lambda r: r.created_at,
        reverse=True,
    )[:_LOOKBACK_COUNT]

    if len(same_business) < 5:
        return 0, f"too few recent reviews to assess clustering ({len(same_business)})"

    five_star = sum(1 for r in same_business if r.rating == 5)
    one_star = sum(1 for r in same_business if r.rating == 1)
    total = len(same_business)

    five_ratio = five_star / total
    one_ratio = one_star / total
    dominant_ratio = max(five_ratio, one_ratio)

    if dominant_ratio < _CONCENTRATION_THRESHOLD:
        return 0, f"rating distribution looks normal ({five_star}x5-star, {one_star}x1-star of {total})"

    over = dominant_ratio - _CONCENTRATION_THRESHOLD
    score = min(100, int(round(50 + over / (1 - _CONCENTRATION_THRESHOLD) * 50)))
    extreme = "5-star" if five_ratio >= one_ratio else "1-star"
    return score, f"{dominant_ratio:.0%} of last {total} reviews are {extreme}"
