"""
Combines the 3 MVP signals (spec §14) into one 0-100 suspicion score and
derives the visibility_status using the spec's exact thresholds:
  0-30   -> RECOMMENDED
  31-70  -> NOT_RECOMMENDED
  71-100 -> HIDDEN
"""
from app.config import settings
from app.fake_review.content_pattern import score_content_pattern
from app.fake_review.rating_clustering import score_rating_clustering
from app.fake_review.timing_pattern import score_timing_pattern
from app.schemas import FakeReviewAnalysisRequest, FakeReviewAnalysisResponse, SignalResult

# Content-pattern is weighted highest since it's the most direct fake-review
# indicator; timing/rating-clustering are corroborating signals.
_WEIGHTS = {
    "CONTENT_PATTERN": 0.5,
    "TIMING_PATTERN": 0.25,
    "RATING_CLUSTERING": 0.25,
}


def _visibility_for(score: int) -> str:
    if score <= settings.low_confidence_max:
        return "RECOMMENDED"
    if score <= settings.medium_confidence_max:
        return "NOT_RECOMMENDED"
    return "HIDDEN"


def analyze(request: FakeReviewAnalysisRequest) -> FakeReviewAnalysisResponse:
    review = request.review
    recent = request.recent_reviews

    content_score, content_detail = score_content_pattern(review, recent)
    timing_score, timing_detail = score_timing_pattern(review, recent)
    rating_score, rating_detail = score_rating_clustering(review, recent)

    signals = [
        SignalResult(signal_type="CONTENT_PATTERN", score=content_score, detail=content_detail),
        SignalResult(signal_type="TIMING_PATTERN", score=timing_score, detail=timing_detail),
        SignalResult(signal_type="RATING_CLUSTERING", score=rating_score, detail=rating_detail),
    ]

    combined = sum(s.score * _WEIGHTS[s.signal_type] for s in signals)
    suspicion_score = int(round(combined))

    return FakeReviewAnalysisResponse(
        review_id=review.id,
        suspicion_score=suspicion_score,
        visibility_status=_visibility_for(suspicion_score),
        signals=signals,
    )
