from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


def to_camel(snake: str) -> str:
    parts = snake.split("_")
    return parts[0] + "".join(p.title() for p in parts[1:])


class CamelModel(BaseModel):
    """Base model that (de)serializes as camelCase JSON to match the Java/Jackson side."""
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


# ---------------------------------------------------------------------------
# Shared review shape
# ---------------------------------------------------------------------------
class ReviewInput(CamelModel):
    id: str
    business_id: str
    rating: int = Field(ge=1, le=5)
    content: str = ""
    created_at: datetime


# ---------------------------------------------------------------------------
# §14 Fake-review detection
# ---------------------------------------------------------------------------
class FakeReviewAnalysisRequest(CamelModel):
    review: ReviewInput
    # Recent reviews on the same business (e.g. last 30-90 days), used as the
    # comparison window for content/timing/rating-clustering signals. The
    # target review itself may or may not be included; this service dedupes by id.
    recent_reviews: List[ReviewInput] = []


class SignalResult(CamelModel):
    signal_type: str  # CONTENT_PATTERN | TIMING_PATTERN | RATING_CLUSTERING
    score: int = Field(ge=0, le=100)
    detail: str


class FakeReviewAnalysisResponse(CamelModel):
    review_id: str
    suspicion_score: int = Field(ge=0, le=100)
    visibility_status: str  # RECOMMENDED | NOT_RECOMMENDED | HIDDEN
    signals: List[SignalResult]


# ---------------------------------------------------------------------------
# §15 AI review summary
# ---------------------------------------------------------------------------
class SummaryReviewInput(CamelModel):
    id: str
    rating: int
    content: str = ""


class SummaryGenerationRequest(CamelModel):
    business_id: str
    reviews: List[SummaryReviewInput]


class SummaryGenerationResponse(CamelModel):
    business_id: str
    summary_text: str
    review_count_considered: int
