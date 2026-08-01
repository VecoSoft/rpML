from fastapi import APIRouter

from app.fake_review.scorer import analyze
from app.schemas import FakeReviewAnalysisRequest, FakeReviewAnalysisResponse

router = APIRouter(prefix="/fake-review", tags=["fake-review"])


@router.post("/analyze", response_model=FakeReviewAnalysisResponse, response_model_by_alias=True)
def analyze_review(request: FakeReviewAnalysisRequest) -> FakeReviewAnalysisResponse:
    return analyze(request)
