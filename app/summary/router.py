from fastapi import APIRouter

from app.schemas import SummaryGenerationRequest, SummaryGenerationResponse
from app.summary.graph import generate_summary

router = APIRouter(prefix="/review-summary", tags=["review-summary"])


@router.post("/generate", response_model=SummaryGenerationResponse, response_model_by_alias=True)
def generate(request: SummaryGenerationRequest) -> SummaryGenerationResponse:
    summary_text = generate_summary(request.reviews)
    return SummaryGenerationResponse(
        business_id=request.business_id,
        summary_text=summary_text,
        review_count_considered=len(request.reviews),
    )
