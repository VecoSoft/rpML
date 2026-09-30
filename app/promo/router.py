from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.promo.captions import generate_captions

router = APIRouter(prefix="/promo", tags=["promo"])

CAPTION_DEADLINE_SECONDS = 7.5
_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="captions")


class CaptionRequest(BaseModel):
    """Field names match the Java side (promo.PromoCaptionService.MlRequest) exactly."""
    businessName: str = Field(min_length=1, max_length=200)
    category: Optional[str] = None
    area: Optional[str] = None
    type: Literal["MENU_ITEM", "OFFER", "EVENT", "ANNOUNCEMENT", "GENERAL"] = "GENERAL"
    offer: Optional[Dict[str, Any]] = None
    menuItem: Optional[Dict[str, Any]] = None
    event: Optional[Dict[str, Any]] = None
    tone: Literal["friendly", "premium", "urgent"] = "friendly"
    language: Literal["bn", "en"] = "en"


class CaptionResponse(BaseModel):
    captions: List[str]


@router.post("/captions", response_model=CaptionResponse)
def captions(request: CaptionRequest) -> CaptionResponse:
    """
    3 validated captions, or 503 when the model is unavailable / produced fewer than 3 valid ones.
    The backend treats any error as "use the template fallback", so failing loudly is safe.
    """
    if not settings.gemini_api_key:
        raise HTTPException(status_code=503, detail="Caption model not configured")
    # Hard deadline: the client's own timeout doesn't cover its internal backoff or model
    # "thinking" time, and the backend stops waiting at 8s anyway.
    future = _pool.submit(generate_captions, request.model_dump())
    try:
        result = future.result(timeout=CAPTION_DEADLINE_SECONDS)
    except TimeoutError as exc:
        future.cancel()
        raise HTTPException(status_code=503, detail="Caption model too slow") from exc
    except Exception as exc:  # network / quota / model errors
        raise HTTPException(status_code=503, detail=f"Caption model unavailable: {type(exc).__name__}") from exc
    if len(result) < 3:
        raise HTTPException(status_code=503, detail="Not enough valid captions")
    return CaptionResponse(captions=result)
