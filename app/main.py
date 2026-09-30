from fastapi import FastAPI

from app.fake_review.router import router as fake_review_router
from app.promo.router import router as promo_router
from app.summary.router import router as summary_router

app = FastAPI(
    title="Business Review Platform — ML Service",
    description="Fake-review detection (§14) + AI review summary via LangGraph (§15) + business promotion captions.",
    version="0.1.0",
)

app.include_router(fake_review_router)
app.include_router(summary_router)
app.include_router(promo_router)


@app.get("/health")
def health():
    return {"status": "ok"}
