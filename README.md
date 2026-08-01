# ML Service — Fake-Review Detection + AI Review Summary

Python (FastAPI) microservice implementing the two AI-heavy features from the
spec that don't belong in the Java/Spring Boot DAO layer:

- **§14 Fake-review detection** — 3 signals (content-pattern via multilingual
  embeddings + LLM template-check, timing-pattern, rating-clustering) combined
  into a single 0-100 suspicion score and a derived `visibility_status`.
- **§15 AI review summary** — an actual **LangGraph** pipeline (theme
  extraction → synthesis) producing the short per-business summary.

Spring Boot calls this service over REST (see `fakereview.FakeReviewMlClient`
and `summary.SummaryMlClient` on the Java side) and persists the results
through the existing DAO repositories — this service is stateless and owns
no database of its own.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in GEMINI_API_KEY
uvicorn app.main:app --host 0.0.0.0 --port 8081
```

First run downloads the embedding model from Hugging Face — needs outbound
network access to huggingface.co (not available in every sandboxed environment).

## Endpoints

### `POST /fake-review/analyze`
Request: the review under evaluation + a window of recent reviews for the
same business (for the timing/rating/content comparison signals).
Response: per-signal scores + the combined `suspicionScore` (0-100) and
`visibilityStatus` (`RECOMMENDED` / `NOT_RECOMMENDED` / `HIDDEN`), matching
the spec's thresholds (0-30 / 31-70 / 71-100) exactly.

### `POST /review-summary/generate`
Request: businessId + the reviews to summarize (the caller — Java side — is
responsible for only calling this once every 10 new reviews, per spec §15;
this service does not track that counter itself).
Response: the generated summary text.

## Design notes

- The embedding model is multilingual specifically because reviews are
  frequently Bangla/Banglish, not English-only (spec §14 explicit requirement).
- The content-pattern signal is two-part: (1) embedding cosine-similarity
  against recent reviews on the same business — catches copy-pasted/templated
  text even across languages — and (2) a lightweight LLM judgment call for
  "does this read as generic/templated" independent of any specific match.
- This service does not decide *what to do* with `HIDDEN`/`NOT_RECOMMENDED`
  results (notifying owners, moving reviews to a queue, recomputing rating
  aggregates) — that orchestration stays on the Java side, which already owns
  the transactional writes to `review`, `fake_review_signal`, and `business`.
