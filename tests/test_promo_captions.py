"""Offline tests for caption validation — no network, the LLM is a fake."""
import json
import unittest

from app.promo.captions import allowed_numbers, generate_captions, is_valid

REQ = {
    "businessName": "KFC Gulshan 1",
    "category": "Restaurant & Food",
    "area": "Gulshan",
    "type": "OFFER",
    "offer": {"title": "Zinger Buy 1 Get 1", "offerType": "BUY_ONE_GET_ONE", "offerPrice": 450, "validUntil": "30 Oct"},
    "tone": "friendly",
    "language": "en",
}


class FakeLLM:
    def __init__(self, captions):
        self.captions = captions

    def invoke(self, prompt):
        class R:
            content = "```json\n" + json.dumps(self.captions, ensure_ascii=False) + "\n```"
        return R()


class CaptionValidationTest(unittest.TestCase):
    def test_numbers_must_come_from_the_data(self):
        allowed = allowed_numbers(REQ)
        self.assertTrue(is_valid("Zinger Buy 1 Get 1 for ৳450 until 30 Oct 🎉", allowed))
        self.assertFalse(is_valid("Now only ৳299!", allowed))            # invented price
        self.assertFalse(is_valid("৪৫০ টাকায় ৫০% ছাড়!", allowed))       # invented 50% (Bangla digits)
        self.assertTrue(is_valid("মাত্র ৪৫০ টাকায় জিঙ্গার!", allowed))   # 450 in Bangla digits is real

    def test_claims_emojis_hashtags_and_length(self):
        allowed = allowed_numbers(REQ)
        self.assertFalse(is_valid("The best zinger in Dhaka", allowed))
        self.assertFalse(is_valid("ঢাকার সেরা জিঙ্গার", allowed))
        self.assertFalse(is_valid("Yum 🎉🎉🎉", allowed))
        self.assertFalse(is_valid("#a #b #c #d", allowed))
        self.assertFalse(is_valid("x" * 221, allowed))

    def test_generate_drops_invalid_and_keeps_three(self):
        fake = FakeLLM([
            "Zinger Buy 1 Get 1 at KFC Gulshan 1 🎉",
            "No.1 zinger in town!",
            "Grab it before 30 Oct #KFC",
            "Only ৳450 at KFC Gulshan 1",
        ])
        out = generate_captions(REQ, llm=fake)
        self.assertEqual(3, len(out))
        self.assertNotIn("No.1 zinger in town!", out)

    def test_garbage_output_returns_nothing(self):
        class Broken:
            def invoke(self, prompt):
                class R:
                    content = "sorry, I can't"
                return R()
        self.assertEqual([], generate_captions(REQ, llm=Broken()))


if __name__ == "__main__":
    unittest.main()


class CaptionDeadlineTest(unittest.TestCase):
    def test_slow_model_returns_503_within_the_deadline(self):
        import time
        from unittest import mock

        from fastapi import HTTPException

        from app.promo import router as promo_router

        def slow(_req):
            time.sleep(2)
            return ["a", "b", "c"]

        with mock.patch.object(promo_router, "generate_captions", slow), \
                mock.patch.object(promo_router, "CAPTION_DEADLINE_SECONDS", 0.3), \
                mock.patch.object(promo_router.settings, "gemini_api_key", "test-key"):
            started = time.monotonic()
            with self.assertRaises(HTTPException) as ctx:
                promo_router.captions(promo_router.CaptionRequest(businessName="KFC Gulshan 1"))
            self.assertEqual(ctx.exception.status_code, 503)
            self.assertLess(time.monotonic() - started, 1.5)
