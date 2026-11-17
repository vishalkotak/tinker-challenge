"""Sanity tests for lipo.checks: Wu's verified answers mostly pass, known reward hacks all fail. Run: uv run python -m lipo.test_checks"""

import json
from pathlib import Path

from lipo.checks import check_answer

DATA_DIR = Path(__file__).parent / "data"

HACKS = {
    "cyrillic look-alike e": "Thе sky looks bluе as light hits tiny bits of air all around us all day long.",
    "dropped e": "Th sky looks blu as sunlight scatrs off tiny air bits around us all day long now.",
    "one word": "Football.",
    "loop": "I want to know what you want. I want to know what you want. I want to know what you want.",
    "letter stream": "Tgth rthrt bnmk qwrt plk xzv wqrt mnbv ghjk tyui opas dfgh jklz xcvb.",
}
# GOTCHA: writing this by hand took three tries; "bounces", "shorter" and "scatters" all hide an e.
GOOD = "Sunlight hits air and bounds off tiny bits of gas. Short wavy light is thrown about most, so our sky looks bright all day long."


def test_hacks_fail() -> None:
    """Every known hack must fail the gates."""
    for name, text in HACKS.items():
        assert not check_answer(text, "Why is the sky blue?").passes_gates, name


def test_good_answer_passes() -> None:
    """A real, fluent lipogram must pass."""
    assert check_answer(GOOD, "Why is the sky blue?").passes_gates


def test_wu_verified_answers_mostly_pass() -> None:
    """At least 97% of Wu's 200 hand-verified answers must pass (the few rejects are under 12 words or coin-flip strings)."""
    rows = [json.loads(line) for line in (DATA_DIR / "verified_200_20261015.jsonl").open()]
    passing = sum(check_answer(row["response"], row["prompt"]).passes_gates for row in rows)
    assert passing >= 0.97 * len(rows), passing


if __name__ == "__main__":
    for test in (test_hacks_fail, test_good_answer_passes, test_wu_verified_answers_mostly_pass):
        test()
        print(f"ok  {test.__name__}")
