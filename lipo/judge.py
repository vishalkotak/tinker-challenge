"""Quality judge using Kevin Wu's exact grading prompt (MIT), run on a Tinker-served model."""

import asyncio
import json
import re
from dataclasses import dataclass

from lipo.chat import ChatModel

# GOTCHA: verbatim from Wu's core/judge.py (github.com/kevinyvv/qwen-3-1.7b-finetune-proj, MIT). Do not
# edit: the head-to-head is only fair if both systems are scored with the same rubric text.
JUDGE_SYSTEM = """You grade short answers written by a small chat model. You will see a user's request and the model's answer. Score two things from 1 to 5.

Coherence: is the answer readable, sensible English?
5 = clear and natural. Every word is a real, correctly spelled English word, sentences are grammatical, and the ideas connect from start to finish.
4 = clear overall, with at most one minor awkward phrase or slip.
3 = understandable but with noticeable problems: broken phrasing, a misspelled or made-up word, or some repetition.
2 = hard to follow: several misspelled or made-up words, broken grammar, or looping text.
1 = gibberish, mostly repetition, or not English.
Repetition counts against coherence. If the same phrase or line appears three or more times, or the answer keeps circling back to the same words instead of moving forward, coherence is 3 at most. If most of the answer is repetition, it is 1 or 2.
Unusual or old-fashioned word choices are fine as long as the words are real and the meaning is clear.

Responds: does the answer address the request?
5 = directly attempts what was asked.
4 = attempts what was asked, with small gaps or slight drift.
3 = partly on topic, or attempts only part of the request.
2 = mostly off topic, or generic filler that could follow almost any request.
1 = ignores the request.
Do not judge factual accuracy. Wrong facts, wrong definitions or made-up details don't lower either score, as long as the answer attempts the request.

Some answers are marked as cut off: they stopped because they hit a length limit. Judge only the text that is there. Do not mark a cut-off answer down for ending mid-sentence or for not reaching later parts of the request (for example, giving 2 of 5 requested items before the cut). Ask whether what is there is coherent and on its way to doing what was asked. Do not reward or penalize length otherwise.

Reply with only this JSON object and no other text:
{"coherence": <1-5>, "responds": <1-5>, "note": "<one sentence explaining the scores>"}"""

PASS_SCORE = 4


@dataclass(frozen=True)
class Judgment:
    """One judge verdict; scores are None when the judge's reply couldn't be parsed after retries."""

    coherence: int | None
    responds: int | None
    note: str

    @property
    def usable(self) -> bool:
        """Wu's quality bar: both scores at least 4 (the e-check is applied separately)."""
        return self.coherence is not None and self.responds is not None and min(self.coherence, self.responds) >= PASS_SCORE


def _parse(reply: str) -> Judgment | None:
    """Extract the JSON verdict from a judge reply, or None if it is malformed."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    scores = [data.get("coherence"), data.get("responds")]
    if not all(isinstance(score, int) and 1 <= score <= 5 for score in scores):
        return None
    return Judgment(scores[0], scores[1], str(data.get("note", "")))


async def judge(model: ChatModel, prompt: str, answer: str, cut_off: bool = False, attempts: int = 2) -> Judgment:
    """Score one answer with Wu's rubric, retrying once on an unreadable reply."""
    label = "Answer (cut off at the length limit)" if cut_off else "Answer"
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"Request:\n<<<\n{prompt}\n>>>\n\n{label}:\n<<<\n{answer}\n>>>"},
    ]
    for _ in range(attempts):
        (reply,) = await model.reply(messages, samples=1, max_tokens=2000, temperature=0.0)
        parsed = _parse(reply)
        if parsed is not None:
            return parsed
    # GOTCHA: an unparseable verdict is recorded as None scores (never usable), not silently dropped,
    # so judge failures can't inflate the usable rate by shrinking the denominator.
    return Judgment(None, None, "judge reply unreadable")


async def judge_many(model: ChatModel, items: list[tuple[str, str, bool]], concurrency: int = 32) -> list[Judgment]:
    """Judge (prompt, answer, cut_off) triples concurrently, preserving input order."""
    limiter = asyncio.Semaphore(concurrency)

    async def one(item: tuple[str, str, bool]) -> Judgment:
        """Judge a single item under the concurrency limit."""
        async with limiter:
            return await judge(model, *item)

    return list(await asyncio.gather(*(one(item) for item in items)))
