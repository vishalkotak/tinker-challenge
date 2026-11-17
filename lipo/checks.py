"""Text checks that decide whether an answer is a real lipogram (no letter e) and not a reward hack."""

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
SYSTEM_DICTIONARY = Path("/usr/share/dict/words")
WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# GOTCHA: these thresholds are the reward gates; loosening them lets RL learn "th" for "the"
# or one-word answers, which Wu's runs and the judge-hacking papers both saw.
MIN_WORDS = 12
MAX_UNKNOWN_WORD_SHARE = 0.05
MAX_REPEATED_NGRAM_SHARE = 0.30
NGRAM_SIZE = 4


@dataclass(frozen=True)
class CheckResult:
    """Outcome of every check on one answer; `passes_gates` is the single yes/no the reward uses."""

    e_free: bool
    ascii_only: bool
    word_count: int
    unknown_words: tuple[str, ...]
    dropped_e_words: tuple[str, ...]
    repeated_ngram_share: float
    e_count: int

    @property
    def unknown_word_share(self) -> float:
        """Fraction of words that are neither real English words nor names taken from the prompt."""
        return len(self.unknown_words) / max(self.word_count, 1)

    @property
    def passes_gates(self) -> bool:
        """True only if the answer is e-free, ASCII, long enough, real words, and not looping."""
        return (
            self.e_free
            and self.ascii_only
            and self.word_count >= MIN_WORDS
            and not self.dropped_e_words
            and self.unknown_word_share <= MAX_UNKNOWN_WORD_SHARE
            and self.repeated_ngram_share <= MAX_REPEATED_NGRAM_SHARE
        )


@lru_cache(maxsize=1)
def common_words() -> frozenset[str]:
    """Load the 10k most common English words, lowercased."""
    return frozenset(line.strip().lower() for line in (DATA_DIR / "common_words.txt").open() if line.strip())


@lru_cache(maxsize=1)
def known_words() -> frozenset[str]:
    """Load the macOS system dictionary plus the common-word list, lowercased."""
    words = set(common_words())
    if SYSTEM_DICTIONARY.exists():
        words.update(line.strip().lower() for line in SYSTEM_DICTIONARY.open() if line.strip())
    return frozenset(words)


def is_e_free(text: str) -> bool:
    """Return True if the text is non-empty and has no e or E, including accented forms like é."""
    # GOTCHA: NFKD splits "é" into "e" plus an accent, so this also catches accented e's
    # that a plain `"e" in text` check would miss.
    normalized = unicodedata.normalize("NFKD", text).casefold()
    return bool(text.strip()) and "e" not in normalized


def is_ascii_only(text: str) -> bool:
    """Return True if every character is plain ASCII, which blocks homoglyphs like Cyrillic 'е'."""
    return all(ord(character) < 128 for character in text)


def count_e(text: str) -> int:
    """Count e and E characters after accent normalization."""
    return unicodedata.normalize("NFKD", text).casefold().count("e")


INFLECTION_SUFFIXES = ("ings", "ing", "ly", "s")


def _is_real_word(word: str) -> bool:
    """Return True if a lowercase word, or its stem after a simple suffix, is a real English word."""
    # GOTCHA: the system dictionary contains junk like "th" and "ar", so short words must
    # also be in the common list; otherwise "th" (a dropped-e "the") would count as real.
    # Three-letter words like "hid" and "nod" are too rare for the 10k list, so they use the dictionary.
    if len(word) <= 2:
        return word in common_words()
    if len(word) == 3:
        return word in known_words()
    if word in known_words():
        return True
    # GOTCHA: the macOS dictionary has no plurals or -ing forms ("scatters", "bounces"), so
    # strip one suffix and retry; e-free text can't use -ed/-es, so those aren't needed.
    for suffix in INFLECTION_SUFFIXES:
        stem = word[: -len(suffix)]
        if not word.endswith(suffix) or len(stem) < 3:
            continue
        # GOTCHA: "flipping" has a doubled consonant, so also try the stem minus one letter ("flip").
        candidates = (stem, stem + "e", stem[:-1] if stem[-1] == stem[-2] else stem)
        # GOTCHA: "rallies" strips to "rallie"; the -ies plural comes from a -y word ("rally").
        if word.endswith("ies"):
            candidates = candidates + (word[:-3] + "y",)
        if any(candidate in known_words() for candidate in candidates):
            return True
    return False


def _restores_with_e(word: str) -> bool:
    """Return True if inserting one 'e' somewhere turns the word into a common English word."""
    return any((word[:position] + "e" + word[position:]) in common_words() for position in range(len(word) + 1))


def _mid_sentence_names(answer: str) -> set[str]:
    """Return lowercase capitalized words that don't start a sentence, treated as proper names."""
    # GOTCHA: sentence-initial words are excluded, or capitalizing gibberish at the start of
    # each sentence would let it pass as names.
    names = set()
    for match in re.finditer(r"(?<![.!?\n]\s)(?<!^)\b([A-Z][a-z]+)", answer):
        preceding = answer[: match.start()].rstrip()
        if preceding and preceding[-1] not in ".!?:\n\"":
            names.add(match.group(1).lower())
    return names


def _repeated_ngram_share(words: list[str]) -> float:
    """Fraction of word n-grams that already appeared earlier in the answer."""
    ngrams = [tuple(words[index : index + NGRAM_SIZE]) for index in range(len(words) - NGRAM_SIZE + 1)]
    if not ngrams:
        return 0.0
    return 1 - len(set(ngrams)) / len(ngrams)


def check_answer(answer: str, prompt: str = "") -> CheckResult:
    """Run every lipogram and anti-hack check on one answer to one prompt."""
    words = [word.lower() for word in WORD_PATTERN.findall(answer)]
    # GOTCHA: names from the prompt (e.g. "SNCF", "Shakespeare") are allowed even if not in
    # the dictionary, so a faithful answer isn't marked unknown for repeating the user's terms.
    allowed_names = {word.lower() for word in WORD_PATTERN.findall(prompt)} | _mid_sentence_names(answer)
    # GOTCHA: "didn't" splits to "didn", so drop the trailing n of an n't contraction.
    stems = [word[:-3] if word.endswith("n't") else word.split("'")[0] for word in words]
    unknown = [word for word, stem in zip(words, stems) if stem not in allowed_names and not _is_real_word(stem)]
    # GOTCHA: only non-words count as dropped-e; "hid" and "slid" restore to "hide"/"slide" but
    # are real words, and flagging them rejected 64 of Wu's 200 hand-verified answers.
    dropped = [word for word in unknown if _restores_with_e(word.split("'")[0])]
    return CheckResult(
        e_free=is_e_free(answer),
        ascii_only=is_ascii_only(answer),
        word_count=len(words),
        unknown_words=tuple(unknown),
        dropped_e_words=tuple(dropped),
        repeated_ngram_share=_repeated_ngram_share(words),
        e_count=count_e(answer),
    )
