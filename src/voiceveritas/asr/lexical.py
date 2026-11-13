"""Pure lexical/alignment helpers over transcribed word lists.

Everything here is a deterministic function of (words, durations) — no
models, no network — which keeps the ASR feature layer fully unit-testable.

Grounding in the literature (and the project plan): spontaneous speech is
marked by fillers ("uh", "um"), mid-syntactic planning pauses after
function words, repetitions and contractions; read speech carries
written-form vocabulary (longer, rarer words, fewer contractions) and
pauses aligned to clause boundaries.
"""

from __future__ import annotations

import itertools
import re

from voiceveritas.asr.transcribe import WordTimes

# hesitation vocatives (core, unambiguous)
CORE_FILLERS = frozenset({"uh", "um", "erm", "er", "ah", "hmm", "umm", "uhh"})
# discourse markers — common in spontaneous planning, rare in prepared text
DISCOURSE_MARKERS = frozenset(
    {"like", "well", "basically", "actually", "anyway", "anyhow", "sorta", "kinda"}
)
MARKER_BIGRAMS = frozenset({"you know", "i mean", "you see", "i guess", "i think"})

# closed-class approximation (no parser dependency): determiners,
# prepositions, conjunctions, pronouns, auxiliaries
FUNCTION_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "about",
        "into",
        "and",
        "or",
        "but",
        "nor",
        "so",
        "yet",
        "because",
        "although",
        "while",
        "if",
        "when",
        "where",
        "who",
        "whom",
        "whose",
        "which",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "is",
        "am",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "will",
        "would",
        "can",
        "could",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "not",
        "no",
        "as",
        "than",
        "then",
        "there",
        "here",
        "what",
    }
)

# apostrophe class: straight ' or curly U+2019 (ASR output varies)
_APOS = "['" + "\u2019" + "]"
_CONTRACTION_RE = re.compile(r"\b\w+" + _APOS + r"(s|t|re|ve|ll|d|m)\b", re.IGNORECASE)


def count_core_fillers(words: list[str]) -> int:
    return sum(1 for w in words if w.lower().strip(".,!?") in CORE_FILLERS)


def count_discourse_markers(tokens: list[str]) -> int:
    """Markers incl. bigrams ('you know') — pass raw lowercase tokens."""
    n = sum(1 for t in tokens if t in DISCOURSE_MARKERS)
    bigrams = set(MARKER_BIGRAMS)
    text = " ".join(tokens)
    for bg in bigrams:
        n += (
            text.count(f" {bg} ")
            + (1 if text.startswith(bg + " ") else 0)
            + (1 if text.endswith(" " + bg) else 0)
        )
    return n


def count_contractions(tokens: list[str]) -> int:
    return sum(1 for t in tokens if _CONTRACTION_RE.search(t))


def count_immediate_repetitions(tokens: list[str]) -> int:
    return sum(1 for a, b in itertools.pairwise(tokens) if a == b and len(a) > 1)


def interword_gaps(words: list[WordTimes]) -> list[tuple[float, WordTimes, WordTimes]]:
    """(gap_seconds, left_word, right_word) for consecutive word pairs."""
    return [(b.start - a.end, a, b) for a, b in itertools.pairwise(words) if b.start - a.end > 0.0]


def pause_after_function_ratio(words: list[WordTimes], min_gap_s: float = 0.15) -> float | None:
    """Fraction of inter-word pauses whose *preceding* word is a function word.

    Psycholinguistic proxy for syntactic-pause misalignment: spontaneous
    planning pauses tend to occur mid-phrase, right after function words;
    read speech pauses at clause boundaries, after content words.
    """
    pauses = [(g, a, b) for g, a, b in interword_gaps(words) if g >= min_gap_s]
    if not pauses:
        return None
    after_fn = sum(1 for _g, a, _b in pauses if a.word.lower().strip(".,!?") in FUNCTION_WORDS)
    return after_fn / len(pauses)


def word_rate_stats(
    words: list[WordTimes], max_gap_s: float = 0.6
) -> tuple[float | None, float | None]:
    """(mean, std) of words-per-second within contiguous speech runs."""
    if not words:
        return None, None
    runs: list[list[WordTimes]] = [[words[0]]]
    for prev, cur in itertools.pairwise(words):
        if cur.start - prev.end <= max_gap_s:
            runs[-1].append(cur)
        else:
            runs.append([cur])
    rates = []
    for run in runs:
        dur = run[-1].end - run[0].start
        if dur > 0.3:
            rates.append(len(run) / dur)
    if not rates:
        return None, None
    return float(np_mean(rates)), float(np_std(rates))


def np_mean(xs: list[float] | list[int]) -> float:
    return sum(xs) / len(xs)


def np_std(xs: list[float] | list[int]) -> float:
    mu = np_mean(xs)
    return (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5


def word_edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein word distance (for WER against a reference text)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i]
        for j, h in enumerate(hyp, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def wer_vs_reference(reference_text: str, tokens: list[str]) -> float | None:
    """Word error rate of the transcript against a reference answer text."""
    ref = re.findall(r"[a-z']+", reference_text.lower())
    if not ref:
        return None
    return word_edit_distance(ref, tokens) / len(ref)
