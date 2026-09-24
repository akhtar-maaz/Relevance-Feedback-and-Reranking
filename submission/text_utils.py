"""
submission/text_utils.py

Text normalisation used by submission/feedback.py: the same
lowercase-alphanumeric tokeniser as submission/lm_utils.py, plus

  * a small built-in English stopword list (applied to queries and to
    relevance-model expansion terms -- NOT to documents, so document
    lengths / collection statistics stay honest unigram counts), and
  * a from-scratch implementation of the Porter (1980) stemming
    algorithm, with a per-token cache so stemming a 100k+ document
    corpus in prepare() costs one stem() call per *distinct* token.

Standard library only (the grading machine has no internet and no IR
libraries).
"""
import re
from typing import Dict, List

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are aren as at
    be because been before being below between both but by can cannot could
    couldn did didn do does doesn doing don down during each either etc few
    for from further had hadn has hasn have haven having he her here hers
    herself him himself his how however i if in into is isn it its itself
    just let me more most must mustn my myself no nor not now of off on once
    only or other ought our ours ourselves out over own per same shall shan
    she should shouldn so some such than that the their theirs them
    themselves then there these they this those through thus to too under
    until up upon us very via was wasn we were weren what when where whether
    which while who whom whose why will with within without won would
    wouldn yet you your yours yourself yourselves
    """.split()
)


# ---------------------------------------------------------------------------
# Porter stemmer (M.F. Porter, "An algorithm for suffix stripping", 1980).
# Written from the published description of the algorithm.
# ---------------------------------------------------------------------------
_VOWELS = frozenset("aeiou")


def _is_cons(word: str, i: int) -> bool:
    ch = word[i]
    if ch in _VOWELS:
        return False
    if ch == "y":
        return i == 0 or not _is_cons(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """m in [C](VC)^m[V]: the number of vowel->consonant transitions."""
    m = 0
    prev_vowel = False
    for i in range(len(stem)):
        cons = _is_cons(stem, i)
        if cons and prev_vowel:
            m += 1
        prev_vowel = not cons
    return m


def _has_vowel(stem: str) -> bool:
    return any(not _is_cons(stem, i) for i in range(len(stem)))


def _ends_double_cons(word: str) -> bool:
    return len(word) >= 2 and word[-1] == word[-2] and _is_cons(word, len(word) - 1)


def _cvc(word: str) -> bool:
    """*o: stem ends cvc, where the final c is not w, x or y."""
    if len(word) < 3:
        return False
    return (
        _is_cons(word, len(word) - 1)
        and not _is_cons(word, len(word) - 2)
        and _is_cons(word, len(word) - 3)
        and word[-1] not in "wxy"
    )


def _replace(word: str, suffix: str, repl: str, min_m: int) -> str:
    """If word ends with suffix and measure(stem) > min_m, swap suffix."""
    stem = word[: len(word) - len(suffix)]
    if _measure(stem) > min_m:
        return stem + repl
    return word


_STEP2 = [
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
    ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
    ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
]
_STEP3 = [
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
]
_STEP4 = [
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment",
    "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
]


def porter_stem(word: str) -> str:
    if len(word) <= 2:
        return word

    # Step 1a
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2]
    elif word.endswith("ss"):
        pass
    elif word.endswith("s"):
        word = word[:-1]

    # Step 1b
    step1b_extra = False
    if word.endswith("eed"):
        if _measure(word[:-3]) > 0:
            word = word[:-1]
    elif word.endswith("ed") and _has_vowel(word[:-2]):
        word = word[:-2]
        step1b_extra = True
    elif word.endswith("ing") and _has_vowel(word[:-3]):
        word = word[:-3]
        step1b_extra = True
    if step1b_extra:
        if word.endswith(("at", "bl", "iz")):
            word += "e"
        elif _ends_double_cons(word) and word[-1] not in "lsz":
            word = word[:-1]
        elif _measure(word) == 1 and _cvc(word):
            word += "e"

    # Step 1c
    if word.endswith("y") and _has_vowel(word[:-1]):
        word = word[:-1] + "i"

    # Step 2
    for suffix, repl in _STEP2:
        if word.endswith(suffix):
            word = _replace(word, suffix, repl, 0)
            break

    # Step 3
    for suffix, repl in _STEP3:
        if word.endswith(suffix):
            word = _replace(word, suffix, repl, 0)
            break

    # Step 4
    for suffix in _STEP4:
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if _measure(stem) > 1:
                if suffix == "ion":
                    if stem and stem[-1] in "st":
                        word = stem
                else:
                    word = stem
            break

    # Step 5a
    if word.endswith("e"):
        stem = word[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _cvc(stem)):
            word = stem

    # Step 5b
    if _measure(word) > 1 and _ends_double_cons(word) and word.endswith("l"):
        word = word[:-1]

    return word


class Analyzer:
    """tokenize -> (optional) stem, with a cache of stems per surface token.

    The same Analyzer instance must be applied to documents and queries so
    both live in the same term space."""

    def __init__(self, stem: bool = True) -> None:
        self.stem = stem
        self._cache: Dict[str, str] = {}

    def normalize(self, token: str) -> str:
        if not self.stem:
            return token
        cached = self._cache.get(token)
        if cached is None:
            # Numbers and very short tokens are left alone.
            cached = token if (len(token) <= 3 or token.isdigit()) else porter_stem(token)
            self._cache[token] = cached
        return cached

    def analyze(self, text: str) -> List[str]:
        """All tokens of `text` (stopwords kept), normalised."""
        norm = self.normalize
        return [norm(t) for t in _TOKEN_RE.findall(text.lower())]

    def analyze_query(self, text: str) -> List[str]:
        """Query tokens with stopwords removed (falls back to the full
        token list if the query consists only of stopwords)."""
        raw = _TOKEN_RE.findall(text.lower())
        kept = [t for t in raw if t not in STOPWORDS]
        if not kept:
            kept = raw
        return [self.normalize(t) for t in kept]

    def is_stopword_stem(self, term: str) -> bool:
        return term in _STOPWORD_STEMS or term in STOPWORDS


_STOPWORD_STEMS = frozenset(porter_stem(w) if len(w) > 3 else w for w in STOPWORDS)
