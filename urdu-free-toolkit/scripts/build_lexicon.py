#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build ``data/urdu_lexicon.json.gz`` -- the "bundled breadth layer" lexicon
consumed by ``transliterate.py``'s tier 2 lookup (see ``_load_lexicon`` /
``_LEXICON`` there).

Source: en.wiktionary.org, "Category:Urdu lemmas" (CC BY-SA 4.0 + GFDL --
see ``data/SOURCES.md`` for full attribution). This script does NOT touch
rekhta.org or any other copyrighted-poetry site -- see the module docstring
in ``transliterate.py`` and ``data/SOURCES.md`` for why.

WHAT IT DOES
------------
1. Enumerates Urdu headwords via the MediaWiki API's
   ``generator=categorymembers`` over "Category:Urdu lemmas", keeping only
   single-token entries (no spaces) -- multi-word phrases can never be
   looked up by this app's ``transliterate_word_with``, which only ever
   looks up one contiguous run of Perso-Arabic characters at a time, so a
   space-containing dictionary key would simply never hit.
2. For each headword, fetches the rendered page HTML
   (``action=parse&prop=text``) and extracts the Latin-script
   transliteration span from the Urdu section's OWN headword line
   (``<span class="headword-line">...<span class="headword-tr tr Latn">``).
   Scoping strictly to the headword-line is important: earlier, cruder
   attempts that grabbed the first ``tr Latn`` span anywhere in the Urdu
   section frequently picked up a *synonym's* or *etymology ancestor's*
   transliteration instead of the entry's own (e.g. it returned "hurriyat"
   for the entry "آزادی" -- a synonym cross-reference -- rather than
   "āzādī", the word's own romanisation).
3. Normalises Wiktionary's academic Urdu-Latin transliteration scheme
   (macrons, nasalisation tildes, "ś" for ش, "x"/"c" for خ/چ, dotted/underlined
   emphatics, ...) down to this project's plain-ASCII Roman convention
   (long vowels doubled as "aa"/"ii"/"oo", "kh"/"gh"/"sh"/"ch" digraphs,
   nasalisation folded to a trailing "n") -- see ``normalize_roman()``.
4. Mechanically derives a Devanagari spelling from that normalised Roman
   string with a small deterministic Roman->Devanagari syllabifier (see
   ``roman_to_devanagari()``). We deliberately do NOT derive Devanagari
   from the raw Urdu spelling: Urdu's unwritten short vowels are exactly
   as ambiguous for Devanagari as they are for Roman (both need an
   explicit vowel choice), so the already-vowel-resolved Roman string is
   the more reliable source -- e.g. the raw-Urdu character fallback turns
   کتاب into "कताब" (kataab, wrong), whereas converting the
   correct Roman "kitab" gives "किताब".
5. Writes ``{urdu_spelling: [devanagari, roman]}`` to a gzip JSON file.

This is a reasonable, deterministic fallback for the Devanagari field, not
a linguistically perfect one -- retroflex/dental and nukta/non-nukta
distinctions that Roman collapses stay collapsed in the mechanical
Devanagari too (see the docstring of ``roman_to_devanagari`` for specifics).

RESUMABILITY
------------
Progress is checkpointed to ``--checkpoint`` (default
``data/.lexicon_checkpoint.json``) after every page, so an interrupted run
can just be re-launched with the same arguments to continue where it left
off. Re-run later with a higher ``--limit`` to grow the lexicon
incrementally; already-checkpointed titles are never re-fetched unless
``--no-resume`` is passed. The category-member title list is itself cached
to ``--titles-cache`` (default ``data/.lexicon_titles.json``) so growing
``--limit`` doesn't require re-enumerating the category from scratch.

USAGE
-----
    python scripts/build_lexicon.py --limit 6000
    python scripts/build_lexicon.py --limit 12000   # resumes, fetches more
    python scripts/build_lexicon.py --no-resume --limit 500   # start over

Polite to the API: single-threaded, paced with --delay (default 1.0s)
between page fetches, with exponential backoff + Retry-After handling on
429/5xx responses. No personal contact info is sent -- see USER_AGENT.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_URL = "https://en.wiktionary.org/w/api.php"
# Generic, descriptive User-Agent -- no personal email/contact info, per
# Wiktionary API etiquette and this project's own privacy constraint.
USER_AGENT = "UrduFreeToolkitLexiconBot/1.0 (offline lexicon builder; github.com repo)"
CATEGORY = "Category:Urdu lemmas"

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "data" / "urdu_lexicon.json.gz"
DEFAULT_CHECKPOINT = ROOT / "data" / ".lexicon_checkpoint.json"
DEFAULT_TITLES_CACHE = ROOT / "data" / ".lexicon_titles.json"

SINGLE_WORD_RE = re.compile(r"^[؀-ۿ]+$")

# ---------------------------------------------------------------------------
# 1. Polite, retrying MediaWiki API client
# ---------------------------------------------------------------------------


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def api_get(session: requests.Session, params: dict, max_retries: int = 8,
            base_delay: float = 4.0) -> dict:
    """GET the MediaWiki API with exponential backoff on 429/5xx/network
    errors, honouring a Retry-After header when present."""
    params = {**params, "format": "json"}
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = session.get(API_URL, params=params, timeout=30)
        except requests.RequestException as exc:
            last_exc = exc
            wait = base_delay * (attempt + 1)
            print(f"  network error ({exc}); retrying in {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
            continue
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else base_delay * (attempt + 2)
            print(f"  429 rate-limited; backing off {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
            continue
        if resp.status_code >= 500:
            wait = base_delay * (attempt + 1)
            print(f"  {resp.status_code} server error; retrying in {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
            continue
        try:
            resp.raise_for_status()
            return resp.json()
        except (requests.HTTPError, ValueError) as exc:
            last_exc = exc
            wait = base_delay * (attempt + 1)
            print(f"  error ({exc}); retrying in {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
            continue
    raise RuntimeError(f"giving up after {max_retries} retries ({last_exc}): {params}")


def fetch_category_titles(session: requests.Session, category: str = CATEGORY,
                           page_size: int = 500) -> list[str]:
    """Enumerate all page titles in a category via generator=categorymembers,
    paginating through gcmcontinue."""
    titles: list[str] = []
    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": category,
        "gcmlimit": page_size,
        "gcmnamespace": 0,
    }
    cont: dict = {}
    while True:
        data = api_get(session, {**params, **cont})
        pages = data.get("query", {}).get("pages", {})
        for p in pages.values():
            titles.append(p["title"])
        if "continue" in data:
            cont = {"gcmcontinue": data["continue"]["gcmcontinue"]}
            print(f"  ...{len(titles)} titles so far", file=sys.stderr)
            time.sleep(0.4)
        else:
            break
    return titles


# ---------------------------------------------------------------------------
# 2. Headword-line transliteration extraction
# ---------------------------------------------------------------------------

_H2_SPLIT_RE = re.compile(r'(<div class="mw-heading mw-heading2">.*?</div>)')
_H2_ID_RE = re.compile(r'<div class="mw-heading mw-heading2"><h2 id="([^"]+)"')
_HEADWORD_TR_RE = re.compile(r'class="headword-tr tr Latn"[^>]*>([^<]*)<')
_GENERIC_UR_TR_RE = re.compile(r'lang="ur-Latn"[^>]*class="[^"]*tr Latn"[^>]*>([^<]*)<')


def _urdu_section_html(page_html: str) -> str:
    """Split the full rendered page into per-language (h2) chunks and return
    just the Urdu one. Pages can have many language sections (a word shared
    with Persian/Arabic/Punjabi/etc.); we must not read past Urdu's own."""
    parts = _H2_SPLIT_RE.split(page_html)
    sections: dict[str, str] = {}
    cur = None
    buf: list[str] = []
    for part in parts:
        m = _H2_ID_RE.match(part)
        if m:
            if cur is not None:
                sections[cur] = "".join(buf)
            cur = m.group(1)
            buf = []
        else:
            buf.append(part)
    if cur is not None:
        sections[cur] = "".join(buf)
    for key, html in sections.items():
        if key.startswith("Urdu"):
            return html
    return ""


def extract_headword_translit(page_html: str) -> Optional[str]:
    """Pull the Latin-script transliteration out of the Urdu section's own
    headword line (its *first* headword-line, if a word has multiple POS
    sections -- e.g. noun and verb -- we take the first one).

    Scoped strictly inside ``<span class="headword-line">...</span>`` (up
    to the next ``</p>``) so we never pick up a synonym's, translation's,
    or etymology-ancestor's transliteration elsewhere in the section --
    those use different span classes (``mention-tr tr Latn`` etc.) but a
    looser regex over the whole section can still be fooled by templates
    that legitimately reuse the ``tr Latn`` class for a *different* word.
    """
    section = _urdu_section_html(page_html)
    if not section:
        return None
    idx = section.find('class="headword-line"')
    if idx == -1:
        return None
    window = section[idx:idx + 4000]
    end = window.find("</p>")
    if end != -1:
        window = window[:end]
    m = _HEADWORD_TR_RE.search(window)
    if not m:
        m = _GENERIC_UR_TR_RE.search(window)
    if not m:
        return None
    tr = m.group(1).strip()
    return tr or None


# ---------------------------------------------------------------------------
# 3. Wiktionary Roman -> this project's plain-ASCII Roman convention
# ---------------------------------------------------------------------------
# transliterate.py's plain style spells long vowels by doubling
# (aa/ii/oo -- see _diacritize_curated's inverse mapping in transliterate.py)
# and doesn't distinguish retroflex/dental or nukta/non-nukta consonants in
# Roman. Wiktionary's Module:ur-translit scheme uses IAST-ish macrons/
# diacritics (ā ī ū), a combining tilde for nasalisation, "ś" for
# ش (sh), and bare ASCII "x"/"c" for خ/چ (kh/ch). This maps one onto
# the other.

_COMBINING_MACRON = "̄"
_COMBINING_TILDE = "̃"
_COMBINING_ACUTE = "́"
_COMBINING_CARON = "̌"
_COMBINING_DOT_ABOVE = "̇"

_ACUTE_MAP = {"s": "sh"}                       # ś (ش)
_CARON_MAP = {"s": "sh", "z": "zh", "c": "ch"}  # š ž č (seen defensively; not
                                                 # confirmed in the live sample, but these are
                                                 # the standard IAST-ish equivalents)
_DOT_ABOVE_MAP = {"g": "gh"}                    # ġ (غ); dot-above on "n" (ṅ) falls
                                                 # through to plain "n", which is already right


def _convert_letter(base: str, marks: set) -> str:
    lb = base.lower()
    has_macron = _COMBINING_MACRON in marks
    has_tilde = _COMBINING_TILDE in marks
    if lb in "aiu" and has_macron:
        long_form = {"a": "aa", "i": "ii", "u": "oo"}[lb]
        return long_form + "n" if has_tilde else long_form
    if lb in "aeiou" and has_tilde:
        return lb + "n"
    if _COMBINING_ACUTE in marks and lb in _ACUTE_MAP:
        return _ACUTE_MAP[lb]
    if _COMBINING_CARON in marks and lb in _CARON_MAP:
        return _CARON_MAP[lb]
    if _COMBINING_DOT_ABOVE in marks and lb in _DOT_ABOVE_MAP:
        return _DOT_ABOVE_MAP[lb]
    # bare (unmarked) ASCII letters Wiktionary's Urdu scheme uses directly
    # for consonants this app spells as digraphs:
    if lb == "x":   # خ
        return "kh"
    if lb == "c":   # چ
        return "ch"
    # any other combining mark (dot-below ̣ for ṭ/ḍ/ṛ/ḥ/ṣ,
    # line-below ̱, etc.) is dropped, keeping the plain base letter --
    # this collapses retroflex/emphatic distinctions Roman already doesn't
    # carry elsewhere in this app's plain style.
    return lb


def normalize_roman(raw: str) -> Optional[str]:
    """Convert a Wiktionary headword-line transliteration to this project's
    plain-ASCII Roman convention. Returns None if nothing usable survives."""
    if not raw:
        return None
    decomposed = unicodedata.normalize("NFD", raw)
    out: list[str] = []
    i = 0
    n = len(decomposed)
    while i < n:
        ch = decomposed[i]
        if unicodedata.combining(ch):
            i += 1  # stray combining mark with no base -- drop
            continue
        j = i + 1
        marks = set()
        while j < n and unicodedata.combining(decomposed[j]):
            marks.add(decomposed[j])
            j += 1
        out.append(_convert_letter(ch, marks))
        i = j
    result = "".join(out).lower()
    result = re.sub(r"[^a-z]", "", result)
    return result or None


# ---------------------------------------------------------------------------
# 4. Plain Roman -> Devanagari (mechanical, deterministic)
# ---------------------------------------------------------------------------
# A small syllabifier: consonant (+virama if followed by another consonant,
# nothing extra if word-final) + optional vowel matra, or an independent
# vowel form when a vowel isn't preceded by a consonant. This does not
# distinguish dental/retroflex (t/T, d/D, r/R) or nukta/non-nukta (k/q,
# g/gh-nukta, kh-aspirate/kh-nukta, z variants, f/ph, s/sh) consonants --
# those distinctions don't survive in this app's plain Roman convention
# either, so there is no reliable signal left to recover them from. This
# mirrors how the app's own COMMON_WORDS curated dict is often nukta-free
# too (e.g. "ghulaam" -> गुलाम, not गुलाघ्).

_CONSONANT_MAP = {
    "chh": "छ", "kh": "ख", "gh": "घ", "ch": "च", "jh": "झ",
    "th": "थ", "dh": "ध", "ph": "फ", "bh": "भ", "sh": "श",
    "zh": "झ",
    "k": "क", "g": "ग", "j": "ज", "t": "त", "d": "द",
    "n": "न", "p": "प", "b": "ब", "m": "म", "y": "य",
    "r": "र", "l": "ल", "v": "व", "w": "व", "s": "स",
    "h": "ह", "z": "ज़", "f": "फ़", "q": "क़",
}
_VOWEL_MATRA = {
    "aa": "ा", "ii": "ी", "oo": "ू", "ai": "ै", "au": "ौ",
    "a": "", "i": "ि", "u": "ु", "e": "े", "o": "ो",
}
_VOWEL_INDEP = {
    "aa": "आ", "ii": "ई", "oo": "ऊ", "ai": "ऐ", "au": "औ",
    "a": "अ", "i": "इ", "u": "उ", "e": "ए", "o": "ओ",
}
_CONS_KEYS_BY_LEN = sorted(_CONSONANT_MAP, key=len, reverse=True)
_VOWEL_KEYS_BY_LEN = sorted(_VOWEL_MATRA, key=len, reverse=True)


def _tokenize_roman(word: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    i, n = 0, len(word)
    while i < n:
        for key in _CONS_KEYS_BY_LEN:
            if word.startswith(key, i):
                tokens.append(("C", key))
                i += len(key)
                break
        else:
            for key in _VOWEL_KEYS_BY_LEN:
                if key and word.startswith(key, i):
                    tokens.append(("V", key))
                    i += len(key)
                    break
            else:
                tokens.append(("X", word[i]))
                i += 1
    return tokens


def roman_to_devanagari(word: str) -> Optional[str]:
    """Deterministically syllabify a plain-ASCII Roman word (as produced by
    ``normalize_roman``) into Devanagari. See module docstring point 4 for
    why this is derived from Roman rather than the raw Urdu spelling."""
    if not word:
        return None
    tokens = _tokenize_roman(word)
    out: list[str] = []
    n = len(tokens)
    i = 0
    while i < n:
        typ, val = tokens[i]
        if typ == "C":
            base = _CONSONANT_MAP[val]
            nxt = tokens[i + 1] if i + 1 < n else None
            if nxt is not None and nxt[0] == "V":
                out.append(base + _VOWEL_MATRA[nxt[1]])
                i += 2
            else:
                is_last = i == n - 1
                out.append(base if is_last else base + "्")  # virama
                i += 1
        elif typ == "V":
            out.append(_VOWEL_INDEP[val])
            i += 1
        else:
            out.append(val)
            i += 1
    result = "".join(out)
    return result or None


# ---------------------------------------------------------------------------
# 5. Checkpointing
# ---------------------------------------------------------------------------


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# 6. Main pipeline
# ---------------------------------------------------------------------------


def build(limit: int, delay: float, checkpoint_path: Path,
          titles_cache_path: Path, resume: bool, category: str,
          output_path: Optional[Path] = None) -> dict:
    session = make_session()

    titles = load_json(titles_cache_path, None) if resume else None
    if not titles:
        print("Enumerating category members...", file=sys.stderr)
        titles = fetch_category_titles(session, category)
        save_json(titles_cache_path, titles)
    print(f"{len(titles)} total category members", file=sys.stderr)

    single_word_titles = [t for t in titles
                           if len(t) >= 2 and SINGLE_WORD_RE.match(t)]
    print(f"{len(single_word_titles)} single-word entries after filtering "
          f"phrases/punctuation-only titles", file=sys.stderr)

    checkpoint: dict = load_json(checkpoint_path, {}) if resume else {}
    print(f"resuming with {len(checkpoint)} already-processed titles"
          if checkpoint else "starting fresh", file=sys.stderr)

    # entries already attempted (success OR confirmed-miss) are recorded in
    # checkpoint keyed by title -> [deva, roman] or None (miss, don't retry)
    remaining = [t for t in single_word_titles if t not in checkpoint]

    accepted_count = sum(1 for v in checkpoint.values() if v)
    fetched_this_run = 0
    for title in remaining:
        if accepted_count >= limit:
            break
        try:
            data = api_get(session, {"action": "parse", "page": title, "prop": "text"})
        except RuntimeError as exc:
            print(f"  SKIP {title}: {exc}", file=sys.stderr)
            checkpoint[title] = None
            continue
        page_html = data.get("parse", {}).get("text", {}).get("*", "")
        raw_tr = extract_headword_translit(page_html)
        entry = None
        if raw_tr:
            roman = normalize_roman(raw_tr)
            if roman:
                deva = roman_to_devanagari(roman)
                if deva:
                    entry = [deva, roman]
        checkpoint[title] = entry
        if entry:
            accepted_count += 1
        fetched_this_run += 1
        if fetched_this_run % 25 == 0:
            save_json(checkpoint_path, checkpoint)
            print(f"  ...{accepted_count} accepted / {fetched_this_run} fetched "
                  f"this run (title: {title} -> {entry})", file=sys.stderr)
            if output_path is not None:
                # write the gz incrementally too, so a long run leaves a
                # genuinely usable (if partial) file at every checkpoint,
                # not just at the very end.
                write_lexicon(_checkpoint_to_lexicon(checkpoint), output_path)
        time.sleep(delay)

    save_json(checkpoint_path, checkpoint)
    print(f"done: {accepted_count} accepted entries total in checkpoint "
          f"({fetched_this_run} fetched this run)", file=sys.stderr)

    return _checkpoint_to_lexicon(checkpoint)


def _checkpoint_to_lexicon(checkpoint: dict) -> dict:
    return {
        unicodedata.normalize("NFC", title): entry
        for title, entry in checkpoint.items()
        if entry
    }


def write_lexicon(lexicon: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8") as fh:
        json.dump(lexicon, fh, ensure_ascii=False, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=6000,
                     help="target number of accepted entries (default: 6000)")
    ap.add_argument("--delay", type=float, default=1.0,
                     help="seconds to sleep between page fetches (default: 1.0)")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                     help=f"output gzip JSON path (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                     help="resumable progress file (default: data/.lexicon_checkpoint.json)")
    ap.add_argument("--titles-cache", type=Path, default=DEFAULT_TITLES_CACHE,
                     help="cached category member list (default: data/.lexicon_titles.json)")
    ap.add_argument("--category", default=CATEGORY,
                     help=f"MediaWiki category to enumerate (default: {CATEGORY!r})")
    ap.add_argument("--no-resume", action="store_true",
                     help="ignore any existing checkpoint/titles cache and start over")
    args = ap.parse_args()

    lexicon = build(
        limit=args.limit,
        delay=args.delay,
        checkpoint_path=args.checkpoint,
        titles_cache_path=args.titles_cache,
        resume=not args.no_resume,
        category=args.category,
        output_path=args.output,
    )
    write_lexicon(lexicon, args.output)
    size = args.output.stat().st_size
    print(f"wrote {len(lexicon)} entries to {args.output} ({size:,} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
