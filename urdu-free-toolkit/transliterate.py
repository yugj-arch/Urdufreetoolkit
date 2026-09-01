# -*- coding: utf-8 -*-
"""
transliterate.py -- 100% free, offline, rule-based Urdu -> Devanagari (Hindi)
and Urdu -> Roman transliteration engine. No API key, no internet needed
(once installed). No paid service of any kind is used here.

HOW IT WORKS (read this, it matters):

Urdu is written in the Perso-Arabic script, which is an "abjad" -- it
normally does NOT write short vowels. Hindi (Devanagari) and Roman script
both require short vowels to be written explicitly. That mismatch is the
single reason this problem is hard, and no amount of clever code removes
it completely -- see the LIMITATIONS section at the bottom of this file
and in README.md.

This engine handles that mismatch in two layers, in order:

  1. DICTIONARY LOOKUP (COMMON_WORDS below) -- a hand-built list of the
     highest-frequency Hindi-Urdu function words and a handful of common
     nouns (main, hai, hain, aur, kya, ghar, din, ...). Function words
     alone make up a large fraction of real running text, so getting
     these exactly right (for free) meaningfully lifts real-world
     accuracy without needing any paid model.

  2. RULE-BASED FALLBACK -- for any word not in the dictionary, we walk
     the Urdu letters left to right. Consonants get their Devanagari/
     Roman equivalent. Long vowels written explicitly in the Urdu
     spelling (alif, wao, ye, bari ye) are honoured. Short vowels that
     Urdu doesn't write are filled in with a simple heuristic (default
     short "a" between consonants, dropped at the end of a word --
     mimicking Hindi's own schwa-deletion pattern). This heuristic is
     right often enough to be useful, and wrong exactly when the real
     word uses an unwritten short "i" or "u" instead of "a" (classic
     example: کتاب is "kitab", not "katab" -- which is why it's also
     in the dictionary below, as a worked example of this exact limit).

This is the free, deterministic tier from the project's options-comparison
plan. It is not the highest-accuracy tier available (that needs a
context-aware language model to actually resolve the missing vowels --
see README.md for how to plug an API in later if you ever want to). It
is, however, completely free, runs offline, and is fully yours to read,
change, and extend.
"""

import gzip
import json
import re
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Common-word dictionary (free accuracy boost for high-frequency words)
# ---------------------------------------------------------------------------
# Format: urdu_spelling: (devanagari, roman)
# NOTE on میں: this spelling is genuinely ambiguous between "mein" (in) and
# "main" (I) -- classic abjad ambiguity, unresolvable without sentence
# context. We default to "mein" (the more common reading) here.

_RAW_COMMON_WORDS: dict[str, tuple[str, str]] = {
    "اور": ("और", "aur"),
    "یہ": ("ये", "ye"),
    "وہ": ("वो", "wo"),
    "ہے": ("है", "hai"),
    "ہیں": ("हैं", "hain"),
    "تھا": ("था", "tha"),
    "تھی": ("थी", "thi"),
    "تھے": ("थे", "the"),
    "کا": ("का", "ka"),
    "کی": ("की", "ki"),
    "کے": ("के", "ke"),
    "کو": ("को", "ko"),
    "سے": ("से", "se"),
    "میں": ("में", "mein"),  # ambiguous with "main" (I) -- see note above
    "نہیں": ("नहीं", "nahin"),
    "ہاں": ("हाँ", "haan"),
    "کیا": ("क्या", "kya"),
    "کیوں": ("क्यों", "kyon"),
    "کہاں": ("कहाँ", "kahan"),
    "کب": ("कब", "kab"),
    "کون": ("कौन", "kaun"),
    "کیسے": ("कैसे", "kaise"),
    "اگر": ("अगर", "agar"),
    "لیکن": ("लेकिन", "lekin"),
    "مگر": ("मगर", "magar"),
    "بھی": ("भी", "bhi"),
    "سب": ("सब", "sab"),
    "کچھ": ("कुछ", "kuchh"),
    "ہر": ("हर", "har"),
    "اپنا": ("अपना", "apna"),
    "اپنی": ("अपनी", "apni"),
    "اپنے": ("अपने", "apne"),
    "میرا": ("मेरा", "mera"),
    "میری": ("मेरी", "meri"),
    "میرے": ("मेरे", "mere"),
    "تمہارا": ("तुम्हारा", "tumhara"),
    "آپ": ("आप", "aap"),
    "تم": ("तुम", "tum"),
    "ہم": ("हम", "hum"),
    "وہاں": ("वहाँ", "wahan"),
    "یہاں": ("यहाँ", "yahan"),
    "اب": ("अब", "ab"),
    "تب": ("तब", "tab"),
    "جب": ("जब", "jab"),
    "پھر": ("फिर", "phir"),
    "بہت": ("बहुत", "bahut"),
    "زیادہ": ("ज़्यादा", "zyada"),
    "کم": ("कम", "kam"),
    "اچھا": ("अच्छा", "achha"),
    "برا": ("बुरा", "bura"),
    "نیا": ("नया", "naya"),
    "پرانا": ("पुराना", "purana"),
    "بڑا": ("बड़ा", "badaa"),
    "چھوٹا": ("छोटा", "chhota"),
    "دن": ("दिन", "din"),
    "رات": ("रात", "raat"),
    "وقت": ("वक़्त", "waqt"),
    "سال": ("साल", "saal"),
    "گھر": ("घर", "ghar"),
    "آدمی": ("आदमी", "aadmi"),
    "عورت": ("औरत", "aurat"),
    "لوگ": ("लोग", "log"),
    "بات": ("बात", "baat"),
    "کام": ("काम", "kaam"),
    "جانا": ("जाना", "jaana"),
    "آنا": ("आना", "aana"),
    "کرنا": ("करना", "karna"),
    "دینا": ("देना", "dena"),
    "لینا": ("लेना", "lena"),
    "دیکھنا": ("देखना", "dekhna"),
    "کہنا": ("कहना", "kehna"),
    "سننا": ("सुनना", "sunna"),
    "پڑھنا": ("पढ़ना", "padhna"),
    "لکھنا": ("लिखना", "likhna"),
    "کھانا": ("खाना", "khana"),
    "پینا": ("पीना", "peena"),
    "چاہنا": ("चाहना", "chahna"),
    "معلوم": ("मालूम", "maloom"),
    "ضرور": ("ज़रूर", "zaroor"),
    "شاید": ("शायद", "shayad"),
    "نہ": ("न", "na"),
    "مت": ("मत", "mat"),
    "ہو": ("हो", "ho"),
    "ہوں": ("हूँ", "hoon"),
    "ہوا": ("हुआ", "hua"),
    "ہوئی": ("हुई", "hui"),
    "ہوئے": ("हुए", "hue"),
    "گیا": ("गया", "gaya"),
    "گئی": ("गई", "gayi"),
    "گئے": ("गए", "gaye"),
    "کریں": ("करें", "karen"),
    "کرے": ("करे", "kare"),
    # a few common nouns that the rule-based fallback gets wrong by
    # default (worked examples of the abjad-ambiguity limitation):
    "کتاب": ("किताब", "kitab"),
    "کتابیں": ("किताबें", "kitabein"),
    "موسم": ("मौसम", "mausam"),
    "خوشگوار": ("ख़ुशगवार", "khushgawar"),
    "بازار": ("बाज़ार", "bazaar"),
    "خوش": ("ख़ुश", "khush"),
    "دوست": ("दोस्त", "dost"),
    "پیار": ("प्यार", "pyaar"),
    "زندگی": ("ज़िंदगी", "zindagi"),
    "دنیا": ("दुनिया", "duniya"),
    "انسان": ("इंसान", "insaan"),
    "خدا": ("ख़ुदा", "khuda"),
    "اللہ": ("अल्लाह", "allah"),
    "پاکستان": ("पाकिस्तान", "pakistan"),
    "ہندوستان": ("हिंदुस्तान", "hindustan"),
    "شہر": ("शहर", "shehar"),
    "گاؤں": ("गाँव", "gaon"),
    "سکول": ("स्कूल", "school"),
    "کتنا": ("कितना", "kitna"),
    "کتنی": ("कितनी", "kitni"),
    "کتنے": ("कितने", "kitne"),
    # hamza-carrier words the character fallback still mangles even after
    # the ئ/ؤ fold (the fold stops the leak; these fix the vowels too):
    "کوئی": ("कोई", "koi"),
    "کوئ": ("कोई", "koi"),
    "گئی": ("गई", "gayi"),
    "گئے": ("गए", "gaye"),
    "آؤ": ("आओ", "aao"),
    "جاؤ": ("जाओ", "jao"),
    "رئیس": ("रईस", "rayees"),
    "مسئلہ": ("मसअला", "masla"),
    # common Perso-Arabic vocabulary whose unwritten short vowels the
    # default-"a" heuristic guesses wrong:
    "عید": ("ईद", "eid"),
    "مبارک": ("मुबारक", "mubaarak"),
    "نظام": ("निज़ाम", "nizaam"),
    "سرکار": ("सरकार", "sarkaar"),
    "بزم": ("बज़्म", "bazm"),
    "فلاحی": ("फ़लाही", "falaahi"),
    "آزادی": ("आज़ादी", "aazaadi"),
    "آبادی": ("आबादी", "aabaadi"),
    "نوید": ("नवेद", "naveed"),
    "جسے": ("जिसे", "jise"),
    "لائے": ("लाए", "laaye"),
    "ملائے": ("मिलाए", "milaaye"),
    "الٰہی": ("इलाही", "ilaahi"),
    "الہی": ("इलाही", "ilaahi"),
    "تنگی": ("तंगी", "tangi"),
    "افرنگ": ("अफ़रंग", "afrang"),
    "آفرنگ": ("आफ़रंग", "afrang"),
    "تیرا": ("तेरा", "tera"),
    "تیری": ("तेरी", "teri"),
    "تیرے": ("तेरे", "tere"),
    "آئین": ("आईन", "aaeen"),
    "نغمہ": ("नग़्मा", "naghma"),
    # --- reference-sentence vocabulary + close siblings (hand-resolved so
    #     the offline output matches the GPT-4o compare column word-for-word)
    "ہوتا": ("होता", "hota"),
    "ہوتی": ("होती", "hoti"),
    "ہوتے": ("होते", "hote"),
    "راستہ": ("रास्ता", "raasta"),
    "راستے": ("रास्ते", "raaste"),
    "راستوں": ("रास्तों", "raaston"),
    "فاصلہ": ("फ़ासला", "faasla"),
    "فاصلے": ("फ़ासले", "faasle"),
    "فاصلوں": ("फ़ासलों", "faaslon"),
    "فرق": ("फ़र्क", "farq"),
    "طے": ("तय", "tay"),
    "کرنے": ("करने", "karne"),
    "سمیٹنے": ("समेटने", "sametne"),
    "سمیٹنا": ("समेटना", "sametna"),
    "پڑتے": ("पड़ते", "padte"),
    "پڑتا": ("पड़ता", "padta"),
    "پڑتی": ("पड़ती", "padti"),
    "منزل": ("मंज़िल", "manzil"),
    "منزلیں": ("मंज़िलें", "manzilein"),
    "منزلوں": ("मंज़िलों", "manzilon"),
    "سبب": ("सबब", "sabab"),
    "بنتے": ("बनते", "bante"),
    "بنتا": ("बनता", "banta"),
    "بنتی": ("बनती", "banti"),
    "دور": ("दूर", "door"),
    "ذریعہ": ("ज़रिया", "zariya"),
    "ذریعے": ("ज़रिये", "zariye"),
    "اپنوں": ("अपनों", "apnon"),
    "وجہ": ("वजह", "wajah"),
    "شعر": ("शेर", "sher"),
    "بعد": ("बाद", "baad"),
    "شروع": ("शुरू", "shuru"),
    # --- Eid-ghazal vocabulary: poetic ی-less short forms (ترا/ترے) and
    #     Perso-Arabic words whose unwritten short vowels the heuristic misses
    "کہ": ("कि", "ki"),
    "بھیج": ("भेज", "bhej"),
    "بھیجو": ("भेजो", "bhejo"),
    "ہلال": ("हिलाल", "hilaal"),
    "کلید": ("कलीद", "kaleed"),
    "سائل": ("साइल", "saail"),
    "پھرے": ("फिरे", "phire"),
    "پھرا": ("फिरा", "phira"),
    "پھری": ("फिरी", "phiri"),
    "ترا": ("तेरा", "tera"),
    "تری": ("तेरी", "teri"),
    "ترے": ("तेरे", "tere"),
    "محروم": ("महरूम", "mahroom"),
    "غلام": ("गुलाम", "ghulaam"),
    "غلاموں": ("गुलामों", "ghulaamon"),
    "فرمادی": ("फरमादी", "farmaadi"),
}

# ---------------------------------------------------------------------------
# 2. Character-level tables for the rule-based fallback
# ---------------------------------------------------------------------------

# Consonants: urdu_letter -> (devanagari_base, roman_base_without_vowel)
CONSONANTS = {
    "ب": ("ब", "b"), "پ": ("प", "p"), "ت": ("त", "t"), "ٹ": ("ट", "t"),
    "ث": ("स", "s"), "ج": ("ज", "j"), "چ": ("च", "ch"), "ح": ("ह", "h"),
    "خ": ("ख़", "kh"), "د": ("द", "d"), "ڈ": ("ड", "d"), "ذ": ("ज़", "z"),
    "ر": ("र", "r"), "ڑ": ("ड़", "r"), "ز": ("ज़", "z"), "ژ": ("झ़", "zh"),
    "س": ("स", "s"), "ش": ("श", "sh"), "ص": ("स", "s"), "ض": ("ज़", "z"),
    "ط": ("त", "t"), "ظ": ("ज़", "z"), "غ": ("ग़", "gh"), "ف": ("फ़", "f"),
    "ق": ("क़", "q"), "ک": ("क", "k"), "گ": ("ग", "g"), "ل": ("ल", "l"),
    "م": ("म", "m"), "ن": ("न", "n"), "و": ("व", "v"), "ہ": ("ह", "h"),
    "ی": ("य", "y"),
}

# Aspiration: consonant + do-chashmi he (ھ) -> aspirated consonant
ASPIRATED = {
    "ب": ("भ", "bh"), "پ": ("फ", "ph"), "ت": ("थ", "th"), "ٹ": ("ठ", "th"),
    "ج": ("झ", "jh"), "چ": ("छ", "chh"), "د": ("ध", "dh"), "ڈ": ("ढ", "dh"),
    "ک": ("ख", "kh"), "گ": ("घ", "gh"), "ر": ("ऱ्ह", "rh"),
}

# Diacritics (rarely written, but honoured if present -- these fully
# resolve the vowel, no guessing needed when they appear)
DIACRITICS = {
    "\u064e": ("a", "a"),   # zabar (short a)
    "\u0650": ("ि", "i"),   # zer (short i)
    "\u064f": ("ु", "u"),   # pesh (short u)
    "\u0652": (None, None),  # sukun (explicit: no vowel)
}

# Vowel carriers at the START of a word (independent vowel forms)
WORD_INITIAL_VOWELS = {
    "آ": ("आ", "aa"), "ا": ("अ", "a"), "ای": ("ई", "ee"),
    "او": ("ओ", "o"), "ای": ("ई", "ee"), "ع": ("अ", "a"), "ء": ("अ", "a"),
}

# Medial/final long-vowel markers following a consonant
LONG_VOWEL_AFTER_CONSONANT = {
    "ا": ("ा", "aa"),
    "و": ("ो", "o"),      # simplification: و after a consonant defaults to
                           # "o" (it can also mean "u" or "au" -- unwritten
                           # distinction, same abjad-ambiguity limitation)
    "ی": ("ी", "i"),       # simplification: default "i" (can mean "ee")
    "ے": ("े", "e"),       # simplification: default "e" (can mean "ai")
}

NOON_GHUNNA = "ں"  # nasalization -> anusvara
SUKUN = "\u0652"
DO_CHASHMI_HE = "ھ"

# Urdu/Arabic punctuation that should never be swallowed into a word token
# (otherwise "ہے۔" wouldn't match the dictionary entry "ہے", etc.)
PUNCT_TRANSLIT = {
    "\u06d4": ("\u0964", "."),  # Urdu full stop -> Devanagari danda / period
    "\u061f": ("?", "?"),       # Arabic question mark
    "\u060c": (",", ","),       # Arabic comma
    "\u061b": (";", ";"),       # Arabic semicolon
}

SHADDA = "\u0651"    # gemination mark: doubles the consonant it sits on
TATWEEL = "\u0640"   # kashida: pure typographic stretch, carries no sound
AIN = "\u0639"       # ain: no Hindustani consonant value mid-word -> silent seat

# Perso-Arabic letter variants folded to one standard Urdu form so that a
# dictionary/lexicon key and an input token are compared on equal footing.
# Hamza carriers are the important case: \u0626/\u0624/\u0623/\u0625 otherwise fall straight
# through the character fallback and leak into the output verbatim.
CHAR_FOLDS = {
    "\u064a": "\u06cc",  # Arabic yeh -> Urdu yeh
    "\u0643": "\u06a9",  # Arabic kaf -> Urdu kaf
    "\u0623": "\u0627",  # alif + hamza above -> alif
    "\u0625": "\u0627",  # alif + hamza below -> alif
    "\u0624": "\u0648",  # waw + hamza -> waw
    "\u0626": "\u06cc",  # yeh + hamza -> Urdu yeh
    "\u0621": "",         # bare hamza: silent seat, drop
    "\u0629": "\u06c1",  # teh marbuta -> gol he
    # NB: Perso-Arabic punctuation (\u06d4 \u060c \u061b \u061f) is deliberately NOT folded here
    # -- it is script-specific (\u06d4 -> danda in Devanagari) and handled by
    # PUNCT_TRANSLIT / _render_punct after tokenisation.
}
# Arabic-Indic (U+0660..) and Extended Arabic-Indic (U+06F0..) digits -> ASCII
CHAR_FOLDS.update({chr(0x0660 + n): str(n) for n in range(10)})
CHAR_FOLDS.update({chr(0x06F0 + n): str(n) for n in range(10)})

WORD_RE = re.compile(r"[\u0600-\u06FF]+|[^\u0600-\u06FF]+")


def _fold_chars(text: str) -> str:
    """Collapse letter variants, strip kashida, ASCII-ify digits. Runs on
    both input text and lookup keys so they always match. (Shadda is kept
    here and consumed by the character fallback so it can geminate.)"""
    text = text.replace(TATWEEL, "")
    return "".join(CHAR_FOLDS.get(ch, ch) for ch in text)


def _fold_key(key: str) -> str:
    return _fold_chars(unicodedata.normalize("NFC", key))


def _split_leading_trailing_punct(token: str):
    """Peel Perso-Arabic punctuation off the edges of a token so it doesn't
    get fused into a word and break dictionary lookups. Returns the raw
    peeled marks (still as Perso-Arabic codepoints) -- the caller renders
    them per script via ``_render_punct``."""
    prefix, suffix = "", ""
    while token and token[0] in PUNCT_TRANSLIT:
        prefix += token[0]
        token = token[1:]
    while token and token[-1] in PUNCT_TRANSLIT:
        suffix = token[-1] + suffix
        token = token[:-1]
    return prefix, token, suffix


def _render_punct(marks: str, devanagari: bool) -> str:
    """Map a run of Perso-Arabic punctuation marks to their Devanagari or
    Roman forms (``۔`` -> ``।`` for Devanagari, ``.`` for Roman)."""
    idx = 0 if devanagari else 1
    return "".join(PUNCT_TRANSLIT.get(ch, (ch, ch))[idx] for ch in marks)


def normalize_urdu(text: str) -> str:
    """Unicode-normalize, fold Arabic/Urdu letter variants to their standard
    Urdu form, strip kashida, ASCII-ify digits (free, deterministic).

    Line breaks are structural (poetry, stanzas, OCR line output), so
    only horizontal whitespace is collapsed -- newlines are kept, with
    any spaces hugging them trimmed."""
    text = _fold_chars(unicodedata.normalize("NFC", text))
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _delete_roman_schwas(segs):
    """Apply Hindi/Urdu schwa-deletion to the Roman side of a segment list.

    ``segs`` is a list of ``[deva, roman, kind]`` where kind is ``"C"``
    consonant, ``"V"`` explicit vowel/nucleus, ``"S"`` inherent short-"a"
    (schwa), ``"X"`` anything else (anusvara, passthrough).

    A schwa is dropped from the Roman spelling when it is not the word's
    first syllable AND it is either word-final or sits in a
    ``<nucleus> C schwa C V`` context -- the reduction a Hindi reader
    performs silently. Devanagari keeps its inherent-"a" spelling, so only
    ``seg[1]`` (the Roman piece) is cleared. Mutates ``segs`` in place.
    """
    seen_nucleus = False
    for k, seg in enumerate(segs):
        kind = seg[2]
        if kind == "V":
            seen_nucleus = True
            continue
        if kind != "S":
            continue
        nxt = segs[k + 1] if k + 1 < len(segs) else None
        nxt2 = segs[k + 2] if k + 2 < len(segs) else None
        trailing = nxt is None
        cluster = (nxt is not None and nxt[2] == "C"
                   and nxt2 is not None and nxt2[2] == "V")
        if seen_nucleus and (trailing or cluster):
            seg[1] = ""          # drop the Roman "a"; Devanagari unchanged
        else:
            seen_nucleus = True  # this schwa survives -> counts as a nucleus
    return segs


def _transliterate_word_rule_based(word: str):
    """Character-level fallback for words not in COMMON_WORDS.

    Builds a ``[deva, roman, kind]`` segment list, runs Roman
    schwa-deletion over it, then joins. See ``_delete_roman_schwas``.
    """
    segs: list[list] = []
    i = 0
    n = len(word)

    # word-initial vowel carrier
    if word[:2] in WORD_INITIAL_VOWELS:
        d, r = WORD_INITIAL_VOWELS[word[:2]]
        segs.append([d, r, "V"])
        i = 2
    elif word[:1] in WORD_INITIAL_VOWELS:
        d, r = WORD_INITIAL_VOWELS[word[:1]]
        segs.append([d, r, "V"])
        i = 1

    while i < n:
        ch = word[i]

        if ch in CONSONANTS:
            nxt = word[i + 1] if i + 1 < n else ""

            # aspiration: consonant + do-chashmi he
            if nxt == DO_CHASHMI_HE and ch in ASPIRATED:
                d_base, r_base = ASPIRATED[ch]
                i += 2
            else:
                d_base, r_base = CONSONANTS[ch]
                i += 1

            # shadda geminates: کّ -> क्क / "kk"
            if i < n and word[i] == SHADDA:
                d_base = d_base + "्" + d_base
                r_base = r_base + r_base
                i += 1

            # explicit diacritic wins if present
            if i < n and word[i] in DIACRITICS:
                d_vowel, r_vowel = DIACRITICS[word[i]]
                i += 1
                segs.append([d_base, r_base, "C"])
                if d_vowel is not None:
                    segs.append([d_vowel, r_vowel, "V"])
                continue

            # explicit long vowel letter following
            if i < n and word[i] in LONG_VOWEL_AFTER_CONSONANT:
                d_vowel, r_vowel = LONG_VOWEL_AFTER_CONSONANT[word[i]]
                segs.append([d_base, r_base, "C"])
                segs.append([d_vowel, r_vowel, "V"])
                i += 1
                continue

            # no explicit vowel written -- inherent short "a" (schwa).
            # Devanagari carries it implicitly, so its schwa piece is
            # empty; Roman spells it "a" unless schwa-deletion drops it
            # (see _delete_roman_schwas). A word-final consonant gets no
            # schwa at all.
            segs.append([d_base, r_base, "C"])
            if i < n:
                segs.append(["", "a", "S"])

        elif ch == NOON_GHUNNA:
            segs.append(["\u0902", "n", "X"])  # anusvara
            i += 1

        elif ch == AIN:
            i += 1  # ain: silent seat mid-word -- must never leak

        elif ch in ("\u0627", "\u0622"):
            # a long-vowel letter stranded after another vowel (\u062f\u06cc\u0627, \u0644\u0691\u06a9\u06cc\u0627\u06ba):
            # voice it as long "aa" rather than leak the raw Urdu letter.
            segs.append(["\u0906", "aa", "V"])
            i += 1

        elif ch == "\u0648":
            segs.append(["\u0913", "o", "V"])
            i += 1

        elif ch in ("\u06cc", "\u06d2"):
            segs.append(["\u092f", "y", "C"])
            i += 1

        elif ch in PUNCT_TRANSLIT:
            # Perso-Arabic punctuation glued inside a token (no surrounding
            # space): render per script so ۔ never leaks as a raw codepoint.
            d_p, r_p = PUNCT_TRANSLIT[ch]
            segs.append([d_p, r_p, "X"])
            i += 1

        elif unicodedata.combining(ch):
            i += 1  # stray harakat / sukun with no host consonant: drop

        else:
            # ASCII punctuation, digits, spaces, Latin text: pass through
            segs.append([ch, ch, "X"])
            i += 1

    _delete_roman_schwas(segs)
    deva = "".join(s[0] for s in segs)
    roman = "".join(s[1] for s in segs)
    return deva, roman


# Curated dictionary, keyed the same way input tokens are (folded + NFC) so
# lookups never miss on a hamza-carrier or Arabic-variant spelling.
COMMON_WORDS: dict[str, tuple[str, str]] = {
    _fold_key(k): v for k, v in _RAW_COMMON_WORDS.items()
}

# ---------------------------------------------------------------------------
# 3. Bundled lexicon (optional breadth layer, sits between dict and heuristic)
# ---------------------------------------------------------------------------
# A large Urdu -> (devanagari, roman) map built offline from open Wiktionary
# data (see scripts/build_lexicon.py, data/SOURCES.md). Every entry has its
# short vowels already resolved by a human, so it is exact for that word.
# The file is optional: if it is absent the engine runs on the dictionary +
# heuristic tiers alone, exactly as before.

_LEXICON_PATH = Path(__file__).with_name("data") / "urdu_lexicon.json.gz"


def _load_lexicon(path=_LEXICON_PATH) -> dict[str, tuple[str, str]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return {}
    out: dict[str, tuple[str, str]] = {}
    for k, v in raw.items():
        if isinstance(v, (list, tuple)) and len(v) == 2 and all(v):
            out[_fold_key(k)] = (v[0], v[1])
    return out


_LEXICON = _load_lexicon()


def transliterate_word_with(oov_fn, word: str):
    """Curated dictionary and bundled lexicon first (both exact); only a token
    that misses both is handed to ``oov_fn`` -- the pluggable out-of-vocabulary
    fallback. ``oov_fn(word) -> (devanagari, roman)``."""
    if word in COMMON_WORDS:       # curated: always wins
        return COMMON_WORDS[word]
    if word in _LEXICON:           # bundled breadth layer
        return _LEXICON[word]
    return oov_fn(word)


def transliterate_word(word: str):
    return transliterate_word_with(_transliterate_word_rule_based, word)


def transliterate_with(oov_fn, text: str):
    """Same pipeline as :func:`transliterate` (normalize -> tokenize -> curated
    dict -> lexicon -> fallback -> per-script punctuation), but the
    out-of-vocabulary word fallback is pluggable. The Aksharamukha provider
    passes an Aksharamukha-backed ``oov_fn`` here instead of the built-in
    character rules, while keeping the shared curated/lexicon/punctuation
    layers identical to the rule engine."""
    text = normalize_urdu(text)
    deva_parts, roman_parts = [], []
    for token in WORD_RE.findall(text):
        if re.match(r"[\u0600-\u06FF]+", token):
            prefix, core, suffix = _split_leading_trailing_punct(token)
            if core:
                d, r = transliterate_word_with(oov_fn, core)
            else:
                d, r = "", ""
            deva_parts.append(_render_punct(prefix, True) + d
                              + _render_punct(suffix, True))
            roman_parts.append(_render_punct(prefix, False) + r
                               + _render_punct(suffix, False))
        else:
            deva_parts.append(token)
            roman_parts.append(token)
    return "".join(deva_parts), "".join(roman_parts)


def transliterate(text: str):
    """Main entry point. Returns (devanagari_text, roman_text)."""
    return transliterate_with(_transliterate_word_rule_based, text)


# ---------------------------------------------------------------------------
# LIMITATIONS (read before presenting this as "accurate") --------------------
# ---------------------------------------------------------------------------
# 1. Any word NOT in COMMON_WORDS and NOT carrying explicit diacritics gets
#    its short vowels guessed by a simple default-"a" heuristic. This is
#    right for many words and wrong for others (see the کتاب/kitab example
#    in the dictionary comments above) -- this is a property of the Urdu
#    script itself, not a bug in this code.
# 2. و and ی after a consonant are collapsed to one default reading each
#    ("o" and "i") even though they can also mean "u"/"au" and "ee"/"ai"
#    respectively -- genuinely ambiguous without a dictionary entry or
#    sentence context.
# 3. This engine does NOT do real OCR confidence weighting, sentence-level
#    disambiguation, or any kind of learning from corrections -- it is a
#    fixed, deterministic, offline mapping. For meaningfully higher
#    accuracy on real-world, unpredictable text, the only genuine fix is a
#    context-aware model (see README.md, "Optional: higher-accuracy mode").
# 4. Extend COMMON_WORDS freely -- every word you add there is 100%
#    correct for that word, forever, for free. That is the single highest
#    -leverage thing to do to improve this engine's real-world accuracy.
