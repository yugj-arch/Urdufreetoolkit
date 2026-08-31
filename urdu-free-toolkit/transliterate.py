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

import re
import unicodedata

# ---------------------------------------------------------------------------
# 1. Common-word dictionary (free accuracy boost for high-frequency words)
# ---------------------------------------------------------------------------
# Format: urdu_spelling: (devanagari, roman)
# NOTE on میں: this spelling is genuinely ambiguous between "mein" (in) and
# "main" (I) -- classic abjad ambiguity, unresolvable without sentence
# context. We default to "mein" (the more common reading) here.

COMMON_WORDS = {
    "اور": ("और", "aur"),
    "یہ": ("यह", "yeh"),
    "وہ": ("वह", "voh"),
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
    "بڑا": ("बड़ा", "bada"),
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
ARABIC_PUNCT_MAP = {
    "\u06d4": ".",  # Urdu full stop
    "\u061f": "?",  # Arabic question mark
    "\u060c": ",",  # Arabic comma
    "\u061b": ";",  # Arabic semicolon
}

WORD_RE = re.compile(r"[\u0600-\u06FF]+|[^\u0600-\u06FF]+")


def _split_leading_trailing_punct(token: str):
    """Peels Arabic punctuation off the edges of a token so it doesn't
    get fused into a word and break dictionary lookups."""
    prefix, suffix = "", ""
    while token and token[0] in ARABIC_PUNCT_MAP:
        prefix += ARABIC_PUNCT_MAP[token[0]]
        token = token[1:]
    while token and token[-1] in ARABIC_PUNCT_MAP:
        suffix = ARABIC_PUNCT_MAP[token[-1]] + suffix
        token = token[:-1]
    return prefix, token, suffix


def normalize_urdu(text: str) -> str:
    """Unicode-normalize and fold a couple of common Arabic/Urdu character
    variants to their standard Urdu codepoint (free, deterministic)."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u064a", "\u06cc")  # Arabic yeh -> Urdu yeh
    text = text.replace("\u0643", "\u06a9")  # Arabic kaf -> Urdu kaf
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _transliterate_word_rule_based(word: str):
    """Character-level fallback for words not in COMMON_WORDS."""
    deva, roman = "", ""
    i = 0
    n = len(word)

    # word-initial vowel carrier
    if word[:2] in WORD_INITIAL_VOWELS:
        d, r = WORD_INITIAL_VOWELS[word[:2]]
        deva += d
        roman += r
        i = 2
    elif word[:1] in WORD_INITIAL_VOWELS:
        d, r = WORD_INITIAL_VOWELS[word[:1]]
        deva += d
        roman += r
        i = 1

    while i < n:
        ch = word[i]

        if ch in CONSONANTS:
            nxt = word[i + 1] if i + 1 < n else ""
            nxt2 = word[i + 2] if i + 2 < n else ""

            # aspiration: consonant + do-chashmi he
            if nxt == DO_CHASHMI_HE and ch in ASPIRATED:
                d_base, r_base = ASPIRATED[ch]
                i += 2
            else:
                d_base, r_base = CONSONANTS[ch]
                i += 1

            # explicit diacritic wins if present
            if i < n and word[i] in DIACRITICS:
                d_vowel, r_vowel = DIACRITICS[word[i]]
                i += 1
                if d_vowel is None:
                    deva += d_base
                    roman += r_base
                else:
                    deva += d_base + d_vowel
                    roman += r_base + r_vowel
                continue

            # explicit long vowel letter following
            if i < n and word[i] in LONG_VOWEL_AFTER_CONSONANT:
                d_vowel, r_vowel = LONG_VOWEL_AFTER_CONSONANT[word[i]]
                deva += d_base + d_vowel
                roman += r_base + r_vowel
                i += 1
                continue

            # no explicit vowel written -- heuristic:
            # default short "a", but drop it at the very end of the word
            # (mimics Hindi's own schwa-deletion pattern).
            # NOTE: Devanagari consonants already carry an inherent "a"
            # sound by default, so we do NOT append a literal "a" to the
            # Devanagari string here -- only Roman needs it spelled out.
            is_last = (i >= n)
            if is_last:
                deva += d_base
                roman += r_base
            else:
                deva += d_base
                roman += r_base + "a"

        elif ch == NOON_GHUNNA:
            deva += "\u0902"  # anusvara
            roman += "n"
            i += 1

        elif ch in (SUKUN,):
            i += 1  # already consumed by consonant branch normally

        else:
            # punctuation, digits, spaces, anything else: pass through
            # (Urdu full stop -> plain period for readability)
            deva += "." if ch == "\u06d4" else ch
            roman += "." if ch == "\u06d4" else ch
            i += 1

    return deva, roman


def transliterate_word(word: str):
    if word in COMMON_WORDS:
        return COMMON_WORDS[word]
    return _transliterate_word_rule_based(word)


def transliterate(text: str):
    """Main entry point. Returns (devanagari_text, roman_text)."""
    text = normalize_urdu(text)
    deva_parts, roman_parts = [], []
    for token in WORD_RE.findall(text):
        if re.match(r"[\u0600-\u06FF]+", token):
            prefix, core, suffix = _split_leading_trailing_punct(token)
            if core:
                d, r = transliterate_word(core)
            else:
                d, r = "", ""
            deva_parts.append(prefix + d + suffix)
            roman_parts.append(prefix + r + suffix)
        else:
            deva_parts.append(token)
            roman_parts.append(token)
    return "".join(deva_parts), "".join(roman_parts)


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
