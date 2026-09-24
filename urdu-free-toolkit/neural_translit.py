# -*- coding: utf-8 -*-
"""neural_translit.py -- offline, free Urdu -> Devanagari + Roman transliteration
backed by a trained character-level Transformer (``training/translit/``).

A line the GPT column has already read (``data/translit_model/gpt.json.gz``,
``training/translit/gpt_distill.py``) comes back exactly as GPT wrote it.
Otherwise, per word, most-trusted source first:

  G. GPT's own reading, distilled from its replies on real poetry and prose:
     a (prev, word) / (word, next) context reading, else -- after the
     homograph rules below -- GPT's majority spelling in each script. GPT's
     joins between known word pairs (hyphens, unwritten izafat -e-) too.
  0. sentence-context rules for the few truly ambiguous words (میں
     main/mein, کیا kyā/kiyā, بن ban/bin, سو so/sau, جلد jald/jild,
     گر gir/gar, کل kul/kal)
  1. the exact dictionary (``data/translit_dictionary.tsv``) -- the 10,000
     most frequent Urdu word forms (~92% of running text), each reviewed
     in all three spellings (``training/translit/build_dictionary.py``)
  2. curated high-frequency words (``transliterate.COMMON_WORDS`` + the R
     overrides below)
  3. the gold lexicon -- ~23k Urdu words and inflections from Wiktionary,
     readings chosen with 700k human romanisations as tie-break evidence
  4. the neural model (beam search) for everything else, its beam reranked
     by how ~47k words are actually romanised by people (Aksharantar /
     Dakshina) -- evidence only decides vowels, never doubling

English loanwords and names get their English spelling in plain Roman
(school, station, london -- ~700 from Wiktionary etymologies and glosses);
Devanagari and the Rekhta-style Roman stay phonetic.

Every Roman string is rendered from one canonical "R" reading
(``urdu_nn/scheme.py``), so the Plain (kitaab) and Diacritic
(kitāb) spellings always agree, and the Devanagari is decoded *from* that
reading. Line breaks, punctuation, digits and Latin text pass through
untouched: the engine never drops or invents a line.

Degrades gracefully: without torch or the model file it still runs on the
dictionary + curated + lexicon tiers, with the rule engine for unknown words.
"""
from __future__ import annotations

import gzip
import json
import math
import re
import threading
import unicodedata
from pathlib import Path

import transliterate as _rule
import difflib

from urdu_nn.scheme import fold_roman, hindi_fold, norm_urdu, to_plain, to_rekhta

MODEL_DIR = Path(__file__).with_name("data") / "translit_model"
MODEL_PATH = MODEL_DIR / "model.pt"
LEXICON_PATH = MODEL_DIR / "lexicon.json.gz"
EVIDENCE_PATH = MODEL_DIR / "evidence.json.gz"
ENGLISH_PATH = MODEL_DIR / "english.json"
DICTIONARY_PATH = _rule.EXACT_DICTIONARY_PATH
GPT_PATH = MODEL_DIR / "gpt.json.gz"
HINDI_FORMS_PATH = MODEL_DIR / "hindi_forms.json.gz"
EVIDENCE_WEIGHT = 2.0      # how far human romanisations may pull the beam (tuned on Dakshina dev)
CASUAL_WEIGHT = 4.0        # ... the model's own casual-Roman beam, for words with none (tuned on gold dev)
HINDI_BONUS = 2.0          # a beam reading that is a real Hindi word (tuned on gold dev)

# R readings for curated words whose Wiktionary-majority reading is a
# different word or a different register (Rekhta-style Roman is rendered from
# these; curated Plain/Devanagari still come from COMMON_WORDS).
CURATED_R = {
    "میں": "me~", "کیا": "kyā", "کیوں": "kyo~", "کہ": "ki", "وہ": "vo", "یہ": "ye",
    "ہم": "ham", "اچھا": "acchā", "بڑا": "baṛā", "چھوٹا": "choṭā", "زیادہ": "zyāda",
    "تو": "to", "نہ": "na", "اس": "is", "ان": "in", "کچھ": "kuch", "دو": "do",
    "ہو": "ho", "ہوں": "hū~", "گئے": "gae", "گئی": "gaī", "ہوئے": "hue", "ہوئی": "huī",
    "لیے": "liye", "لئے": "liye", "کر": "kar", "پر": "par", "جو": "jo",
}

# Arabic article: sun letters assimilate the l (ar-rahmān, as-salām, ad-dīn)
_ART_CONS = {"ت": ("t", "त"), "ث": ("s", "स"), "د": ("d", "द"), "ذ": ("z", "ज़"),
             "ر": ("r", "र"), "ز": ("z", "ज़"), "س": ("s", "स"), "ش": ("sh", "श"),
             "ص": ("s", "स"), "ض": ("z", "ज़"), "ط": ("t", "त"), "ظ": ("z", "ज़"),
             "ل": ("l", "ल"), "ن": ("n", "न")}
_ART_READING = re.compile(r"^a(l|t|s|sh|d|z|r|n)-")

_URDU_RUN = re.compile(r"[؀-ۿݐ-ݿ]+")
_ZER = "ِ"
_SENT_END = set("۔؟!?.\n")
_PASSIVE_NEXT = {"گیا", "گئی", "گئے", "گیی", "جاتا", "جاتی", "جاتے", "جائے", "جا", "جانا"}
_PERFECT_NEXT = {"ہے", "ہیں", "تھا", "تھی", "تھے", "ہوگا", "ہو"}
_VERB_NEXT = _PASSIVE_NEXT | {"جاتا", "جاؤ", "جاؤں", "جائیں", "رہا", "رہی", "رہے", "رہیں",
                              "سکتا", "سکتی", "سکتے", "سکا", "سکی", "سکے", "کر",
                              "چکا", "چکی", "چکے", "گا", "گی", "گے"}
_JALD_NEXT = {"ہی", "از", "سے", "بازی", "ہو", "ہوا", "ہوئی", "ہوئے", "ہوں",
              "سو", "اٹھ", "آ", "جا", "چل", "کر", "پہنچ", "لوٹ", "واپس", "شروع",
              "ختم", "مل", "بن"}
_FALL_NEXT = {"پڑا", "پڑی", "پڑے", "پڑتا", "پڑتی", "پڑتے", "جائے", "جاتا", "جاتی", "جاتے"}
_KUL_NEXT = {"آبادی", "تعداد", "رقبہ", "رقبے", "ملا", "رقم", "مجموعی", "وقتی",
             "جماعتی", "تعدا", "لاگت", "ووٹ", "ووٹوں", "نمبر", "نمبروں", "اثاثے",
             "آمدنی", "خرچ", "مدت", "لمبائی", "اراکین", "ارکان", "افراد", "سیٹوں"}
_MAIN_PREV = {"اور", "کہ", "جب", "اگر", "تو", "مگر", "لیکن", "پھر", "کیونکہ", "بلکہ",
              "اب", "ہاں", "جو"}
_MAIN_NEXT = {"نے", "ہوں", "خود"}

# placeholders resolved when a line is joined: conjunctive و (sher-o-shāyarī)
# and a written izafat (dil-e-nādāñ) both hyphenate onto the next word
_O = "\x01"
_IZAFAT = "\x02"


class NeuralTransliterator:
    def __init__(self, model_path: Path = MODEL_PATH, lexicon_path: Path = LEXICON_PATH,
                 evidence_path: Path = EVIDENCE_PATH, english_path: Path = ENGLISH_PATH,
                 dictionary_path: Path | None = DICTIONARY_PATH,
                 beam: int = 5, device: str = "cpu", evidence_weight: float = EVIDENCE_WEIGHT,
                 gpt_path: Path | None = GPT_PATH,
                 hindi_forms_path: Path | None = HINDI_FORMS_PATH,
                 casual_weight: float = CASUAL_WEIGHT, hindi_bonus: float = HINDI_BONUS):
        self.beam = beam
        self.device = device
        self.evidence_weight = evidence_weight
        self.casual_weight = casual_weight
        self.hindi_bonus = hindi_bonus
        # urdu -> Hindi words that spell it (hindi_fold keys), for beam reranking
        self.hindi_forms: dict[str, set[str]] = {}
        if hindi_forms_path and Path(hindi_forms_path).exists():
            with gzip.open(hindi_forms_path, "rt", encoding="utf-8") as fh:
                self.hindi_forms = {k: set(v) for k, v in json.load(fh).items()}
        # GPT column, distilled: verbatim lines, per-script word readings,
        # context readings ("L\tprev\tw" / "R\tw\tnext") and pair joins
        gpt = {}
        if gpt_path and Path(gpt_path).exists():
            with gzip.open(gpt_path, "rt", encoding="utf-8") as fh:
                gpt = json.load(fh)
        self.gpt_lines: dict[str, list[str]] = gpt.get("lines", {})
        self.gpt_words: dict[str, list[str | None]] = gpt.get("words", {})
        self.gpt_ctx: dict[str, list[str | None]] = gpt.get("ctx", {})
        self.gpt_joins: dict[str, list[str | None]] = gpt.get("joins", {})
        self.gpt_merge: dict[str, list[str | None]] = gpt.get("merge", {})
        self.gpt_future: dict[str, list[float]] = gpt.get("future_merge", {})
        self.gpt_o_join: list[str] = gpt.get("o_join", ["-", "-", "-"])
        # reviewed words: urdu -> (devanagari, plain, rekhta), used verbatim
        self.exact: dict[str, tuple[str, str, str]] = {
            norm_urdu(k): v for k, v in _rule.load_exact_dictionary(dictionary_path).items()}
        self.lexicon: dict[str, tuple[str | None, str]] = {}
        if lexicon_path.exists():
            with gzip.open(lexicon_path, "rt", encoding="utf-8") as fh:
                self.lexicon = {k: (v[0], v[1]) for k, v in json.load(fh).items()}
        # English loanwords / names: casual Roman spells them the English way
        # (school, station, london); Devanagari and Rekhta stay phonetic
        self.english: dict[str, str] = {}
        if english_path and Path(english_path).exists():
            self.english = json.loads(Path(english_path).read_text(encoding="utf-8"))
        self.evidence: dict[str, dict[str, float]] = {}
        if evidence_path and Path(evidence_path).exists():
            with gzip.open(evidence_path, "rt", encoding="utf-8") as fh:
                self.evidence = json.load(fh)
        self.model = None
        self.vocab = None
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str], str | None] = {}   # (task, src) -> decoded text
        if model_path.exists():
            try:
                self._load_model(model_path)
            except Exception:           # torch missing / incompatible file -> lexicon tiers only
                self.model = None

    def _load_model(self, model_path: Path):
        import torch
        from urdu_nn.model import TASK_TOKEN, Seq2Seq, Vocab
        ck = torch.load(model_path, map_location=self.device, weights_only=False)
        if "<h>" not in ck["itos"] or TASK_TOKEN.get("h") != "<h>":
            raise ValueError("model file predates the R|devanagari layout")
        vocab = Vocab([])
        vocab.itos = ck["itos"]
        vocab.stoi = {s: i for i, s in enumerate(vocab.itos)}
        model = Seq2Seq(**ck["cfg"])
        model.load_state_dict({k: v.float() for k, v in ck["state"].items()})
        model.eval()
        self.model, self.vocab = model.to(self.device), vocab
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    @property
    def has_model(self) -> bool:
        return self.model is not None

    # -- model calls ------------------------------------------------------------

    def _decode(self, task: str, sources: list[str], valid) -> dict[str, list[tuple[str, float]]]:
        """Beam-decode ``sources`` with the given task tag -> per source, the
        hypotheses ``valid(text)`` accepts as (text, score), best first.
        Cached across calls."""
        out: dict[str, list[tuple[str, float]]] = {}
        if self.model is None:
            return out
        todo = [s for s in dict.fromkeys(sources)
                if (task, s) not in self._cache and all(c in self.vocab.stoi for c in s)]
        if todo:
            import torch
            from urdu_nn.model import beam_search, pad_batch
            with self._lock, torch.inference_mode():
                for i in range(0, len(todo), 64):
                    chunk = todo[i:i + 64]
                    src = pad_batch([self.vocab.encode_src(task, s) for s in chunk], self.device)
                    for s, hyps in zip(chunk, beam_search(self.model, src, beam=self.beam)):
                        texts = [(self.vocab.decode(ids), sc) for ids, sc in hyps]
                        self._cache[(task, s)] = [(t, sc) for t, sc in texts if valid(t)]
        for s in sources:
            hit = self._cache.get((task, s))
            if hit:
                out[s] = hit
        return out

    def _joint(self, words: list[str], keys: dict[str, str]) -> dict[str, tuple[str, str]]:
        """urdu -> (R, devanagari); ``keys`` maps a model source (which may
        carry harakat) to its plain lookup key for the evidence tables."""
        def ok(t):
            return t.count("|") == 1 and all(t.split("|"))
        beams = self._decode("j", words, ok)
        # a word nobody's romanisation covers: the model's own casual-Roman
        # beam (learnt from 700k human romanisations) stands in for them
        bare = [w for w, h in beams.items() if len(h) > 1 and not self.evidence.get(keys.get(w, w))]
        casual = (self._decode("c", bare, lambda t: bool(t) and "|" not in t)
                  if bare and self.casual_weight else {})
        res = {}
        for w, hyps in beams.items():
            key = keys.get(w, w)
            best = hyps[0][0] if len(hyps) == 1 else self._rerank(
                hyps, self.evidence.get(key), casual.get(w), self.hindi_forms.get(key))
            res[w] = tuple(best.split("|"))
        return res

    def _rerank(self, hyps: list[tuple[str, float]], ev: dict[str, float] | None,
                casual: list[tuple[str, float]] | None, hindi: set[str] | None) -> str:
        """Rescore "R|devanagari" beam hypotheses with
        * agreement with how people romanise the word (``ev``, human; else
          ``casual``, the model's casual-Roman beam as a probability spread),
          compared gemination-blind: casual typists drop doubled consonants,
          so this may choose vowels (dushman vs dashman) but never doubling;
        * a bonus when the Devanagari is a real Hindi word spelling this Urdu
          word (``hindi``: hindi_fold keys) -- Devanagari writes the short
          vowels Urdu leaves out (चक्कियों, not चुकियों, for چکیوں)."""
        refs: list[tuple[str, float]] = []
        if ev:
            total = sum(ev.values())
            scale = self.evidence_weight * min(1.0, total / 2.0) / total
            refs = [(_degeminate(fold_roman(e)), w * scale) for e, w in ev.items()]
        elif casual:
            top = max(s for _, s in casual)
            ps = [(_degeminate(fold_roman(t)), math.exp(s - top)) for t, s in casual]
            z = sum(p for _, p in ps)
            refs = [(t, self.casual_weight * p / z) for t, p in ps]

        def score(h):
            rich, deva = h[0].split("|")
            s = h[1]
            if refs:
                f = _degeminate(fold_roman(to_plain(rich)))
                s += sum(w * difflib.SequenceMatcher(None, f, e).ratio() ** 2 for e, w in refs)
            if hindi and hindi_fold(deva) in hindi:
                s += self.hindi_bonus
            return s
        return max(hyps, key=score)[0]

    def _casual_override(self, key: str, rich: str) -> str | None:
        """Plain-Roman fallback for a model reading that strong human evidence
        rejects outright (names above all: سید is written "syed", never "sed").
        Returns the dominant casual spelling, or None to keep the model's."""
        ev = self.evidence.get(key)
        if not ev:
            return None
        top, w = max(ev.items(), key=lambda kv: kv[1])
        total = sum(ev.values())
        if total < 5 or w / total < 0.6:
            return None
        mine = _degeminate(fold_roman(to_plain(rich)))
        if difflib.SequenceMatcher(None, mine, _degeminate(fold_roman(top))).ratio() >= 0.75:
            return None
        return top

    def _deva_for(self, readings: list[str]) -> dict[str, str]:
        """R reading -> Hindi-orthography Devanagari."""
        return {r: h[0][0] for r, h in
                self._decode("h", readings, lambda t: bool(t) and "|" not in t).items()}

    # -- curated tier -------------------------------------------------------------

    @staticmethod
    def _context(key: str, ctx: dict):
        """Homographs read from the sentence around them -> (devanagari,
        plain, R) or None. ``ctx``: prev / next word (None across a sentence
        boundary), sent_start, ne_before (نے earlier in the sentence)."""
        prev, nxt = ctx["prev"], ctx["next"]
        if key == "میں" and (ctx["sent_start"] or prev in _MAIN_PREV or nxt in _MAIN_NEXT):
            return ("मैं", "main", "mai~")
        if key == "کیا" and not ctx["sent_start"] and (
                nxt in _PASSIVE_NEXT
                or (ctx["ne_before"] and (nxt is None or nxt in _PERFECT_NEXT))):
            return ("किया", "kiya", "kiyā")
        if key == "بن" and nxt in _VERB_NEXT:             # ban gayā / bin (ibn, without)
            return ("बन", "ban", "ban")
        if key == "سو" and nxt in _VERB_NEXT:             # so gayā / sau (hundred)
            return ("सो", "so", "so")
        if key == "جلد" and (nxt in _JALD_NEXT or nxt in _VERB_NEXT):   # jald hī / jild (volume)
            return ("जल्द", "jald", "jald")
        if key == "گر" and (nxt in _VERB_NEXT or nxt in _FALL_NEXT):    # gir gayā / gar (if)
            return ("गिर", "gir", "gir")
        if key == "کل":                                   # kul ābādī (total) / kal (day)
            if nxt in _KUL_NEXT:
                return ("कुल", "kul", "kul")
            return ("कल", "kal", "kal")
        return None

    def _gpt_ctx(self, key: str, ctx: dict) -> list[str | None] | None:
        """GPT's reading of ``key`` next to these neighbours, per script
        (None where GPT's data has no context-specific reading)."""
        res = [None, None, None]
        for c in (self.gpt_ctx.get(f"L\t{ctx['prev']}\t{key}") if ctx["prev"] else None,
                  self.gpt_ctx.get(f"R\t{key}\t{ctx['next']}") if ctx["next"] else None):
            if c:                                   # the next word decides last
                res = [c[s] or res[s] for s in range(3)]
        return res if any(res) else None

    def _gpt_join(self, a: str, b: str) -> list[str | None] | None:
        """How GPT joins the words ``a b`` -> per script " ", "-", "-e-" or
        "+" (written as one word, ``gpt_merge``), or None when GPT's data
        never joined them. A script without data borrows another's join
        (never a "+", which needs that script's own merged spelling)."""
        j = self.gpt_joins.get(f"{a}\t{b}")
        if not j:
            return None
        fill = next((x for x in (j[1], j[2], j[0]) if x and x != "+"), None)
        return [x or fill for x in j]

    def _curated(self, key: str):
        """-> (devanagari, plain, R-or-None) or None. R is None when it still
        has to come from the model."""
        folded = _rule._fold_key(key)
        if folded not in _rule.COMMON_WORDS:
            return None
        deva, plain = _rule.COMMON_WORDS[folded]
        rich = CURATED_R.get(key)
        if rich is None:
            lex = self.lexicon.get(key)
            if lex and _loose(to_plain(lex[1])) == _loose(plain):
                rich = lex[1]
        return deva, plain, rich

    def _article(self, key: str, after_word: bool):
        """ال + a word we know exactly -> (devanagari, plain, rekhta), with the
        article assimilated to a sun letter; "ul" after another word
        (bain ul-aqvāmī, dār ul-hukūmat), "al" at the start of a phrase.
        Devanagari is None when the lexicon has none for the word."""
        if not key.startswith("ال") or len(key) < 4:
            return None
        rest = key[2:]
        if rest in self.exact:
            deva, plain, dia = self.exact[rest]
        else:
            lex = self.lexicon.get(rest)
            if not lex or not lex[1]:
                return None
            deva, plain, dia = lex[0], to_plain(lex[1]), to_rekhta(lex[1])
        cons, cons_d = _ART_CONS.get(rest[0], ("l", "ल"))
        vowel, vowel_d = ("u", "उ") if after_word else ("a", "अ")
        return ((vowel_d + cons_d + "-" + deva) if deva else None,
                f"{vowel}{cons}-{plain}", f"{vowel}{cons}-{dia}")

    @staticmethod
    def _article_after_word(entry: tuple[str, str, str]) -> tuple[str, str, str]:
        """A dictionary reading that is itself an article form (ad-dīn,
        as-salām) takes "u" after another word: nasīr ud-dīn."""
        deva, plain, dia = entry
        if not _ART_READING.match(plain):
            return entry
        return ("उ" + deva[1:] if deva.startswith("अ") else deva, "u" + plain[1:],
                "u" + dia[1:] if dia.startswith("a") else dia)

    # -- public -------------------------------------------------------------------

    def analyze(self, words: list[str]) -> list[tuple[str, str, str | None, str]]:
        """Per standalone word -> (devanagari, plain, R-or-None, rekhta), no
        sentence context. Used to draft the exact dictionary for review."""
        _, keys, plan = self._resolve("\n".join(words), context=False)
        out = []
        for i, p in sorted(plan.items()):
            dia = p[4] or (to_rekhta(p[2]) if p[2] else _rule.transliterate_word(
                _rule._fold_key(keys[i][1]), style="diacritic")[1])
            out.append((p[0], p[1], p[2], dia))
        return out

    def transliterate(self, text: str) -> tuple[str, str, str]:
        """-> (devanagari, roman_plain, roman_diacritic)."""
        if not self.gpt_lines:
            return self._transliterate(text)
        lines = _clean(text).split("\n")
        hits = [self.gpt_lines.get(_line_key(l)) if l.strip() else None for l in lines]
        if not any(hits):
            return self._transliterate(text)
        # lines GPT has read come back verbatim; the rest go through the
        # engine together (one model batch), newline = sentence boundary
        miss = [l for l, h in zip(lines, hits) if l.strip() and not h]
        rows = []
        if miss:
            outs = [s.split("\n") for s in self._transliterate("\n".join(miss))]
            rows = (list(zip(*outs)) if all(len(o) == len(miss) for o in outs)
                    else [self._transliterate(l) for l in miss])
        it = iter(rows)
        out = ([], [], [])
        for line, hit in zip(lines, hits):
            row = hit or (next(it) if line.strip() else ("", "", ""))
            for s in range(3):
                out[s].append(row[s])
        return tuple("\n".join(o) for o in out)

    def _gpt_seps(self, parts: list[str], keys: dict):
        """GPT's joins between Urdu words separated by a single space ->
        (separator index -> (devanagari, plain, rekhta) joiner,
         word index -> per-script replacement, where GPT writes a pair as
         one word: the pair's spelling on the first word, "" on the second)."""
        seps, words = {}, {}
        if not self.gpt_joins:
            return seps, words
        o_word = "و" in self.gpt_words
        order = sorted(keys)
        for a, b in zip(order, order[1:]):
            if b != a + 2 or parts[a + 1] != " " or keys[a][3] or keys[b][0]:
                continue
            ka, kb = keys[a][1], keys[b][1]
            j = self._gpt_join(ka, kb)
            merged = self.gpt_merge.get(f"{ka}\t{kb}") or (None, None, None)
            if j is None and o_word and "و" in (ka, kb):
                j = self.gpt_o_join
            elif j is None and kb in _FUTURE and kb in self.gpt_future:
                # an unseen verb + گا/گے/گی: written as one word as often as
                # GPT does (karega, jaaunga) -- spelled from the verb's reading
                j = ["+" if r >= 0.5 else " " for r in self.gpt_future[kb]]
                merged = [("+", kb)] * 3
            if not j:
                continue
            sep = []
            for s, x in enumerate(j):
                if x == "+" and merged[s] and s not in words.get(a, {}) \
                        and s not in words.get(b, {}):
                    words.setdefault(a, {})[s] = merged[s]
                    words.setdefault(b, {})[s] = ""
                    sep.append("")
                else:
                    sep.append(_JOIN_TEXT.get(x, _JOIN_TEXT[" "])[s])
            seps[a + 1] = tuple(sep)
        return seps, words

    def _transliterate(self, text: str) -> tuple[str, str, str]:
        parts, keys, plan = self._resolve(text)
        seps, merged = self._gpt_seps(parts, keys)
        o_word = "و" in self.gpt_words        # GPT's data reads و as a word
        deva_out, plain_out, dia_out = [], [], []
        for i, part in enumerate(parts):
            if i not in keys:
                sep = seps.get(i, (part, part, part))
                deva_out.append(sep[0])
                plain_out.append(sep[1])
                dia_out.append(sep[2])
                continue
            prefix, key, voc, suffix = keys[i]
            after_digit = bool(i and parts[i - 1][-1:].isdigit() and not prefix)
            if after_digit and self.gpt_words:
                if key == "ء":                  # 1980ء -> GPT keeps just the year
                    deva_out.append(_rule._render_punct(suffix, True))
                    plain_out.append(_rule._render_punct(suffix, False))
                    dia_out.append(_rule._render_punct(suffix, False))
                    continue
                deva_out.append(" ")            # 98فیصد -> "98 feesad", as GPT spaces it
                plain_out.append(" ")
                dia_out.append(" ")
            if key == "و" and 0 < i < len(parts) - 1 and not prefix and not suffix and not o_word:
                d = pl = dia = _O
            else:
                d, pl, rich, _, dia = plan[i]
                if dia is None:
                    dia = to_rekhta(rich) if rich else _rule.transliterate_word(
                        _rule._fold_key(key), style="diacritic")[1]
                # izafat written with a zer / hamza: dil-e-nādāñ, ḳhāna-e-dil
                # (unless GPT's join for this pair already decided it)
                if voc.endswith(_ZER) or key.endswith("ۂ") or key.endswith("ٔ"):
                    if not suffix and i + 1 not in seps:
                        d, pl, dia = d + _IZAFAT, pl + _IZAFAT, dia + _IZAFAT
            if i in merged:                    # GPT writes this word with its neighbour
                m = merged[i]
                d, pl, dia = (_future(m[s], x, s) if isinstance(m.get(s), tuple) else m.get(s, x)
                              for s, x in enumerate((d, pl, dia)))
            deva_out.append(_rule._render_punct(prefix, True) + d + _rule._render_punct(suffix, True))
            plain_out.append(_rule._render_punct(prefix, False) + pl + _rule._render_punct(suffix, False))
            dia_out.append(_rule._render_punct(prefix, False) + dia + _rule._render_punct(suffix, False))
        return (_join("".join(deva_out), "ओ", "ए"), _join("".join(plain_out), "o", "e"),
                _join("".join(dia_out), "o", "e"))

    def _resolve(self, text: str, context: bool = True):
        """Tokenise ``text`` and pick every Urdu word's reading -> (parts,
        keys, plan) where plan[idx] = [devanagari, plain, R, None, rekhta];
        rekhta is None when it is rendered from R."""
        parts = _split(_clean(text))
        order = [i for i, p in enumerate(parts) if _URDU_RUN.fullmatch(p)]

        keys: dict[int, tuple[str, str, str, str]] = {}   # idx -> (prefix, key, voc, suffix)
        for i in order:
            prefix, core, suffix = _rule._split_leading_trailing_punct(parts[i])
            keys[i] = (prefix, norm_urdu(core), norm_urdu(core, keep_harakat=True), suffix)

        # pass 1: decide each word's tier; collect what the model must fill
        plan: dict[int, list] = {}     # idx -> [deva, plain, R, model_src, rekhta]
        done: set[int] = set()         # final already (context rule / dictionary)
        gpt_over: dict[int, list] = {}  # idx -> GPT's spelling for some scripts only
        from_gpt: set[int] = set()      # plain Roman taken from GPT's tables
        over_none = (None, None, None)
        ne_before = False
        for n, i in enumerate(order):
            prefix, key, voc, suffix = keys[i]
            prev_i = order[n - 1] if n else None
            nxt_i = order[n + 1] if n + 1 < len(order) else None
            sent_start = (prev_i is None or _ends_sentence(keys[prev_i][3])
                          or _ends_sentence("".join(parts[prev_i + 1:i]))
                          or _ends_sentence(prefix))
            if sent_start:
                ne_before = False
            nxt_blocked = (nxt_i is None or _ends_sentence(suffix)
                           or _ends_sentence("".join(parts[i + 1:nxt_i])))
            ctx = {"prev": None if sent_start else keys[prev_i][1],
                   "next": None if nxt_blocked else keys[nxt_i][1],
                   "sent_start": sent_start, "ne_before": ne_before}
            if key == "نے":
                ne_before = True
            if not key:
                plan[i] = ["", "", "", None, ""]
                done.add(i)
                continue
            src = voc if voc != key else key      # written harakat help the model
            hit = self._context(key, ctx) if context else None
            # GPT: its context reading beats the homograph rules, which beat
            # its majority reading; a script GPT's data lacks falls through
            over = [None, None, None] if hit else list(self.gpt_words.get(key) or over_none)
            gctx = self._gpt_ctx(key, ctx) if context and self.gpt_ctx else None
            if gctx:
                over = [gctx[s] or over[s] for s in range(3)]
            if all(over):
                plan[i] = [over[0], over[1], None, None, over[2]]
                done.add(i)
                from_gpt.add(i)
                continue
            if any(over):
                gpt_over[i] = over
            if hit:
                plan[i] = [hit[0], hit[1], hit[2], None, None]
                done.add(i)
                continue
            if key in self.exact:
                entry = self.exact[key]
                if not sent_start and key.startswith("ال"):
                    entry = self._article_after_word(entry)
                plan[i] = [entry[0], entry[1], None, None, entry[2]]
                done.add(i)
                continue
            cur = self._curated(key)
            if cur:
                plan[i] = [cur[0], cur[1], cur[2], src if cur[2] is None else None, None]
                continue
            lex = self.lexicon.get(key)
            # after another word, ال is the Arabic construct (dār ul-hukūmat) even
            # when the lexicon lists the fused form; at a phrase start the
            # lexicon wins (الماری = almārī, not al-mārī)
            art = (self._article(key, after_word=not sent_start)
                   if (not lex or not sent_start) else None)
            if art:
                plan[i] = [art[0], art[1], None, None, art[2]]
                if art[0]:
                    done.add(i)
            elif lex:
                plan[i] = [lex[0], None, lex[1], None, None]
            else:
                plan[i] = [None, None, None, src, None]

        src_key = {p[3]: keys[i][1] for i, p in plan.items() if p[3]}
        joint = self._joint(list(src_key), src_key)
        need_deva = [p[2] for p in plan.values() if p[0] is None and p[2]]
        deva_of = self._deva_for(need_deva)

        # pass 2: fill from the model (or the rule engine as last resort)
        for i, p in plan.items():
            if i in done:
                continue
            key = keys[i][1]
            deva, plain, rich, src, dia = p
            if src:
                hit = joint.get(src)
                if plain is not None:           # curated: model supplies only R
                    if hit and _loose(to_plain(hit[0])) == _loose(plain):
                        rich = hit[0]
                    else:
                        rich = _plain_to_rich(plain, key)
                elif hit:
                    rich, deva = hit
                    plain = self._casual_override(key, rich)
            if deva is None and rich:
                deva = deva_of.get(rich)
            if rich:
                rich = _mukhtafi(key, rich)
            if (rich is None and dia is None) or deva is None:
                rd, rp = _rule.transliterate_word(_rule._fold_key(key))
                deva = deva or rd
                if rich is None:
                    plain = plain or rp
            if plain is None and key in self.english:
                plain = self.english[key]
            p[:] = [deva, plain if plain is not None else (to_plain(rich) if rich else ""),
                    rich, None, dia]
        for i, over in gpt_over.items():
            p = plan[i]
            p[0], p[1], p[4] = over[0] or p[0], over[1] or p[1], over[2] or p[4]
            if over[1]:
                from_gpt.add(i)
        if self.gpt_words:                  # GPT's plain Roman writes ڑ as r: laraa, tukre
            for i, p in plan.items():
                if i not in from_gpt and "ڑ" in keys[i][1] and p[1]:
                    p[1] = _retroflex_r(keys[i][1], p[1])
        return parts, keys, plan


# ---------------------------------------------------------------------------

_JOIN_TEXT = {" ": (" ", " ", " "), "-": ("-", "-", "-"), "-e-": ("-ए-", "-e-", "-e-")}
_FUTURE = {"گا": ("गा", "ga", "gā"), "گے": ("गे", "ge", "ge"), "گی": ("गी", "gi", "gī")}


def _retroflex_r(key: str, plain: str) -> str:
    """Plain Roman with ڑ spelled r (GPT's habit: laraa, tukre, chhor)
    instead of d. The Urdu's د / ڈ / ڑ map in order onto the reading's
    d's; if the counts differ (a doubled dd, an English spelling) the
    reading is left alone."""
    urdu_d = [c for c in key if c in "دڈڑ"]
    at = [m.start() for m in re.finditer("d", plain)]
    if len(at) != len(urdu_d):
        return plain
    out = list(plain)
    for pos, c in zip(at, urdu_d):
        if c == "ڑ":
            out[pos] = "r"
    return "".join(out)


def _future(mark: tuple[str, str], verb: str, script: int) -> str:
    """Verb + future suffix as one word, the way GPT spells it: karega,
    jaaunga, honge, denge (plain Roman reshapes the verb's ending;
    Devanagari and Rekhta just join: लाएँगे, lāeñge)."""
    suf = _FUTURE[mark[1]][script]
    if script == 1:
        if verb.endswith(("oon", "un")) and suf in ("ga", "gi"):
            return verb[:-3 if verb.endswith("oon") else -2] + "un" + suf    # jaaunga
        if verb.endswith("oon") and suf == "ge":
            return verb[:-3] + "on" + suf                                    # honge
        for end in ("yein", "yen", "ein", "en"):
            if verb.endswith(end):
                return verb[:-len(end)] + ("y" if end[0] == "y" else "") + "en" + suf   # denge
        if verb.endswith("aaye"):
            verb = verb[:-4] + "aae"                                         # jaaegi
    return verb + suf


def _clean(text: str) -> str:
    """NFC, single spaces, no space around line breaks."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    return re.sub(r" *\n *", "\n", text).strip()


def _line_key(line: str) -> str:
    """A line as the GPT verbatim-line table keys it."""
    return re.sub(r"\s+", " ", norm_urdu(unicodedata.normalize("NFC", line))).strip()


def _split(text: str) -> list[str]:
    """Urdu word runs (with any edge punctuation) and everything else, in order."""
    out, last = [], 0
    for m in _URDU_RUN.finditer(text):
        if m.start() > last:
            out.append(text[last:m.start()])
        out.append(m.group())
        last = m.end()
    if last < len(text):
        out.append(text[last:])
    return out


def _join(s: str, o: str, e: str) -> str:
    s = re.sub(f" ?{_O} ?", f"-{o}-", s)          # sher-o-shāyarī
    s = re.sub(f"{_IZAFAT} (?=\\S)", f"-{e}-", s)  # dil-e-nādāñ
    return s.replace(_IZAFAT, f"-{e}")


def _mukhtafi(key: str, rich: str) -> str:
    """Word-final silent he (ہ) is a short a in Urdu -- sabza, parda, jalva --
    though Hindi-derived readings often lengthen it (sabzā)."""
    if key.endswith("ہ") and not key.endswith("ھہ") and rich.endswith("ā"):
        return rich[:-1] + "a"
    return rich


def _ends_sentence(s: str) -> bool:
    return any(ch in _SENT_END for ch in s)


def _degeminate(s: str) -> str:
    return re.sub(r"([^aeiou])\1+", r"\1", s)


def _loose(s: str) -> str:
    s = s.lower().replace("w", "v").replace("ee", "i").replace("oo", "u")
    return re.sub(r"([aeiou])\1+", r"\1", s)


def _plain_to_rich(plain: str, key: str) -> str:
    """Last-resort R for a curated word the lexicon and model can't confirm."""
    r = plain.replace("aa", "ā").replace("ee", "ī").replace("oo", "ū").replace("w", "v")
    if key.endswith("ی") and r.endswith("i"):
        r = r[:-1] + "ī"
    elif key.endswith("ا") and r.endswith("a"):
        r = r[:-1] + "ā"
    elif key.endswith("ں") and r.endswith("n"):
        r = r[:-1] + "~"
    return r


_ENGINE: NeuralTransliterator | None = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> NeuralTransliterator:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = NeuralTransliterator()
        return _ENGINE


def transliterate(text: str) -> tuple[str, str, str]:
    """-> (devanagari, roman_plain, roman_diacritic)."""
    return get_engine().transliterate(text)
