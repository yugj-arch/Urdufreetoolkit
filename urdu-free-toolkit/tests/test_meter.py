# -*- coding: utf-8 -*-
"""Taqti: scanning readings and reading a misread line against its ghazal."""
from __future__ import annotations

import pytest

from urdu_nn.meter import Ghazal, scans, syllables

# Ghalib, "dil-e-nādāñ tujhe huā kyā hai": khafīf, fā'ilātun mafā'ilun fa'ilun
# (-u-- u-u- uu-), the first foot may be fa'ilātun (uu--), the last fa'lun (--)
KHAFIF = ["-u--u-u-uu-", "uu--u-u-uu-", "-u--u-u---", "uu--u-u---"]
LINES = [
    "dil-e-nādā~ tujhe huā kyā hai",
    "āxir is dard kī davā kyā hai",
    "ham hai~ muśtāq aur vo be-zār",
    "yā ilāhī ye mājrā kyā hai",
    "mai~ bhī mu~h me~ zabān rakhtā hū~",
    "kāś pūcho ki muddaā kyā hai",
    "phir ye haṅgāma ai xudā kyā hai",
    "abr kyā cīz hai havā kyā hai",
]


@pytest.mark.parametrize("line", LINES)
def test_every_misra_scans_in_its_meter(line):
    assert any(scans(syllables(line), m) for m in KHAFIF)


def test_syllable_rules():
    assert syllables("yār") == ["-", "x"]                  # long vowel + consonant: - u, the u dropped at the end
    assert syllables("dil-e-gul") == ["u", "-u", "-", ]    # di-le-gul: the l carries over to the izafat
    assert syllables("ko") == ["-u"]                       # a final long vowel may be short
    assert syllables("tohfa") == ["-", "-u"]               # तोहफ़ा is tuh-fa


def test_a_misread_izafat_doesnt_fit_its_ghazal():
    g = Ghazal([syllables(l) for l in LINES])
    right = syllables("phir ye haṅgāma ai xudā kyā hai")
    wrong = syllables("phir ye haṅgāma-e-ai xudā kyā hai")
    assert g.support(right, 6) > g.support(wrong, 6) + 3


def test_engine_rereads_a_line_by_the_meter(monkeypatch):
    import rekhta_translit as rt
    eng = rt.RekhtaTransliterator.__new__(rt.RekhtaTransliterator)
    from urdu_nn.rekhta_roman import Reader
    eng.reader, eng.model, eng.use_meter = Reader(), None, True
    # the readings are stood in for by their R lines; line 6 has a stray izafat
    rich = dict(enumerate(LINES))
    rich["bad"] = "phir ye haṅgāma-e-ai xudā kyā hai"
    devas = list(range(len(LINES)))
    devas[6] = "bad"
    monkeypatch.setattr(eng, "rich_line", lambda run, d: rich[d])
    monkeypatch.setattr(eng, "_izafat_toggles", lambda run, d: [6] if d == "bad" else [])
    monkeypatch.setattr(eng, "_scorer", lambda: (lambda runs, ds: [-0.1] * len(ds)))
    monkeypatch.setattr(rt, "_lines_up", lambda run, d: True)
    monkeypatch.setattr(rt, "_clean", lambda d: d)
    import urdu_nn.rekhta_lexicon as rl
    monkeypatch.setattr(rl, "deva_fold", lambda d, izafat=True: "same")
    out = eng.by_meter([f"run{i}" for i in devas], devas)
    assert out == list(range(len(LINES)))                  # only line 6 changes, to its toggle
