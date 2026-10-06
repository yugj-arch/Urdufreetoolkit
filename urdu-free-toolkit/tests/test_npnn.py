# -*- coding: utf-8 -*-
"""The numpy runtime (``urdu_nn.npnn``) that runs our transliterators on the
deployed site, where there is no torch: it must read the same checkpoints and
give the same readings as torch does locally."""
from __future__ import annotations

import io
import pickle
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from urdu_nn import npnn

ROOT = Path(__file__).resolve().parents[1]
WORD_MODEL = ROOT / "data" / "translit_model" / "model.pt"
LINE_MODEL = ROOT / "data" / "rekhta_model" / "model.pt"
LINES = ["دل ہی تو ہے نہ سنگ و خشت درد سے بھر نہ آئے کیوں",
         "وسوسے دل میں نہ رکھ خوف رسن لے کے نہ چل", "محبت", "کتاب پڑھ رہا ہوں"]


def _needs(path: Path):
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} not present")


def test_backend_can_be_forced_to_numpy(monkeypatch):
    monkeypatch.setenv("URDU_NN_BACKEND", "numpy")
    assert npnn.use_torch() is False


def test_load_pt_refuses_anything_but_tensors():
    # a checkpoint is a pickle: only the three globals a state dict needs may load
    class Evil:
        def __reduce__(self):
            return (print, ("pwned",))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("m/data.pkl", pickle.dumps({"x": Evil()}, protocol=2))
    buf.seek(0)
    with pytest.raises(pickle.UnpicklingError, match="not allowed"):
        npnn.load_pt(buf)


def test_load_pt_matches_torch_load():
    torch = pytest.importorskip("torch")
    _needs(WORD_MODEL)
    a = torch.load(WORD_MODEL, map_location="cpu", weights_only=False)
    b = npnn.load_pt(WORD_MODEL)
    assert a["cfg"] == b["cfg"] and a["itos"] == b["itos"]
    assert a["state"].keys() == b["state"].keys()
    for k, v in a["state"].items():
        assert b["state"][k].dtype == np.float16
        assert np.array_equal(v.numpy(), b["state"][k]), k


def test_seq2seq_beam_search_matches_torch():
    torch = pytest.importorskip("torch")
    _needs(WORD_MODEL)
    from urdu_nn.model import Seq2Seq, beam_search, pad_batch
    ck = torch.load(WORD_MODEL, map_location="cpu", weights_only=False)
    tm = Seq2Seq(**ck["cfg"])
    tm.load_state_dict({k: v.float() for k, v in ck["state"].items()})
    tm.eval()
    nm = npnn.Seq2Seq(npnn.load_pt(WORD_MODEL)["state"], ck["cfg"])
    from urdu_nn.vocab import Vocab
    vocab = Vocab.from_itos(ck["itos"])
    seqs = [vocab.encode_src("j", w) for w in "محبت دل خوف رسن کتاب زندگی".split()]
    with torch.inference_mode():
        want = beam_search(tm, pad_batch(seqs), beam=5, max_len=64)
    got = npnn.beam_search(nm, seqs, beam=5, max_len=64)
    for w, g in zip(want, got):
        assert [ids for ids, _ in w] == [ids for ids, _ in g]
        assert np.allclose([s for _, s in w], [s for _, s in g], atol=1e-4)


@pytest.fixture(scope="module")
def teachers():
    pytest.importorskip("torch")
    pytest.importorskip("sentencepiece")
    from urdu_nn import rekhta_models as rm
    if not rm.model_dir("ur2hi"):
        pytest.skip("Rekhta models not downloaded (data/rekhta_models/)")
    from urdu_nn.rekhta_teacher import Teacher
    return Teacher("ur2hi", device="cpu"), rm.NumpyTeacher("ur2hi")


def test_numpy_teacher_reads_and_scores_like_torch(teachers):
    torch_t, np_t = teachers
    assert np_t(LINES) == torch_t(LINES)
    assert np_t(LINES)[1] == "वसवसे दिल में न रख ख़ौफ़-ए-रसन ले के न चल"   # the model card's line
    deva = torch_t(LINES)
    assert np.allclose(np_t.logprob(LINES, deva), torch_t.logprob(LINES, deva), atol=1e-4)


def test_rekhta_engine_on_numpy_matches_torch(monkeypatch):
    pytest.importorskip("torch")
    _needs(LINE_MODEL)
    import rekhta_translit as rt
    text = "\n".join(LINES)
    want = rt.RekhtaTransliterator(mode="student").transliterate_full(text)
    monkeypatch.setenv("URDU_NN_BACKEND", "numpy")
    eng = rt.RekhtaTransliterator(mode="student")
    assert eng.model.net.torch is False and eng.reader.word_model.net.torch is False
    assert eng.transliterate_full(text) == want


def test_engines_import_and_run_without_torch():
    # the Vercel build: torch can't even be imported
    _needs(LINE_MODEL)
    code = (
        "import sys; sys.modules['torch'] = None\n"
        "from providers.translit.rekhta import PROVIDER\n"
        "assert PROVIDER.available() == (True, ''), PROVIDER.available()\n"
        "import rekhta_translit as rt\n"
        "eng = rt.RekhtaTransliterator(mode='student')\n"
        "d, r, _ = eng.transliterate('محبت')\n"
        "assert d == 'मोहब्बत' and r == 'mohabbat', (d, r)   # Rekhta's spelling\n"
        "import neural_translit as nt\n"
        "assert nt.NeuralTransliterator().has_model\n"
    )
    res = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                         text=True, encoding="utf-8")
    assert res.returncode == 0, res.stderr[-2000:]
