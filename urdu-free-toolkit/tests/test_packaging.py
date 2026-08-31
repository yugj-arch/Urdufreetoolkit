from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_requirements_has_phase1_core():
    req = (ROOT / "requirements.txt").read_text().lower()
    for pkg in ("flask", "openai", "python-dotenv", "pillow", "pytest"):
        assert pkg in req


def test_gitignore_covers_jobs_and_caches():
    gi = (ROOT / ".gitignore").read_text()
    for pat in ("jobs/", "__pycache__/", ".pytest_cache/"):
        assert pat in gi
